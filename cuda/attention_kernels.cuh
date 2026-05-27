#pragma once
#include "utils.cuh"

// ============================================================
// PagedAttention kernel: each thread block handles one query head
// Uses shared memory for softmax reduction
// ============================================================

template<int32_t BLOCK_SIZE>
__global__ void paged_attention_kernel(
    const float* __restrict__ q,
    const half* __restrict__ kv_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    float* __restrict__ out,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t batch_size
) {
    int32_t head_idx = blockIdx.x;
    int32_t batch_idx = blockIdx.y;
    if (head_idx >= num_heads || batch_idx >= batch_size) return;

    int32_t kv_head = head_idx / (num_heads / num_kv_heads);
    int32_t seq_len = context_lens[batch_idx];
    const int32_t* blocks = block_tables[batch_idx * (gridDim.x > 1 ? batch_size : 1) + batch_idx];

    // Shared memory for softmax
    extern __shared__ char shared_mem[];
    float* logits = (float*)shared_mem;
    float* weights = (float*)(shared_mem + seq_len * sizeof(float));

    float scale = 1.0f / sqrtf((float)head_dim);

    // Each thread handles a subset of head_dim
    int32_t tid = threadIdx.x;
    int32_t total_threads = blockDim.x;

    // Initialize output accumulator
    float acc[8] = {0};

    // Process one KV position per thread (up to seq_len positions)
    for (int32_t s = tid; s < seq_len; s += total_threads) {
        // Find block containing position s
        int32_t block_idx = s / BLOCK_SIZE;
        int32_t pos_in_block = s % BLOCK_SIZE;
        int32_t physical_block = blocks[block_idx];

        // Load key from KV cache
        const half* k_ptr = kv_cache +
            physical_block * BLOCK_SIZE * num_kv_heads * head_dim +
            pos_in_block * num_kv_heads * head_dim +
            kv_head * head_dim;

        // Compute q . k
        float dot = 0.0f;
        for (int32_t d = 0; d < head_dim; d += 4) {
            float4 q_vec = load_float4(q + head_idx * head_dim + d);
            float4 k_vec = load_half4(k_ptr + d);
            dot += q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                   q_vec.z * k_vec.z + q_vec.w * k_vec.w;
        }
        logits[s] = dot * scale;
    }
    __syncthreads();

    // Softmax: find max
    float max_logit = -1e9f;
    for (int32_t s = tid; s < seq_len; s += total_threads) {
        max_logit = fmaxf(max_logit, logits[s]);
    }
    // Block reduce max
    for (int32_t stride = total_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) max_logit = fmaxf(max_logit, max_logit + stride < seq_len ? logits[max_logit + stride] : -1e9f);
        __syncthreads();
    }
    // Simplified: just do the max in shared memory
    if (tid == 0) {
        for (int32_t s = 1; s < seq_len; s++)
            max_logit = fmaxf(max_logit, logits[s]);
    }
    __syncthreads();

    // Softmax: compute exp and sum
    float sum_exp = 0.0f;
    for (int32_t s = tid; s < seq_len; s += total_threads) {
        weights[s] = expf(logits[s] - max_logit);
        sum_exp += weights[s];
    }
    // Block reduce sum
    for (int32_t stride = total_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sum_exp += sum_exp + stride < seq_len ? 0.0f : 0.0f;
        __syncthreads();
    }
    if (tid == 0) {
        for (int32_t s = 1; s < seq_len; s++)
            sum_exp += weights[s];
        if (sum_exp > 0.0f) {
            for (int32_t s = 0; s < seq_len; s++)
                weights[s] /= sum_exp;
        }
    }
    __syncthreads();

    // Weighted sum of V
    for (int32_t d = tid; d < head_dim; d += total_threads) {
        float sum = 0.0f;
        for (int32_t s = 0; s < seq_len; s++) {
            int32_t block_idx = s / BLOCK_SIZE;
            int32_t pos_in_block = s % BLOCK_SIZE;
            int32_t physical_block = blocks[block_idx];

            const half* v_ptr = kv_cache +
                physical_block * BLOCK_SIZE * num_kv_heads * head_dim +
                pos_in_block * num_kv_heads * head_dim +
                kv_head * head_dim + d;

            sum += weights[s] * half2float(*v_ptr);
        }
        out[head_idx * head_dim + d] = sum;
    }
}

// ============================================================
// Flash-style attention kernel (contiguous KV, tiled)
// ============================================================

template<int32_t BLOCK_M, int32_t BLOCK_N, int32_t TILE_D>
__global__ void flash_attn_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int32_t num_heads,
    int32_t head_dim,
    int32_t q_len,
    int32_t kv_len,
    int32_t batch_idx
) {
    // Each block handles one head
    int32_t head_idx = blockIdx.x;
    if (head_idx >= num_heads) return;

    extern __shared__ char shared[];
    float* s_q = (float*)shared;
    float* s_k = s_q + BLOCK_M * TILE_D;
    float* s_v = s_k + BLOCK_N * TILE_D;
    float* s_o = s_v + BLOCK_N * TILE_D;
    float* s_l = s_o + BLOCK_M * TILE_D;
    float* s_m = s_l + BLOCK_M;

    int32_t tid = threadIdx.x;
    int32_t q_offset = (head_idx * q_len + batch_idx * q_len * num_heads) * head_dim;
    int32_t k_offset = (head_idx * kv_len) * head_dim;
    int32_t v_offset = k_offset; // same layout for v

    // Initialize output
    for (int32_t i = tid; i < BLOCK_M * TILE_D; i += blockDim.x) s_o[i] = 0.0f;
    for (int32_t i = tid; i < BLOCK_M; i += blockDim.x) { s_m[i] = -1e9f; s_l[i] = 0.0f; }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)head_dim);

    // Load Q tile
    for (int32_t i = tid; i < BLOCK_M * TILE_D; i += blockDim.x) {
        int32_t q_row = i / TILE_D;
        int32_t q_col = i % TILE_D;
        if (q_row < BLOCK_M && q_col < head_dim) {
            s_q[i] = q[q_offset + q_row * head_dim + q_col];
        }
    }
    __syncthreads();

    // Iterate over K/V tiles
    for (int32_t kv_start = 0; kv_start < kv_len; kv_start += BLOCK_N) {
        int32_t kv_end = min(kv_start + BLOCK_N, kv_len);
        int32_t kv_tiles = kv_end - kv_start;

        // Load K tile
        for (int32_t i = tid; i < kv_tiles * TILE_D; i += blockDim.x) {
            int32_t k_row = i / TILE_D + kv_start;
            int32_t k_col = i % TILE_D;
            if (k_col < head_dim) {
                s_k[i] = k[k_offset + k_row * head_dim + k_col];
            }
        }
        // Load V tile
        for (int32_t i = tid; i < kv_tiles * TILE_D; i += blockDim.x) {
            int32_t v_row = i / TILE_D + kv_start;
            int32_t v_col = i % TILE_D;
            if (v_col < head_dim) {
                s_v[i] = v[v_offset + v_row * head_dim + v_col];
            }
        }
        __syncthreads();

        // S = Q @ K^T for this tile
        for (int32_t q_row = tid; q_row < BLOCK_M; q_row += blockDim.x) {
            for (int32_t kv_row = 0; kv_row < kv_tiles; kv_row++) {
                float dot = 0.0f;
                for (int32_t d = 0; d < head_dim; d++) {
                    dot += s_q[q_row * TILE_D + d] * s_k[kv_row * TILE_D + d];
                }
                // Atomic max for m
                float new_m = fmaxf(s_m[q_row], dot * scale);
                float scale_factor = expf(s_m[q_row] - new_m);
                s_l[q_row] = s_l[q_row] * scale_factor + expf(dot * scale - new_m);
                s_m[q_row] = new_m;
            }
        }
        __syncthreads();

        // Update O = l * O * scale_factor + S^T @ V
        for (int32_t q_row = tid; q_row < BLOCK_M; q_row += blockDim.x) {
            float scale_factor = expf(s_m[q_row]); // simplified
            for (int32_t d = 0; d < TILE_D; d++) {
                float sum = 0.0f;
                for (int32_t kv_row = 0; kv_row < kv_tiles; kv_row++) {
                    // S[q_row][kv_row] * V[kv_row][d]
                    // Need to recompute S since we didn't store it
                    float s_val = 0.0f;
                    for (int32_t dd = 0; dd < head_dim; dd++) {
                        s_val += s_q[q_row * TILE_D + dd] * s_k[kv_row * TILE_D + dd];
                    }
                    sum += s_val * s_v[kv_row * TILE_D + d];
                }
                s_o[q_row * TILE_D + d] = s_o[q_row * TILE_D + d] * scale_factor + sum;
            }
        }
        __syncthreads();
    }

    // Write output
    for (int32_t i = tid; i < BLOCK_M * TILE_D; i += blockDim.x) {
        int32_t o_row = i / TILE_D;
        int32_t o_col = i % TILE_D;
        if (o_row < BLOCK_M && o_col < head_dim) {
            int32_t out_offset = (head_idx * q_len + batch_idx * q_len * num_heads) * head_dim;
            out[out_offset + o_row * head_dim + o_col] = s_o[i];
        }
    }
}

// ============================================================
// RoPE kernel: apply rotary position embeddings
// ============================================================

__global__ void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int32_t total_dim,
    int32_t head_dim,
    int32_t num_heads,
    int32_t batch_size
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total = batch_size * total_dim;
    if (idx >= total) return;

    int32_t b = idx / total_dim;
    int32_t d = idx % total_dim;

    int32_t head = d / head_dim;
    int32_t hdim = d % head_dim;

    if (hdim >= head_dim / 2) return; // Only process first half

    int32_t freq_idx = hdim;
    float c = cos[freq_idx];
    float s = sin[freq_idx];

    int32_t i0 = d;
    int32_t i1 = d + (head_dim / 2);
    if (i1 >= total_dim) i1 = d; // Wrap for odd dimensions

    float q0 = q[b * total_dim + i0];
    float q1 = q[b * total_dim + i1];
    float k0 = k[b * total_dim + i0];
    float k1 = k[b * total_dim + i1];

    q[b * total_dim + i0] = q0 * c - q1 * s;
    q[b * total_dim + i1] = q0 * s + q1 * c;
    k[b * total_dim + i0] = k0 * c - k1 * s;
    k[b * total_dim + i1] = k0 * s + k1 * c;
}
