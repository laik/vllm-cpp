#pragma once
#include "utils.cuh"

// ============================================================
// MoE routing kernel: find top-k experts per token
// ============================================================

__global__ void moe_route_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ gate,
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t num_experts,
    int32_t top_k
) {
    int32_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    // Compute gate scores: hidden @ gate^T
    float scores[36]; // max experts = 36 for Qwen3
    for (int32_t e = 0; e < num_experts; e++) {
        float sum = 0.0f;
        for (int32_t d = 0; d < hidden_size; d += 4) {
            float4 h = load_float4(hidden + token_idx * hidden_size + d);
            float4 g = load_float4(gate + e * hidden_size + d);
            sum += h.x * g.x + h.y * g.y + h.z * g.z + h.w * g.w;
        }
        scores[e] = sum;
    }

    // Softmax over experts
    float max_s = scores[0];
    for (int32_t e = 1; e < num_experts; e++) max_s = fmaxf(max_s, scores[e]);
    float sum_e = 0.0f;
    for (int32_t e = 0; e < num_experts; e++) {
        scores[e] = expf(scores[e] - max_s);
        sum_e += scores[e];
    }
    for (int32_t e = 0; e < num_experts; e++) scores[e] /= (sum_e + 1e-9f);

    // Select top-k via partial selection (bubble sort for small k)
    for (int32_t k = 0; k < top_k; k++) {
        int32_t best = k;
        for (int32_t e = k + 1; e < num_experts; e++) {
            if (scores[e] > scores[best]) best = e;
        }
        // Swap
        if (best != k) {
            float tmp_s = scores[k]; scores[k] = scores[best]; scores[best] = tmp_s;
            // We need to track original indices, so store (score, original_idx) pairs
        }
        topk_ids[token_idx * top_k + k] = best;
        topk_weights[token_idx * top_k + k] = scores[k];
    }
}

// ============================================================
// MoE dispatch kernel: scatter tokens to expert buffers
// ============================================================

__global__ void moe_dispatch_kernel(
    const float* __restrict__ hidden,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    float* __restrict__ expert_input,
    float* __restrict__ expert_weights,
    int32_t* __restrict__ expert_token_count,
    int32_t* __restrict__ expert_token_indices,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t num_experts,
    int32_t top_k,
    int32_t max_tokens_per_expert
) {
    int32_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    for (int32_t k = 0; k < top_k; k++) {
        int32_t expert = topk_ids[token_idx * top_k + k];
        float weight = topk_weights[token_idx * top_k + k];

        // Atomic increment to get position in expert buffer
        int32_t pos = atomicAdd(expert_token_count + expert, 1);
        if (pos >= max_tokens_per_expert) continue;

        // Store token index for combine phase
        expert_token_indices[expert * max_tokens_per_expert + pos] = token_idx;

        // Scatter hidden state
        float* dest = expert_input + expert * max_tokens_per_expert * hidden_size + pos * hidden_size;
        const float* src = hidden + token_idx * hidden_size;
        for (int32_t d = 0; d < hidden_size; d += 4) {
            float4 val = load_float4(src + d);
            store_float4(dest + d, val);
        }

        // Store weight
        expert_weights[expert * max_tokens_per_expert + pos] = weight;
    }
}

// ============================================================
// Expert FFN kernel: SwiGLU compute for one expert
// ============================================================

template<int32_t HIDDEN_SIZE, int32_t FFN_SIZE>
__global__ void expert_ffn_kernel(
    const float* __restrict__ input,
    const float* __restrict__ w_gate, // [ffn, hidden]
    const float* __restrict__ w_up,   // [ffn, hidden]
    const float* __restrict__ w_down, // [hidden, ffn]
    float* __restrict__ output,
    int32_t num_tokens
) {
    int32_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    // Gate projection: input @ w_gate^T -> [ffn]
    float gate_out[FFN_SIZE];
    float up_out[FFN_SIZE];

    #pragma unroll
    for (int32_t i = 0; i < FFN_SIZE; i++) {
        float gs = 0.0f, us = 0.0f;
        #pragma unroll
        for (int32_t d = 0; d < HIDDEN_SIZE; d++) {
            gs += input[token_idx * HIDDEN_SIZE + d] * w_gate[i * HIDDEN_SIZE + d];
            us += input[token_idx * HIDDEN_SIZE + d] * w_up[i * HIDDEN_SIZE + d];
        }
        gate_out[i] = gs;
        up_out[i] = us;
    }

    // SwiGLU: silu(gate) * up
    float swiglu_out[FFN_SIZE];
    #pragma unroll
    for (int32_t i = 0; i < FFN_SIZE; i++) {
        float g = gate_out[i];
        swiglu_out[i] = (g / (1.0f + expf(-g))) * up_out[i];
    }

    // Down projection: swiglu_out @ w_down^T -> [hidden]
    #pragma unroll
    for (int32_t d = 0; d < HIDDEN_SIZE; d++) {
        float sum = 0.0f;
        #pragma unroll
        for (int32_t i = 0; i < FFN_SIZE; i++) {
            sum += swiglu_out[i] * w_down[d * FFN_SIZE + i];
        }
        output[token_idx * HIDDEN_SIZE + d] = sum;
    }
}

// ============================================================
// MoE combine kernel: gather expert outputs and weighted sum
// ============================================================

__global__ void moe_combine_kernel(
    const float* __restrict__ expert_output,
    const float* __restrict__ expert_weights,
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_token_indices,
    float* __restrict__ output,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t num_experts,
    int32_t top_k,
    int32_t max_tokens_per_expert
) {
    int32_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    // For each top-k expert, find this token's position and add weighted output
    for (int32_t k = 0; k < top_k; k++) {
        int32_t expert = topk_ids[token_idx * top_k + k];
        float weight = expert_weights[expert * max_tokens_per_expert + /* need to find pos */ 0];

        // Find position: scan expert_token_indices
        int32_t pos = -1;
        for (int32_t p = 0; p < max_tokens_per_expert; p++) {
            if (expert_token_indices[expert * max_tokens_per_expert + p] == token_idx) {
                pos = p;
                break;
            }
        }
        if (pos < 0) continue;

        float w = expert_weights[expert * max_tokens_per_expert + pos];
        const float* src = expert_output + expert * max_tokens_per_expert * hidden_size + pos * hidden_size;
        float* dest = output + token_idx * hidden_size;

        for (int32_t d = 0; d < hidden_size; d += 4) {
            float4 val = load_float4(src + d);
            float4 acc = load_float4(dest + d);
            store_float4(dest + d, make_float4(
                acc.x + val.x * w, acc.y + val.y * w,
                acc.z + val.z * w, acc.w + val.w * w));
        }
    }
}

// ============================================================
// RMSNorm kernel (for use in both attention and MoE paths)
// ============================================================

__global__ void rms_norm_kernel(
    float* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int32_t dim,
    float eps,
    int32_t num_tokens
) {
    int32_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    // Compute RMS
    float sum_sq = 0.0f;
    for (int32_t d = 0; d < dim; d += 4) {
        float4 v = load_float4(input + token_idx * dim + d);
        sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    float rms = sqrtf(sum_sq / dim + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale
    for (int32_t d = 0; d < dim; d++) {
        out[token_idx * dim + d] = weight[d] * input[token_idx * dim + d] * inv_rms;
    }
}
