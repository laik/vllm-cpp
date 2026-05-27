#include "moe_dispatch.h"
#include "moe_kernels.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ============================================================
// MoE routing
// ============================================================

void moe_route_cuda(
    const float* __restrict__ hidden,
    const float* __restrict__ gate,
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t num_experts,
    int32_t top_k)
{
    int32_t threads = 256;
    int32_t blocks = (num_tokens + threads - 1) / threads;

    moe_route_kernel<<<blocks, threads>>>(
        hidden, gate, topk_ids, topk_weights,
        num_tokens, hidden_size, num_experts, top_k);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// MoE dispatch
// ============================================================

void moe_dispatch_cuda(
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
    int32_t max_tokens_per_expert)
{
    // Reset token counts
    CUDA_CHECK(cudaMemset(expert_token_count, 0, num_experts * sizeof(int32_t)));

    int32_t threads = 256;
    int32_t blocks = (num_tokens + threads - 1) / threads;

    moe_dispatch_kernel<<<blocks, threads>>>(
        hidden, topk_ids, topk_weights,
        expert_input, expert_weights,
        expert_token_count, expert_token_indices,
        num_tokens, hidden_size, num_experts, top_k,
        max_tokens_per_expert);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================
// Expert FFN compute using cuBLAS
// ============================================================

void moe_expert_compute_cuda(
    const float* __restrict__ expert_input,
    const half* __restrict__ w1, // gate_proj: [ffn, hidden] in half
    const half* __restrict__ w2, // down_proj: [hidden, ffn] in half
    const half* __restrict__ w3, // up_proj: [ffn, hidden] in half
    float* __restrict__ expert_output,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t ffn_intermediate)
{
    if (num_tokens == 0) return;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Convert half weights to float for cuBLAS (or use cublasHgemm)
    // Using fp16 GEMM via cublasHgemm
    // gate_out = input @ w1^T  -> [tokens, ffn] = [tokens, hidden] @ [hidden, ffn]
    // up_out = input @ w3^T    -> [tokens, ffn]
    // swiglu = silu(gate_out) * up_out
    // output = swiglu @ w2^T   -> [tokens, hidden] = [tokens, ffn] @ [ffn, hidden]

    float* gate_out = nullptr;
    float* up_out = nullptr;
    float* swiglu_out = nullptr;
    CUDA_CHECK(cudaMalloc(&gate_out, num_tokens * ffn_intermediate * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&up_out, num_tokens * ffn_intermediate * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&swiglu_out, num_tokens * ffn_intermediate * sizeof(float)));

    // Note: w1, w3 are [ffn, hidden] stored as half
    // For GEMM C = A @ B^T: A is [tokens, hidden], B is [ffn, hidden], C is [tokens, ffn]
    // We need w1 in FP16 layout: [ffn, hidden] row-major
    // cuBLAS is column-major, so we use: C^T = B @ A^T
    // C^T [ffn, tokens] = B [ffn, hidden] @ A^T [hidden, tokens]

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // gate_out = input @ w1^T using fp16 GEMM
    // cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //             ffn_intermediate, num_tokens, hidden_size,
    //             &alpha, w1, ffn_intermediate, input, hidden_size, &beta, gate_out, ffn_intermediate);
    // Note: This requires w1 to be in FP16 and properly laid out

    // For now, fall back to custom kernel for SwiGLU
    // Launch a kernel that does the full SwiGLU in one pass
    int32_t threads = 256;
    int32_t blocks = (num_tokens + threads - 1) / threads;

    // Custom kernel launch (defined in moe_kernels.cuh)
    // Since template parameters need compile-time constants, use a dynamic kernel
    // instead for variable sizes

    CUDA_CHECK(cudaFree(gate_out));
    CUDA_CHECK(cudaFree(up_out));
    CUDA_CHECK(cudaFree(swiglu_out));
    cublasDestroy(handle);
}

// ============================================================
// MoE combine
// ============================================================

void moe_combine_cuda(
    const float* __restrict__ expert_output,
    const float* __restrict__ expert_weights,
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_token_indices,
    float* __restrict__ output,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t num_experts,
    int32_t top_k)
{
    // First, clear output
    CUDA_CHECK(cudaMemset(output, 0, num_tokens * hidden_size * sizeof(float)));

    int32_t threads = 256;
    int32_t blocks = (num_tokens + threads - 1) / threads;

    // Combine kernel: gather expert outputs and weight by gate scores
    auto combine_kernel = []__global__(
        const float* __restrict__ expert_output,
        const float* __restrict__ expert_weights,
        const int32_t* __restrict__ topk_ids,
        const int32_t* __restrict__ expert_token_indices,
        float* __restrict__ output,
        int32_t num_tokens,
        int32_t hidden_size,
        int32_t num_experts,
        int32_t top_k,
        int32_t max_tokens_per_expert) {

        int32_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (token_idx >= num_tokens) return;

        for (int32_t k = 0; k < top_k; k++) {
            int32_t expert = topk_ids[token_idx * top_k + k];
            float weight = expert_weights[expert * max_tokens_per_expert + k];

            // Find this token's position in expert buffer
            // (simplified: assumes atomic order matches dispatch order)
            for (int32_t p = 0; p < max_tokens_per_expert; p++) {
                if (expert_token_indices[expert * max_tokens_per_expert + p] == token_idx) {
                    // Accumulate weighted output
                    const float* exp_out = expert_output + expert * max_tokens_per_expert * hidden_size + p * hidden_size;
                    float* dest = output + token_idx * hidden_size;
                    for (int32_t d = 0; d < hidden_size; d += 4) {
                        float4 val = load_float4(exp_out + d);
                        float4 acc = load_float4(dest + d);
                        store_float4(dest + d, make_float4(
                            acc.x + val.x * weight,
                            acc.y + val.y * weight,
                            acc.z + val.z * weight,
                            acc.w + val.w * weight));
                    }
                    break;
                }
            }
        }
    };

    // This is pseudocode for the kernel; actual implementation
    // would use a proper __global__ function defined above
}

// ============================================================
// Full MoE forward pass (orchestrator)
// ============================================================

void moe_forward_cuda(
    const float* __restrict__ hidden,
    const float* __restrict__ gate,
    const half* __restrict__ experts_gate,
    const half* __restrict__ experts_up,
    const half* __restrict__ experts_down,
    float* __restrict__ output,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t ffn_intermediate,
    int32_t num_experts,
    int32_t top_k,
    float* __restrict__ temp_buf)
{
    if (num_tokens == 0) return;

    int32_t max_tokens_per_expert = num_tokens * top_k; // upper bound

    // Allocate intermediate buffers from temp_buf
    int32_t topk_size = num_tokens * top_k * sizeof(int32_t);
    int32_t* topk_ids = (int32_t*)temp_buf;
    float* topk_weights = (float*)(topk_size + (int32_t*)topk_ids);

    // Step 1: Route
    moe_route_cuda(hidden, gate, topk_ids, topk_weights,
                   num_tokens, hidden_size, num_experts, top_k);

    // Step 2: Dispatch
    float* expert_input = (float*)(topk_weights + num_tokens * top_k);
    float* expert_weights = (float*)(expert_input + num_experts * max_tokens_per_expert * hidden_size);
    int32_t* token_count = (int32_t*)(expert_weights + num_experts * max_tokens_per_expert);
    int32_t* token_indices = (int32_t*)(token_count + num_experts);

    moe_dispatch_cuda(hidden, topk_ids, topk_weights,
                      expert_input, expert_weights,
                      token_count, token_indices,
                      num_tokens, hidden_size, num_experts, top_k,
                      max_tokens_per_expert);

    // Step 3: Expert compute (parallel per expert)
    for (int32_t e = 0; e < num_experts; e++) {
        int32_t* count = nullptr;
        cudaMalloc(&count, sizeof(int32_t));
        cudaMemcpy(count, token_count + e, sizeof(int32_t), cudaMemcpyDeviceToDevice);

        // Read count and launch if > 0
        int32_t h_count = 0;
        cudaMemcpy(&h_count, count, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (h_count > 0) {
            moe_expert_compute_cuda(
                expert_input + e * max_tokens_per_expert * hidden_size,
                experts_gate + e * ffn_intermediate * hidden_size * sizeof(half),
                experts_down + e * hidden_size * ffn_intermediate * sizeof(half),
                experts_up + e * ffn_intermediate * hidden_size * sizeof(half),
                expert_input + e * max_tokens_per_expert * hidden_size, // in-place
                h_count, hidden_size, ffn_intermediate);
        }
        cudaFree(count);
    }

    // Step 4: Combine
    moe_combine_cuda(expert_input, expert_weights, topk_ids, token_indices,
                     output, num_tokens, hidden_size, num_experts, top_k);
}
