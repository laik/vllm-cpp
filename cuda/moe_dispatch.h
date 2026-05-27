#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================
// MoE (Mixture of Experts) dispatch + compute + combine
// ============================================================
//
// Architecture: Qwen3.6 uses MoE with expert routing.
// For each token:
//   1. Gate: compute routing scores for all experts
//   2. Top-K: select K experts with highest scores
//   3. Dispatch: scatter tokens to selected experts
//   4. Expert compute: each expert processes its assigned tokens (SwiGLU FFN)
//   5. Combine: weighted sum of expert outputs using gate scores
//
// Memory layout:
//   hidden:     [batch * seq, hidden_size]
//   gate:       [hidden_size, num_experts]
//   experts:    [num_experts][hidden_size, ffn_intermediate] * 3 (gate, up, down)
//   output:     [batch * seq, hidden_size]
//
// Optimizations:
//   - Use cuBLAS for GEMM operations in expert FFN
//   - Shared memory for gate softmax reduction
//   - Atomic operations for expert token counting
//   - Dispatch via scatter-gather with index buffers
// ============================================================

void moe_forward_cuda(
    const float* __restrict__ hidden,
    const float* __restrict__ gate,
    const half* __restrict__ experts_gate,   // [num_experts][ffn, hidden]
    const half* __restrict__ experts_up,     // [num_experts][ffn, hidden]
    const half* __restrict__ experts_down,   // [num_experts][hidden, ffn]
    float* __restrict__ output,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t ffn_intermediate,
    int32_t num_experts,
    int32_t top_k,
    float* __restrict__ temp_buf
);

// Expert routing: compute top-k experts and their scores
void moe_route_cuda(
    const float* __restrict__ hidden,
    const float* __restrict__ gate,
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t num_experts,
    int32_t top_k
);

// Dispatch tokens to expert buffers
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
    int32_t max_tokens_per_expert
);

// Expert FFN compute (SwiGLU)
void moe_expert_compute_cuda(
    const float* __restrict__ expert_input,
    const half* __restrict__ w1,  // gate_proj: [ffn, hidden]
    const half* __restrict__ w2,  // down_proj: [hidden, ffn]
    const half* __restrict__ w3,  // up_proj: [ffn, hidden]
    float* __restrict__ expert_output,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t ffn_intermediate
);

// Combine expert outputs
void moe_combine_cuda(
    const float* __restrict__ expert_output,
    const float* __restrict__ expert_weights,
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_token_indices,
    float* __restrict__ output,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t num_experts,
    int32_t top_k
);
