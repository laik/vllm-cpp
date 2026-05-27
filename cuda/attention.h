#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================
// PagedAttention kernel declarations
// ============================================================

// PagedAttention: attention with block-managed KV cache
//
// q:       [num_heads, head_dim]
// kv_cache: [num_blocks, block_size, num_kv_heads, head_dim]
// block_tables: [batch_size, max_num_blocks_per_seq]
// seq_lens:    [batch_size]
// attn_weights: [num_heads, total_seq_len] (output)
//
// Supports GQA (num_heads > num_kv_heads) via head grouping.

void paged_attention_cuda(
    const float* __restrict__ q,
    const half* __restrict__ kv_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    float* __restrict__ out,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t block_size,
    int32_t batch_size
);

// Standard scaled dot-product attention (non-paged, contiguous KV)
void flash_attention_cuda(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    float* __restrict__ attn_weights,
    int32_t batch_size,
    int32_t num_heads,
    int32_t head_dim,
    int32_t q_len,
    int32_t kv_len,
    const float* __restrict__ attn_mask
);

// Apply RoPE (Rotary Position Embedding) in-place
void apply_rope_cuda(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int32_t total_dim,
    int32_t head_dim,
    int32_t batch_size
);

// Head-dimension normalization (Qwen3 Q/K norm)
void head_norm_cuda(
    float* __restrict__ x,
    const float* __restrict__ weight,
    float eps,
    int32_t num_heads,
    int32_t head_dim,
    int32_t batch_size
);
