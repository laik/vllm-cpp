#pragma once
#include "config.h"
#include "kv_cache_manager.h"
#include <vector>
#include <cstdint>

// ============================================================
// PagedAttention: vLLM-style block-based KV cache attention
//
// During decoding, each request's KV cache is stored in non-contiguous
// physical blocks. PagedAttention gathers from these scattered blocks
// to compute attention in a single pass.
//
// Key insight: Instead of materializing a contiguous KV cache, we
// index into blocks directly during the Q@K^T and softmax(V) steps.
// ============================================================

// Paged attention kernel: compute attention output for a single token
//
// Parameters:
//   q           - Query tensor for the current token [num_heads * head_dim]
//   block_table - Physical block IDs for this request's sequence
//   seq_len     - Total sequence length (prompt + generated)
//   kv_manager  - Block pool containing KV cache data
//   num_heads    - Total attention heads
//   num_kv_heads - KV heads (GQA grouping)
//   head_dim   - Dimension per head
//   layer      - Transformer layer index
//   out         - Output buffer [num_heads * head_dim]
void paged_attention(
    const float* q,
    const std::vector<int32_t>& block_table,
    int32_t seq_len,
    const BlockPool& kv_manager,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t layer,
    float* out
);

// Batch paged attention: process multiple requests in one call
//
// Each request in the batch has its own block table and sequence length.
// Results are written to separate output buffers.
void paged_attention_batch(
    const std::vector<const float*>& q_ptrs,      // [batch_size]
    const std::vector<const std::vector<int32_t>*>& block_tables, // [batch_size]
    const std::vector<int32_t>& seq_lens,   // [batch_size]
    const BlockPool& kv_manager,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t layer,
    std::vector<float*> out_ptrs            // [batch_size]
);

// Write KV into the block pool for a new token (decode step)
void write_kv_token(
    KVCacheManager& kv_manager,
    const std::vector<int32_t>& block_table,
    int32_t seq_len,
    int32_t layer,
    const float* k_data,  // [num_kv_heads * head_dim]
    const float* v_data   // [num_kv_heads * head_dim]
);

// Write KV into the block pool for prefill (all positions at once)
void write_kv_prefill(
    KVCacheManager& kv_manager,
    const std::vector<int32_t>& block_table,
    int32_t seq_len,
    int32_t layer,
    const float* k_data,  // [seq_len * num_kv_heads * head_dim]
    const float* v_data   // [seq_len * num_kv_heads * head_dim]
);

// Read KV cache from block pool into contiguous buffers (for prefill attention)
void read_kv_prefill(
    const BlockPool& kv_manager,
    const std::vector<int32_t>& block_table,
    int32_t seq_len,
    int32_t layer,
    int32_t num_kv_heads,
    int32_t head_dim,
    float* k_out,  // [seq_len * num_kv_heads * head_dim]
    float* v_out   // [seq_len * num_kv_heads * head_dim]
);
