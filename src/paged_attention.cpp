#include "paged_attention.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>

// ============================================================
// paged_attention: Compute attention using scattered block tables
// ============================================================

void paged_attention(
    const float* q,
    const std::vector<int32_t>& block_table,
    int32_t seq_len,
    const BlockPool& kv_manager,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t layer,
    float* out)
{
    int32_t heads_per_group = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt((float)head_dim);
    int32_t kv_dim = kv_manager.kv_dim();

    for (int32_t h = 0; h < num_heads; h++) {
        int32_t kv_h = h / heads_per_group;
        const float* q_h = q + h * head_dim;

        float* o_h = out + h * head_dim;
        std::fill(o_h, o_h + head_dim, 0.0f);

        // Pass 1: find max logit
        float max_logit = -1e9f;
        for (int32_t p = 0; p < seq_len; p++) {
            int32_t block_idx = p / BLOCK_SIZE;
            int32_t pos_in_block = p % BLOCK_SIZE;
            if ((size_t)block_idx >= block_table.size()) continue;
            int32_t block_id = block_table[block_idx];

            const float* k_h = kv_manager.get_k(block_id, layer);
            if (!k_h) continue;
            k_h += pos_in_block * kv_dim + kv_h * head_dim;

            float dot = 0.0f;
            for (int32_t d = 0; d < head_dim; d++) dot += q_h[d] * k_h[d];
            float logit = dot * scale;
            if (logit > max_logit) max_logit = logit;
        }

        // Pass 2: softmax sum
        float sum_exp = 0.0f;
        for (int32_t p = 0; p < seq_len; p++) {
            int32_t block_idx = p / BLOCK_SIZE;
            int32_t pos_in_block = p % BLOCK_SIZE;
            if ((size_t)block_idx >= block_table.size()) continue;
            int32_t block_id = block_table[block_idx];

            const float* k_h = kv_manager.get_k(block_id, layer);
            if (!k_h) continue;
            k_h += pos_in_block * kv_dim + kv_h * head_dim;

            float dot = 0.0f;
            for (int32_t d = 0; d < head_dim; d++) dot += q_h[d] * k_h[d];
            sum_exp += std::exp(dot * scale - max_logit);
        }
        sum_exp += 1e-9f;

        // Pass 3: weighted V
        for (int32_t p = 0; p < seq_len; p++) {
            int32_t block_idx = p / BLOCK_SIZE;
            int32_t pos_in_block = p % BLOCK_SIZE;
            if ((size_t)block_idx >= block_table.size()) continue;
            int32_t block_id = block_table[block_idx];

            const float* k_h = kv_manager.get_k(block_id, layer);
            const float* v_h_base = kv_manager.get_v(block_id, layer);
            if (!k_h || !v_h_base) continue;

            k_h += pos_in_block * kv_dim + kv_h * head_dim;
            const float* v_h = v_h_base + pos_in_block * kv_dim + kv_h * head_dim;

            float dot = 0.0f;
            for (int32_t d = 0; d < head_dim; d++) dot += q_h[d] * k_h[d];
            float weight = std::exp(dot * scale - max_logit) / sum_exp;

            for (int32_t d = 0; d < head_dim; d++) o_h[d] += weight * v_h[d];
        }
    }
}

// ============================================================
// paged_attention_batch
// ============================================================

void paged_attention_batch(
    const std::vector<const float*>& q_ptrs,
    const std::vector<const std::vector<int32_t>*>& block_tables,
    const std::vector<int32_t>& seq_lens,
    const BlockPool& kv_manager,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t layer,
    std::vector<float*> out_ptrs)
{
    for (size_t i = 0; i < q_ptrs.size(); i++) {
        if (block_tables[i]) {
            paged_attention(
                q_ptrs[i],
                *block_tables[i],
                seq_lens[i],
                kv_manager,
                num_heads,
                num_kv_heads,
                head_dim,
                layer,
                out_ptrs[i]);
        }
    }
}

// ============================================================
// write_kv_token: Write single token to block pool (decode step)
// ============================================================

void write_kv_token(
    KVCacheManager& kv_manager,
    const std::vector<int32_t>& block_table,
    int32_t seq_len,
    int32_t layer,
    const float* k_data,
    const float* v_data)
{
    int32_t block_idx = (seq_len - 1) / BLOCK_SIZE;
    int32_t pos_in_block = (seq_len - 1) % BLOCK_SIZE;
    if (block_idx < 0 || (size_t)block_idx >= block_table.size()) return;

    int32_t block_id = block_table[block_idx];
    int32_t kv_dim = kv_manager.block_pool().kv_dim();
    kv_manager.write_kv(block_id, layer, pos_in_block, k_data, v_data, kv_dim);
}

// ============================================================
// write_kv_prefill: Write entire prefill KV to block pool
// ============================================================

void write_kv_prefill(
    KVCacheManager& kv_manager,
    const std::vector<int32_t>& block_table,
    int32_t seq_len,
    int32_t layer,
    const float* k_data,
    const float* v_data)
{
    int32_t kv_dim = kv_manager.block_pool().kv_dim();
    for (int32_t p = 0; p < seq_len; p++) {
        int32_t block_idx = p / BLOCK_SIZE;
        int32_t pos_in_block = p % BLOCK_SIZE;
        if (block_idx < 0 || (size_t)block_idx >= block_table.size()) continue;

        int32_t block_id = block_table[block_idx];
        kv_manager.write_kv(block_id, layer, pos_in_block,
                            k_data + p * kv_dim,
                            v_data + p * kv_dim,
                            kv_dim);
    }
}

// ============================================================
// read_kv_prefill: Read KV from block pool into contiguous buffers
// ============================================================

void read_kv_prefill(
    const BlockPool& kv_manager,
    const std::vector<int32_t>& block_table,
    int32_t seq_len,
    int32_t layer,
    int32_t,  // num_kv_heads (unused)
    int32_t,  // head_dim (unused)
    float* k_out,
    float* v_out)
{
    int32_t kv_dim = kv_manager.kv_dim();
    for (int32_t p = 0; p < seq_len; p++) {
        int32_t block_idx = p / BLOCK_SIZE;
        int32_t pos_in_block = p % BLOCK_SIZE;
        if (block_idx < 0 || (size_t)block_idx >= block_table.size()) continue;

        int32_t block_id = block_table[block_idx];
        const float* k_src = kv_manager.get_k(block_id, layer);
        const float* v_src = kv_manager.get_v(block_id, layer);
        if (!k_src || !v_src) continue;

        k_src += pos_in_block * kv_dim;
        v_src += pos_in_block * kv_dim;

        std::memcpy(k_out + p * kv_dim, k_src, kv_dim * sizeof(float));
        std::memcpy(v_out + p * kv_dim, v_src, kv_dim * sizeof(float));
    }
}
