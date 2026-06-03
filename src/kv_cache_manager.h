#pragma once
#include "config.h"
#include "quantization.h"
#include <cstdint>
#include <vector>
#include <mutex>
#include <memory>
#include <cstring>

// ============================================================
// PagedAttention KV Cache Manager — typed block storage
//
// Supports FP32, FP16, FP8_E4M3 KV cache backends.
// Blocks store raw bytes; conversion to/from float is done
// at the access boundary (paged_attention fuses dequant).
//
// Memory savings vs FP32:
//   FP8_E4M3: 4x smaller (1 byte/elem vs 4)
//   FP16:     2x smaller (2 bytes/elem vs 4)
// ============================================================

// Byte size per element for each KV dtype
inline constexpr int32_t kv_elem_size(KVDtype dtype) {
    switch (dtype) {
        case KVDtype::FP32:     return 4;
        case KVDtype::FP16:     return 2;
        case KVDtype::FP8_E4M3: return 1;
    }
    return 4;
}

// One physical block: stores KV cache for BLOCK_SIZE tokens in one layer
struct KVsBlock {
    int32_t block_id = -1;
    int32_t ref_count = 0;
    bool free = true;
    KVDtype dtype = KVDtype::FP32;

    // Raw byte storage: [num_layers * BLOCK_SIZE * kv_dim * elem_size]
    std::vector<uint8_t>* k_data = nullptr;
    std::vector<uint8_t>* v_data = nullptr;

    // Size of one layer's KV in bytes
    int64_t layer_kv_bytes() const {
        return (int64_t)BLOCK_SIZE * kv_elem_size(dtype);
    }
};

// Block pool managing physical blocks
class BlockPool {
public:
    BlockPool(int32_t num_blocks, int32_t num_layers, int32_t kv_dim,
              KVDtype kv_dtype = KVDtype::FP32);
    ~BlockPool();

    // Allocate a free block, return block_id (-1 if none available)
    int32_t alloc();

    // Free a block (decrement ref_count, physically free if reaches 0)
    void free(int32_t block_id);

    // Increment reference count (for prefix sharing)
    void ref(int32_t block_id);

    // Decrement reference count
    void deref(int32_t block_id);

    // ---- Raw byte access (for fused dequant kernels) ----
    uint8_t* get_k_raw(int32_t block_id, int32_t layer);
    uint8_t* get_v_raw(int32_t block_id, int32_t layer);
    const uint8_t* get_k_raw(int32_t block_id, int32_t layer) const;
    const uint8_t* get_v_raw(int32_t block_id, int32_t layer) const;

    // ---- Float access (auto-converts FP8/FP16 → FP32) ----
    // These allocate/fill a temp buffer — use get_k_raw for perf-critical paths
    float* get_k(int32_t block_id, int32_t layer);
    float* get_v(int32_t block_id, int32_t layer);
    const float* get_k(int32_t block_id, int32_t layer) const;
    const float* get_v(int32_t block_id, int32_t layer) const;

    // Copy a block's KV data (for preemption checkpointing)
    void copy_to(int32_t dst, int32_t src);

    // Get total free/used blocks
    int32_t free_count() const;
    int32_t used_count() const;

    // Reset entire pool
    void reset();

    // Get mutable block for direct access
    KVsBlock& get_block(int32_t block_id);
    const KVsBlock& get_block(int32_t block_id) const;

    int32_t num_blocks() const { return (int32_t)blocks.size(); }
    int32_t kv_dim() const { return kv_dim_; }
    int32_t num_layers() const { return num_layers_; }
    KVDtype kv_dtype() const { return kv_dtype_; }

private:
    std::vector<KVsBlock> blocks;
    int32_t kv_dim_;
    int32_t num_layers_;
    KVDtype kv_dtype_;
    int32_t next_free_;

    // Conversion buffers for float access API
    mutable std::vector<float> k_convert_buf_;
    mutable std::vector<float> v_convert_buf_;
    mutable std::mutex mtx_;
};

// Virtual memory manager that owns the block pool
class KVCacheManager {
public:
    KVCacheManager(int32_t num_layers, int32_t kv_dim, int32_t num_blocks,
                   KVDtype kv_dtype = KVDtype::FP32);
    ~KVCacheManager();

    // Allocate N blocks (return physical block ids)
    bool alloc_blocks(int32_t num_blocks, std::vector<int32_t>& out);

    // Free all blocks in a list
    void free_blocks(const std::vector<int32_t>& blocks_to_free);

    // Copy blocks from src to dst
    void copy_blocks(const std::vector<int32_t>& src, const std::vector<int32_t>& dst);

    // Get the underlying block pool
    BlockPool& block_pool() { return pool_; }
    const BlockPool& block_pool() const { return pool_; }

    // Write KV data into a block (float input, auto-converts based on dtype)
    void write_kv(int32_t block_id, int32_t layer, int32_t pos_in_block,
                  const float* k_data, const float* v_data, int32_t kv_dim);

    // Read KV data from a block (always returns float)
    void read_kv(int32_t block_id, int32_t layer, int32_t pos_in_block,
                 float* k_data, float* v_data, int32_t kv_dim) const;

    int32_t free_blocks_count() const;
    int32_t used_blocks_count() const;
    KVDtype kv_dtype() const { return pool_.kv_dtype(); }

private:
    BlockPool pool_;
};
