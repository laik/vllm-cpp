#include "kv_cache_manager.h"
#include "config.h"
#include "quantization.h"
#include <cstring>
#include <algorithm>
#include <cstdio>

// ============================================================
// BlockPool
// ============================================================

BlockPool::BlockPool(int32_t num_blocks, int32_t num_layers, int32_t kv_dim, KVDtype kv_dtype)
    : kv_dim_(kv_dim), num_layers_(num_layers), kv_dtype_(kv_dtype), next_free_(0)
{
    int32_t esize = kv_elem_size(kv_dtype);
    // Bytes per layer per block: BLOCK_SIZE * kv_dim * elem_size
    // Total bytes per block: num_layers * BLOCK_SIZE * kv_dim * elem_size
    int64_t layer_bytes = (int64_t)BLOCK_SIZE * kv_dim * esize;
    int64_t block_bytes = (int64_t)num_layers * layer_bytes;

    blocks.resize(num_blocks);
    for (int32_t i = 0; i < num_blocks; i++) {
        blocks[i].block_id = i;
        blocks[i].ref_count = 0;
        blocks[i].free = true;
        blocks[i].dtype = kv_dtype;
        blocks[i].k_data = new std::vector<uint8_t>(block_bytes, 0);
        blocks[i].v_data = new std::vector<uint8_t>(block_bytes, 0);
    }

    // Pre-allocate conversion buffers
    k_convert_buf_.resize(BLOCK_SIZE * kv_dim);
    v_convert_buf_.resize(BLOCK_SIZE * kv_dim);
}

BlockPool::~BlockPool() {
    for (auto& b : blocks) {
        delete b.k_data;
        delete b.v_data;
    }
}

int32_t BlockPool::alloc() {
    std::lock_guard<std::mutex> lock(mtx_);

    for (int32_t i = 0; i < (int32_t)blocks.size(); i++) {
        if (blocks[i].free) {
            blocks[i].free = false;
            blocks[i].ref_count = 1;
            return i;
        }
    }
    return -1;
}

void BlockPool::free(int32_t block_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (block_id < 0 || (size_t)block_id >= blocks.size()) return;

    KVsBlock& b = blocks[block_id];
    b.ref_count--;
    if (b.ref_count <= 0) {
        b.ref_count = 0;
        b.free = true;
        // Clear data (zero out)
        if (b.k_data) std::fill(b.k_data->begin(), b.k_data->end(), 0);
        if (b.v_data) std::fill(b.v_data->begin(), b.v_data->end(), 0);
    }
}

void BlockPool::ref(int32_t block_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (block_id < 0 || (size_t)block_id >= blocks.size()) return;
    blocks[block_id].ref_count++;
}

void BlockPool::deref(int32_t block_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (block_id < 0 || (size_t)block_id >= blocks.size()) return;
    KVsBlock& b = blocks[block_id];
    b.ref_count--;
    if (b.ref_count <= 0) {
        b.ref_count = 0;
        b.free = true;
    }
}

// ---- Raw byte access ----

uint8_t* BlockPool::get_k_raw(int32_t block_id, int32_t layer) {
    if (block_id < 0 || (size_t)block_id >= blocks.size()) return nullptr;
    if (layer < 0 || layer >= num_layers_) return nullptr;
    int64_t layer_bytes = (int64_t)BLOCK_SIZE * kv_dim_ * kv_elem_size(kv_dtype_);
    return blocks[block_id].k_data->data() + layer * layer_bytes;
}

uint8_t* BlockPool::get_v_raw(int32_t block_id, int32_t layer) {
    if (block_id < 0 || (size_t)block_id >= blocks.size()) return nullptr;
    if (layer < 0 || layer >= num_layers_) return nullptr;
    int64_t layer_bytes = (int64_t)BLOCK_SIZE * kv_dim_ * kv_elem_size(kv_dtype_);
    return blocks[block_id].v_data->data() + layer * layer_bytes;
}

const uint8_t* BlockPool::get_k_raw(int32_t block_id, int32_t layer) const {
    return const_cast<BlockPool*>(this)->get_k_raw(block_id, layer);
}

const uint8_t* BlockPool::get_v_raw(int32_t block_id, int32_t layer) const {
    return const_cast<BlockPool*>(this)->get_v_raw(block_id, layer);
}

// ---- Float access (with conversion) ----

// Helper: convert from native KV dtype to float buffer
static void convert_kv_to_float(const uint8_t* src, float* dst, int32_t n, KVDtype dtype) {
    switch (dtype) {
        case KVDtype::FP32:
            std::memcpy(dst, src, n * sizeof(float));
            break;
        case KVDtype::FP16: {
            const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
            for (int32_t i = 0; i < n; i++) {
                uint16_t h = src16[i];
                uint32_t sign = (h >> 15) & 0x1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t frac = h & 0x3FF;
                if (exp == 0) {
                    if (frac == 0) { dst[i] = sign ? -0.0f : 0.0f; }
                    else {
                        uint32_t e = 1;
                        while ((frac & 0x400) == 0) { frac <<= 1; e--; }
                        frac &= 0x3FF;
                        uint32_t bits = (sign << 31) | ((127 - 15 + e) << 23) | (frac << 13);
                        dst[i] = *(float*)&bits;
                    }
                } else if (exp == 31) {
                    dst[i] = sign ? -INFINITY : INFINITY;
                } else {
                    uint32_t bits = (sign << 31) | ((127 - 15 + exp) << 23) | (frac << 13);
                    dst[i] = *(float*)&bits;
                }
            }
            break;
        }
        case KVDtype::FP8_E4M3:
            fp8_e4m3_to_float_batch(src, dst, n);
            break;
    }
}

// Helper: convert from float to native KV dtype
static void convert_float_to_kv(const float* src, uint8_t* dst, int32_t n, KVDtype dtype) {
    switch (dtype) {
        case KVDtype::FP32:
            std::memcpy(dst, src, n * sizeof(float));
            break;
        case KVDtype::FP16: {
            uint16_t* dst16 = reinterpret_cast<uint16_t*>(dst);
            for (int32_t i = 0; i < n; i++) {
                // float → fp16 using hardware or software
                uint32_t bits = *(uint32_t*)&src[i];
                uint32_t sign = (bits >> 31) & 0x1;
                uint32_t exp = (bits >> 23) & 0xFF;
                uint32_t frac = bits & 0x7FFFFF;
                if (exp == 0xFF) {
                    // Inf or NaN
                    dst16[i] = (sign << 15) | (0x1F << 10) | ((frac != 0) ? 1 : 0);
                } else if (exp == 0) {
                    // Zero or subnormal → zero in fp16
                    dst16[i] = sign << 15;
                } else {
                    int e = (int)exp - 127 + 15;
                    if (e <= 0) { dst16[i] = sign << 15; }        // underflow → 0
                    else if (e >= 31) { dst16[i] = (sign << 15) | (0x1F << 10); } // overflow → inf
                    else { dst16[i] = (sign << 15) | (e << 10) | ((frac >> 13) & 0x3FF); }
                }
            }
            break;
        }
        case KVDtype::FP8_E4M3:
            float_to_fp8_e4m3_batch(src, dst, n);
            break;
    }
}

float* BlockPool::get_k(int32_t block_id, int32_t layer) {
    if (kv_dtype_ == KVDtype::FP32) {
        return reinterpret_cast<float*>(get_k_raw(block_id, layer));
    }
    // Convert from native type to float
    const uint8_t* raw = get_k_raw(block_id, layer);
    if (!raw) return nullptr;
    int32_t n = BLOCK_SIZE * kv_dim_;
    std::lock_guard<std::mutex> lock(mtx_);
    if ((int32_t)k_convert_buf_.size() < n) k_convert_buf_.resize(n);
    convert_kv_to_float(raw, k_convert_buf_.data(), n, kv_dtype_);
    return k_convert_buf_.data();
}

float* BlockPool::get_v(int32_t block_id, int32_t layer) {
    if (kv_dtype_ == KVDtype::FP32) {
        return reinterpret_cast<float*>(get_v_raw(block_id, layer));
    }
    const uint8_t* raw = get_v_raw(block_id, layer);
    if (!raw) return nullptr;
    int32_t n = BLOCK_SIZE * kv_dim_;
    std::lock_guard<std::mutex> lock(mtx_);
    if ((int32_t)v_convert_buf_.size() < n) v_convert_buf_.resize(n);
    convert_kv_to_float(raw, v_convert_buf_.data(), n, kv_dtype_);
    return v_convert_buf_.data();
}

const float* BlockPool::get_k(int32_t block_id, int32_t layer) const {
    return const_cast<BlockPool*>(this)->get_k(block_id, layer);
}

const float* BlockPool::get_v(int32_t block_id, int32_t layer) const {
    return const_cast<BlockPool*>(this)->get_v(block_id, layer);
}

void BlockPool::copy_to(int32_t dst, int32_t src) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (dst < 0 || dst >= (int32_t)blocks.size()) return;
    if (src < 0 || src >= (int32_t)blocks.size()) return;

    if (blocks[dst].k_data && blocks[src].k_data) {
        *blocks[dst].k_data = *blocks[src].k_data;
    }
    if (blocks[dst].v_data && blocks[src].v_data) {
        *blocks[dst].v_data = *blocks[src].v_data;
    }
}

int32_t BlockPool::free_count() const {
    int32_t c = 0;
    for (const auto& b : blocks) if (b.free) c++;
    return c;
}

int32_t BlockPool::used_count() const {
    return (int32_t)blocks.size() - free_count();
}

void BlockPool::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& b : blocks) {
        b.ref_count = 0;
        b.free = true;
        if (b.k_data) std::fill(b.k_data->begin(), b.k_data->end(), 0);
        if (b.v_data) std::fill(b.v_data->begin(), b.v_data->end(), 0);
    }
}

KVsBlock& BlockPool::get_block(int32_t block_id) {
    return blocks[block_id];
}

const KVsBlock& BlockPool::get_block(int32_t block_id) const {
    return blocks[block_id];
}

// ============================================================
// KVCacheManager
// ============================================================

KVCacheManager::KVCacheManager(int32_t num_layers, int32_t kv_dim, int32_t num_blocks,
                               KVDtype kv_dtype)
    : pool_(num_blocks, num_layers, kv_dim, kv_dtype)
{
}

KVCacheManager::~KVCacheManager() {}

bool KVCacheManager::alloc_blocks(int32_t num_blocks, std::vector<int32_t>& out) {
    out.clear();
    out.reserve(num_blocks);
    for (int32_t i = 0; i < num_blocks; i++) {
        int32_t bid = pool_.alloc();
        if (bid < 0) return false;
        out.push_back(bid);
    }
    return true;
}

void KVCacheManager::free_blocks(const std::vector<int32_t>& blocks_to_free) {
    for (int32_t bid : blocks_to_free) {
        pool_.free(bid);
    }
}

void KVCacheManager::copy_blocks(const std::vector<int32_t>& src, const std::vector<int32_t>& dst) {
    for (size_t i = 0; i < src.size() && i < dst.size(); i++) {
        pool_.copy_to(dst[i], src[i]);
    }
}

void KVCacheManager::write_kv(int32_t block_id, int32_t layer, int32_t pos_in_block,
                               const float* k_data, const float* v_data, int32_t kv_dim) {
    if (pos_in_block < 0 || pos_in_block >= BLOCK_SIZE) return;

    KVDtype dtype = pool_.kv_dtype();
    int32_t esize = kv_elem_size(dtype);

    // Write to raw storage, converting from float
    uint8_t* k_dst_raw = pool_.get_k_raw(block_id, layer);
    uint8_t* v_dst_raw = pool_.get_v_raw(block_id, layer);
    if (!k_dst_raw || !v_dst_raw) return;

    uint8_t* dst_k = k_dst_raw + pos_in_block * kv_dim * esize;
    uint8_t* dst_v = v_dst_raw + pos_in_block * kv_dim * esize;

    convert_float_to_kv(k_data, dst_k, kv_dim, dtype);
    convert_float_to_kv(v_data, dst_v, kv_dim, dtype);
}

void KVCacheManager::read_kv(int32_t block_id, int32_t layer, int32_t pos_in_block,
                              float* k_data, float* v_data, int32_t kv_dim) const {
    if (pos_in_block < 0 || pos_in_block >= BLOCK_SIZE) return;

    KVDtype dtype = pool_.kv_dtype();
    int32_t esize = kv_elem_size(dtype);

    const uint8_t* k_src_raw = pool_.get_k_raw(block_id, layer);
    const uint8_t* v_src_raw = pool_.get_v_raw(block_id, layer);
    if (!k_src_raw || !v_src_raw) return;

    const uint8_t* src_k = k_src_raw + pos_in_block * kv_dim * esize;
    const uint8_t* src_v = v_src_raw + pos_in_block * kv_dim * esize;

    convert_kv_to_float(src_k, k_data, kv_dim, dtype);
    convert_kv_to_float(src_v, v_data, kv_dim, dtype);
}

int32_t KVCacheManager::free_blocks_count() const {
    return pool_.free_count();
}

int32_t KVCacheManager::used_blocks_count() const {
    return pool_.used_count();
}
