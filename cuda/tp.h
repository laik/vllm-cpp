#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================
// Tensor Parallel (TP) communication primitives for CUDA
// ============================================================
//
// Each GPU holds 1/TP_SIZE slice of column-parallel weight outputs
// and performs NCCL collectives for row-parallel layers.
//
// Column-parallel (Q_proj, K_proj, V_proj, gate_proj, up_proj):
//   - Weight is split along output dim
//   - Each GPU computes its slice independently
//   - No collective needed
//
// Row-parallel (O_proj, down_proj):
//   - Weight is split along input dim
//   - Each GPU computes partial output
//   - NCCL all-reduce to sum partials across GPUs
//
// Attention with TP:
//   - Q is column-parallel (split by head) -> no collective
//   - K, V are column-parallel (split by kv_head) -> no collective
//   - Attention requires ALL heads on each GPU -> all-gather Q, K, V
//   - O_proj is row-parallel -> all-reduce after matmul
// ============================================================

struct TPConfig {
    int32_t tp_size = 1;
    int32_t tp_rank = 0;
    cudaStream_t stream = 0;
    void* nccl_comm = nullptr;  // ncclComm_t*, opaque handle

    bool enabled() const { return tp_size > 1; }
};

// Initialize TP communicator on the given CUDA device
// rank must match the physical GPU ID
int tp_init(int32_t rank, int32_t tp_size, const char* nccl_socket_ifname);

// Destroy TP communicator
int tp_finalize();

// Get current TP config (global singleton)
TPConfig* tp_get();

// All-reduce (sum) for row-parallel layer outputs
int tp_allreduce(float* data, int32_t count, cudaStream_t stream);

// All-gather for attention head reconstruction
int tp_allgather(float* recv_buf, const float* send_buf, int32_t per_rank_count, cudaStream_t stream);

// TP-aware split helpers: given full dimension D, return local offset and size
void tp_split_dim(int32_t full_dim, int32_t* local_offset, int32_t* local_size);

// TP-aware split for attention heads (must align to head boundaries)
void tp_split_heads(int32_t num_heads, int32_t* local_offset, int32_t* local_size);
