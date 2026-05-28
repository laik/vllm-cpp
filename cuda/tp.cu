#include "tp.h"
#include "utils.cuh"
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================================================
// Global TP state (singleton)
// ============================================================
static TPConfig g_tp;

// ============================================================
// TP initialization
// ============================================================

int tp_init(int32_t rank, int32_t tp_size, const char* nccl_socket_ifname)
{
    if (tp_size <= 0 || tp_size > 8) {
        fprintf(stderr, "ERROR: invalid tp_size=%d (must be 1-8)\n", tp_size);
        return -1;
    }
    if (rank < 0 || rank >= tp_size) {
        fprintf(stderr, "ERROR: invalid rank=%d for tp_size=%d\n", rank, tp_size);
        return -1;
    }

    g_tp.tp_size = tp_size;
    g_tp.tp_rank = rank;

    // Set CUDA device for this rank
    cudaError_t err = cudaSetDevice(rank);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaSetDevice(%d) failed: %s\n", rank, cudaGetErrorString(err));
        return -1;
    }

    // Single-GPU "TP" is a no-op
    if (tp_size == 1) {
        g_tp.nccl_comm = nullptr;
        cudaStreamCreate(&g_tp.stream);
        return 0;
    }

    // Create NCCL communicator using CUDA unique pointer for rendezvous
    void* nvod[1];
    int nvod_sizes[1] = {0};
    ncclGetUniqueId(nvod, &nvod_sizes[0]);

    ncclComm_t comm = nullptr;
    ncclResult_t nccl_err;

    // Use env-based rendezvous: rank 0 sets the id, all ranks join
    char* tp_socket = getenv("NCCL_SOCKET_IFNAME");
    if (nccl_socket_ifname) {
        setenv("NCCL_SOCKET_IFNAME", nccl_socket_ifname, 1);
    }

    // For multi-process launch, each process reads its own rank/size from env
    // We use ncclCommInitRank with a shared unique id passed via env
    char* id_str = getenv("NCCL_UNIQUE_ID");
    if (id_str) {
        // Restore unique id from string
        ncclUniqueID uid;
        for (int i = 0; i < 128; i++) {
            unsigned int byte;
            sscanf(id_str + (i * 2), "%02x", &byte);
            ((char*)(&uid))[i] = (char)byte;
        }
        nccl_err = ncclCommInitRank(&comm, tp_size, &uid, rank);
    } else {
        // Single-process multi-thread mode: pass comm pointers array
        nccl_err = ncclCommInitRank(&comm, tp_size, nvod, rank);
    }

    if (nccl_err != ncclSuccess || !comm) {
        fprintf(stderr, "ERROR: ncclCommInitRank failed: %s\n", ncclGetErrorString(nccl_err));
        return -1;
    }

    g_tp.nccl_comm = (void*)comm;
    cudaStreamCreate(&g_tp.stream);

    fprintf(stderr, "TP initialized: rank=%d/%d, device=%d\n", rank, tp_size, rank);
    return 0;
}

// ============================================================
// TP finalize
// ============================================================

int tp_finalize()
{
    if (g_tp.stream) {
        cudaStreamDestroy(g_tp.stream);
        g_tp.stream = 0;
    }
    if (g_tp.nccl_comm) {
        ncclCommDestroy((ncclComm_t)g_tp.nccl_comm);
        g_tp.nccl_comm = nullptr;
    }
    g_tp.tp_size = 1;
    g_tp.tp_rank = 0;
    return 0;
}

TPConfig* tp_get()
{
    return &g_tp;
}

// ============================================================
// Collective operations
// ============================================================

int tp_allreduce(float* data, int32_t count, cudaStream_t stream)
{
    if (g_tp.tp_size <= 1) return 0;

    ncclResult_t err = ncclAllReduce(
        data, data, count, ncclFloat, ncclSum,
        (ncclComm_t)g_tp.nccl_comm,
        stream ? stream : g_tp.stream);

    if (err != ncclSuccess) {
        fprintf(stderr, "NCCL all-reduce failed: %s\n", ncclGetErrorString(err));
        return -1;
    }
    return 0;
}

int tp_allgather(float* recv_buf, const float* send_buf, int32_t per_rank_count, cudaStream_t stream)
{
    if (g_tp.tp_size <= 1) {
        // Copy in place for single-GPU
        if (recv_buf != send_buf)
            cudaMemcpyAsync(recv_buf, send_buf, per_rank_count * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream ? stream : g_tp.stream);
        return 0;
    }

    ncclResult_t err = ncclAllGather(
        send_buf, recv_buf, per_rank_count, ncclFloat,
        (ncclComm_t)g_tp.nccl_comm,
        stream ? stream : g_tp.stream);

    if (err != ncclSuccess) {
        fprintf(stderr, "NCCL all-gather failed: %s\n", ncclGetErrorString(err));
        return -1;
    }
    return 0;
}

// ============================================================
// Dimension splitting helpers
// ============================================================

void tp_split_dim(int32_t full_dim, int32_t* local_offset, int32_t* local_size)
{
    int32_t base = full_dim / g_tp.tp_size;
    int32_t remainder = full_dim % g_tp.tp_size;
    *local_size = base + (g_tp.tp_rank < remainder ? 1 : 0);
    // Offset: sum of sizes for ranks 0..rank-1
    *local_offset = base * g_tp.tp_rank + (g_tp.tp_rank < remainder ? g_tp.tp_rank : remainder);
}

void tp_split_heads(int32_t num_heads, int32_t* local_offset, int32_t* local_size)
{
    // Heads must divide evenly across TP ranks
    if (num_heads % g_tp.tp_size != 0) {
        fprintf(stderr, "ERROR: num_heads=%d is not divisible by tp_size=%d\n",
                num_heads, g_tp.tp_size);
        *local_offset = 0;
        *local_size = num_heads;
        return;
    }
    *local_size = num_heads / g_tp.tp_size;
    *local_offset = g_tp.tp_rank * *local_size;
}
