#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================
// CUDA utility functions for Qwen3.6 inference
// Half-precision (FP16) and vectorized load/store operations
// ============================================================

// ----- FP16 conversion (device) -----

__device__ static inline float half2float(half h) {
    return __half2float(h);
}

__device__ static inline half float2half(float f) {
    return __float2half(f);
}

__device__ static inline float2 half22float2(half2 h) {
    return __half22float2(h);
}

__device__ static inline half2 float22half2(float2 f) {
    return __float22half2_rn(f);
}

// ----- FP16 arithmetic (device) -----

__device__ static inline half half_add(half a, half b) {
    return __hadd(a, b);
}

__device__ static inline half half_mul(half a, half b) {
    return __hmul(a, b);
}

__device__ static inline half2 half2_add(half2 a, half2 b) {
    return __hadd2(a, b);
}

__device__ static inline half2 half2_mul(half2 a, half2 b) {
    return __hmul2(a, b);
}

// ----- Vectorized memory operations -----

// Load 4 floats as float4
__device__ static inline float4 load_float4(const float* ptr) {
    return *(const float4*)ptr;
}

// Store float4
__device__ static inline void store_float4(float* ptr, float4 val) {
    *(float4*)ptr = val;
}

// Load 2 half2 values (4 half = 8 bytes) as float4 equivalent
__device__ static inline float4 load_half4(const half* ptr) {
    half2 h0 = *(const half2*)ptr;
    half2 h1 = *(const half2*)(ptr + 2);
    float2 f0 = half22float2(h0);
    float2 f1 = half22float2(h1);
    return make_float4(f0.x, f0.y, f1.x, f1.y);
}

// Store float4 as 4 half values
__device__ static inline void store_half4(half* ptr, float4 val) {
    half2 h0 = float22half2(make_float2(val.x, val.y));
    half2 h1 = float22half2(make_float2(val.z, val.w));
    *(half2*)ptr = h0;
    *(half2*)(ptr + 2) = h1;
}

// Load 8 half values (16 bytes) - uses two global loads
__device__ static inline void load_half8(const half* ptr, float* out) {
    float4 f0 = load_half4(ptr);
    float4 f1 = load_half4(ptr + 4);
    out[0] = f0.x; out[1] = f0.y; out[2] = f0.z; out[3] = f0.w;
    out[4] = f1.x; out[5] = f1.y; out[6] = f1.z; out[7] = f1.w;
}

// Store 8 floats as 8 half values
__device__ static inline void store_half8(const float* in, half* ptr) {
    float4 f0 = make_float4(in[0], in[1], in[2], in[3]);
    float4 f1 = make_float4(in[4], in[5], in[6], in[7]);
    store_half4(ptr, f0);
    store_half4(ptr + 4, f1);
}

// ----- Shared memory helpers -----

template<typename T>
__device__ static inline void sync_shared() {
    __syncthreads();
}

// ----- Block-level reduce (max + sum for softmax) -----

__device__ static inline void block_reduce_max(float* local, float& result) {
    // Assumes local has blockDim.x elements
    int tid = threadIdx.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local[tid] = fmaxf(local[tid], local[tid + stride]);
        }
        __syncthreads();
    }
    result = local[0];
}

__device__ static inline void block_reduce_sum(float* local, float& result) {
    int tid = threadIdx.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local[tid] += local[tid + stride];
        }
        __syncthreads();
    }
    result = local[0];
}

// ----- Memory alignment helpers -----

template<typename T>
static inline T* align_ptr(T* ptr, size_t alignment) {
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % alignment == 0) return ptr;
    return (T*)((addr + alignment - 1) & ~(alignment - 1));
}

// ----- CUDA error checking (debug mode) -----

#ifdef VLLM_CUDA_DEBUG
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(1);                                                          \
        }                                                                     \
    } while (0)
#else
#define CUDA_CHECK(call) (call)
#endif

// ----- Kernel launch helpers -----

static inline dim3 make_grid(int32_t n, int32_t block_size = 256) {
    return dim3((n + block_size - 1) / block_size, 1, 1);
}

static inline dim3 make_2d_grid(int32_t m, int32_t n, int32_t bx = 32, int32_t by = 32) {
    return dim3((m + bx - 1) / bx, (n + by - 1) / by, 1);
}
