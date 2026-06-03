#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <cmath>
#include <string>
#include <variant>
#include <stdexcept>

// ============================================================
// FP8 types: E4M3 (weights/activations) and E5M2 (gradients)
//
// E4M3: 1 sign, 4 exponent, 3 mantissa — range ~[-448, 448]
//   - Supports NaN (s1111_111) and inf (s1111_110)
//   - Subnormals: exp=0000 with mantissa≠0
// E5M2: 1 sign, 5 exponent, 2 mantissa — range ~[-57344, 57344]
//   - Supports inf (s11111_00) and NaN (s11111_{01,10,11})
// ============================================================

// ---- E4M3 ----
struct fp8_e4m3 {
    uint8_t bits;
    fp8_e4m3() : bits(0) {}
    explicit fp8_e4m3(uint8_t b) : bits(b) {}
};

// ---- E5M2 ----
struct fp8_e5m2 {
    uint8_t bits;
    fp8_e5m2() : bits(0) {}
    explicit fp8_e5m2(uint8_t b) : bits(b) {}
};

// ---- Conversion: FP8 ↔ float ----
// Pre-computed lookup tables for speed
void init_fp8_lut();  // called once on first use

float fp8_e4m3_to_float(fp8_e4m3 v);
float fp8_e5m2_to_float(fp8_e5m2 v);
fp8_e4m3 float_to_fp8_e4m3(float v);
fp8_e5m2 float_to_fp8_e5m2(float v);

// Batch conversion
void fp8_e4m3_to_float_batch(const uint8_t* src, float* dst, int64_t n);
void float_to_fp8_e4m3_batch(const float* src, uint8_t* dst, int64_t n);

// ============================================================
// INT4 packed storage (GPTQ/AWQ)
//
// Two 4-bit values packed into one byte:
//   byte = (low_nibble) | (high_nibble << 4)
// For GPTQ/AWQ: values are unsigned 4-bit (0-15), representing
// quantized weights where:
//   float_weight = (int4_value - zero) * scale
// Group size determines how many weights share one scale/zero.
// ============================================================

// Two INT4 values packed into one uint8_t
struct int4x2 {
    uint8_t packed;
    int4x2() : packed(0) {}
    int4x2(uint8_t lo, uint8_t hi) : packed((lo & 0xF) | ((hi & 0xF) << 4)) {}
    uint8_t lo() const { return packed & 0xF; }
    uint8_t hi() const { return (packed >> 4) & 0xF; }
};

// ============================================================
// Packed INT4 tensor — stores quantized weights for one matrix
//
// Layout (GPTQ, row-wise):
//   qweight: [out_features, in_features // 2] of uint8 (2 int4 per byte)
//   qzeros:  [num_groups, in_features // 8 * bits] — packed zeros, scaled by scales
//   scales:  [num_groups, out_features] — float or half
//   g_idx:   [in_features] — group assignment per input col (GPTQ act-order only)
//
// AWQ layout (differs from GPTQ):
//   qweight: same
//   qzeros:  [in_features // group_size, out_features // 8] (packed into int32)
//   scales:  [in_features // group_size, out_features]
// ============================================================

struct PackedInt4Tensor {
    // Raw quantized data
    std::vector<uint8_t> qweight;   // packed INT4: N int4 values → ceil(N/2) bytes
    std::vector<uint8_t> qzeros;    // packed INT4 zeros per group
    std::vector<float> scales;      // per-group scale factors (stored as float on CPU)
    std::vector<int32_t> g_idx;     // group index per input column (GPTQ act-order only)

    int32_t in_features = 0;
    int32_t out_features = 0;
    int32_t group_size = 128;
    int32_t bits = 4;
    bool desc_act = false;          // GPTQ act-order (g_idx present)
    bool is_awq = false;            // true = AWQ format, false = GPTQ format

    // Derived
    int32_t num_groups() const {
        return (in_features + group_size - 1) / group_size;
    }

    int64_t qweight_bytes() const {
        // Each byte stores 2 INT4 values
        return ((int64_t)in_features * out_features * bits + 7) / 8;
    }

    bool empty() const { return qweight.empty(); }
};

// ============================================================
// Dequantize a single row of a packed INT4 tensor → float
//
// For a row i in [0, out_features):
//   For each column j in [0, in_features):
//     group = g_idx ? g_idx[j] : j / group_size
//     scale = scales[group * out_features + i]
//     zero  = unpack_zero(qzeros, group, i)
//     weight_float[j] = (unpack_int4(qweight, i, j) - zero) * scale
// ============================================================

// Extract a single INT4 value from the packed tensor at (row, col)
inline uint8_t unpack_int4(const PackedInt4Tensor& t, int32_t row, int32_t col) {
    // qweight layout: [out_features, in_features // 2]
    // Each byte stores: qweight[row, col//2].lo = col_even, .hi = col_odd
    int32_t idx = row * (t.in_features / 2) + col / 2;
    if (col % 2 == 0) {
        return t.qweight[idx] & 0xF;
    } else {
        return (t.qweight[idx] >> 4) & 0xF;
    }
}

// Dequantize a single row (out_features dim) — returns [in_features] floats
std::vector<float> dequantize_int4_row(const PackedInt4Tensor& t, int32_t row);

// Dequantize a column range within a row — for batched matmul
// dst must have room for 'count' floats starting at col_offset
void dequantize_int4_row_range(const PackedInt4Tensor& t, int32_t row,
                                int32_t col_offset, int32_t count,
                                float* dst);

// Dequantize full matrix → [out_features * in_features] floats
std::vector<float> dequantize_int4_full(const PackedInt4Tensor& t);

// ============================================================
// FP8 quantized tensor (W8A8)
//
// Weights stored as FP8 E4M3 with per-tensor or per-channel scales.
// Activations quantized to FP8 on-the-fly before matmul.
// ============================================================

struct FP8QuantizedTensor {
    std::vector<uint8_t> data;       // FP8 E4M3 values
    std::vector<float> scales;       // per-channel (out_features) or single
    int32_t in_features = 0;
    int32_t out_features = 0;

    bool per_channel() const { return scales.size() > 1; }
    float scale(int32_t row) const {
        return per_channel() ? scales[row] : scales[0];
    }

    // Dequantize one row → [in_features] floats
    std::vector<float> dequantize_row(int32_t row) const;

    // Dequantize row range
    void dequantize_row_range(int32_t row, int32_t col_offset, int32_t count,
                              float* dst) const;

    bool empty() const { return data.empty(); }
};

// ============================================================
// Quantized tensor variant — unified container for model weights
//
// Model layers reference this instead of raw float* when quantized.
// The variant holds one of:
//   - float* (legacy FP32, owned elsewhere)
//   - PackedInt4Tensor (GPTQ/AWQ)
//   - FP8QuantizedTensor (FP8 W8A8)
// ============================================================

struct QuantizedTensor {
    enum class Type : uint8_t { FP32, FP16, FP8, INT4_GPTQ, INT4_AWQ };

    Type type = Type::FP32;

    // Only one of these is active depending on type
    float* fp32_data = nullptr;               // Type::FP32
    uint16_t* fp16_data = nullptr;            // Type::FP16 (stored as fp16, convert on access)
    FP8QuantizedTensor fp8_tensor;            // Type::FP8
    PackedInt4Tensor int4_tensor;             // Type::INT4_GPTQ / INT4_AWQ

    int64_t numel = 0;
    std::vector<int64_t> shape;

    QuantizedTensor() = default;

    // Create FP32 tensor
    static QuantizedTensor from_fp32(float* data, const std::vector<int64_t>& shape) {
        QuantizedTensor t;
        t.type = Type::FP32;
        t.fp32_data = data;
        t.shape = shape;
        t.numel = data ? 1 : 0;
        for (auto d : shape) t.numel *= d;
        return t;
    }

    // Create INT4 GPTQ tensor
    static QuantizedTensor from_gptq(PackedInt4Tensor&& packed,
                                      const std::vector<int64_t>& shape) {
        QuantizedTensor t;
        t.type = Type::INT4_GPTQ;
        t.int4_tensor = std::move(packed);
        t.shape = shape;
        t.int4_tensor.out_features = (int32_t)shape[0];
        t.int4_tensor.in_features = (int32_t)shape[1];
        return t;
    }

    // Create INT4 AWQ tensor
    static QuantizedTensor from_awq(PackedInt4Tensor&& packed,
                                     const std::vector<int64_t>& shape) {
        QuantizedTensor t;
        t.type = Type::INT4_AWQ;
        t.int4_tensor = std::move(packed);
        t.shape = shape;
        t.int4_tensor.out_features = (int32_t)shape[0];
        t.int4_tensor.in_features = (int32_t)shape[1];
        t.int4_tensor.is_awq = true;
        return t;
    }

    // Create FP8 tensor
    static QuantizedTensor from_fp8(FP8QuantizedTensor&& fp8,
                                     const std::vector<int64_t>& shape) {
        QuantizedTensor t;
        t.type = Type::FP8;
        t.fp8_tensor = std::move(fp8);
        t.shape = shape;
        return t;
    }

    bool is_quantized() const { return type != Type::FP32 && type != Type::FP16; }
    int64_t count() const { return numel; }

    // Dequantize entire tensor to FP32 vector (expensive, use for debugging)
    std::vector<float> to_float_vector() const;

    // Dequantize one row (for out_features-dim access) → [in_features] floats
    // Row index refers to out_features dimension
    std::vector<float> dequantize_row(int32_t row) const;

    // Dequantize a range of one row (for efficient matmul)
    void dequantize_row_range(int32_t row, int32_t col_offset, int32_t count,
                              float* dst) const;
};

// ============================================================
// Quantized matmul: C[m,n] = A[m,k] @ Q[k,n].dequantized
//
// A is float, Q is quantized weight [k, n] (stored n×k internally).
// This fuses dequantization with the matmul for efficiency.
// ============================================================
void quantized_matmul_transpose(
    const float* a, int32_t m, int32_t k,
    const QuantizedTensor& q, int32_t n,
    float* c
);
