#include "quantization.h"
#include <algorithm>
#include <cstring>
#include <cmath>

// ============================================================
// FP8 E4M3 ↔ float lookup tables
// ============================================================

static float  g_e4m3_to_float[256];
static uint8_t g_float_to_e4m3[256];  // indexed by float's first byte? No — use computation
static bool   g_lut_initialized = false;

// E4M3 bit fields:
//   bit 7: sign
//   bits 6-3: exponent (4 bits, bias=7)
//   bits 2-0: mantissa (3 bits)
//
// Special values:
//   s_1111_111 = NaN
//   s_1111_110 = (-1)^s * inf
//   s_0000_xxx = subnormal (exp=-6, mantissa=0.xxx)
//   s_eeee_mmm = (-1)^s * 2^(eeee-7) * 1.mmm  (normal)

static float compute_e4m3_to_float(uint8_t bits) {
    int sign = (bits >> 7) & 1;
    int exp  = (bits >> 3) & 0xF;
    int mant = bits & 0x7;

    if (exp == 0xF) {
        // NaN or inf
        if (mant == 0x7) return std::nanf("");
        if (mant == 0x6) return sign ? -INFINITY : INFINITY;
        // Other mant values in exp=15 are also NaN
        return std::nanf("");
    }

    float value;
    if (exp == 0) {
        // Subnormal: (-1)^s * 2^(-6) * 0.mant
        value = (float)mant / 8.0f * std::ldexp(1.0f, -6);
    } else {
        // Normal: (-1)^s * 2^(exp-7) * 1.mant
        value = (1.0f + (float)mant / 8.0f) * std::ldexp(1.0f, exp - 7);
    }

    return sign ? -value : value;
}

static uint8_t compute_float_to_e4m3(float v) {
    if (std::isnan(v)) return 0xFF;  // NaN
    if (std::isinf(v)) {
        return (v > 0) ? 0x7E : 0xFE;  // +inf / -inf
    }

    int sign = (v < 0) ? 1 : 0;
    v = std::fabs(v);

    if (v == 0.0f) return sign << 7;

    // Find closest E4M3 representation
    // E4M3 max normal: 1.111 * 2^7 = 1.875 * 128 = 240
    // E4M3 max subnormal: 0.111 * 2^-6 = 0.875 / 64 ≈ 0.0137
    // E4M3 min subnormal: 0.001 * 2^-6 = 0.125 / 64 ≈ 0.00195

    // Try all 256 values, find closest
    uint8_t best = 0;
    float best_err = INFINITY;
    for (int i = 0; i < 256; i++) {
        uint8_t candidate = (uint8_t)i;
        if ((candidate >> 7) != sign) continue;  // skip wrong sign
        float ref = g_e4m3_to_float[candidate];
        float err = std::fabs(v - std::fabs(ref));
        if (err < best_err) {
            best_err = err;
            best = candidate;
        }
    }

    return best;
}

void init_fp8_lut() {
    if (g_lut_initialized) return;

    for (int i = 0; i < 256; i++) {
        g_e4m3_to_float[i] = compute_e4m3_to_float((uint8_t)i);
    }

    g_lut_initialized = true;
}

// ============================================================
// Public FP8 conversion API
// ============================================================

float fp8_e4m3_to_float(fp8_e4m3 v) {
    init_fp8_lut();
    return g_e4m3_to_float[v.bits];
}

float fp8_e5m2_to_float(fp8_e5m2 v) {
    // E5M2: 1 sign, 5 exponent (bias=15), 2 mantissa
    int sign = (v.bits >> 7) & 1;
    int exp  = (v.bits >> 2) & 0x1F;
    int mant = v.bits & 0x3;

    if (exp == 0x1F) {
        // inf or NaN
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return std::nanf("");
    }

    float value;
    if (exp == 0) {
        // subnormal: (-1)^s * 2^(-14) * 0.mant
        value = (float)mant / 4.0f * std::ldexp(1.0f, -14);
    } else {
        value = (1.0f + (float)mant / 4.0f) * std::ldexp(1.0f, exp - 15);
    }

    return sign ? -value : value;
}

fp8_e4m3 float_to_fp8_e4m3(float v) {
    init_fp8_lut();

    if (std::isnan(v)) return fp8_e4m3(0xFF);
    if (std::isinf(v)) return fp8_e4m3((v > 0) ? 0x7E : 0xFE);
    if (v == 0.0f) return fp8_e4m3((std::signbit(v) ? 0x80 : 0x00));

    // Search over all E4M3 values for the closest match
    int sign_bit = std::signbit(v) ? 0x80 : 0;
    float abs_v = std::fabs(v);

    uint8_t best = 0;
    float best_err = INFINITY;
    for (int i = 0; i < 256; i++) {
        if ((i & 0x80) != sign_bit) continue;
        float ref = g_e4m3_to_float[i];
        float err = std::fabs(abs_v - std::fabs(ref));
        if (err < best_err) {
            best_err = err;
            best = (uint8_t)i;
        }
    }

    return fp8_e4m3(best);
}

fp8_e5m2 float_to_fp8_e5m2(float v) {
    if (std::isnan(v)) return fp8_e5m2(0x7F);
    if (std::isinf(v)) return fp8_e5m2((v > 0) ? 0x7C : 0xFC);
    if (v == 0.0f) return fp8_e5m2((std::signbit(v) ? 0x80 : 0x00));

    int sign = (v < 0) ? 1 : 0;
    v = std::fabs(v);

    // E5M2: max normal = 1.11 * 2^15 = 57344
    // Find closest by checking all 256 E5M2 values
    uint8_t best = 0;
    float best_err = INFINITY;
    for (int i = 0; i < 256; i++) {
        if (((i >> 7) & 1) != sign) continue;
        float ref = fp8_e5m2_to_float(fp8_e5m2((uint8_t)i));
        float err = std::fabs(v - std::fabs(ref));
        if (err < best_err) {
            best_err = err;
            best = (uint8_t)i;
        }
    }

    return fp8_e5m2(best);
}

void fp8_e4m3_to_float_batch(const uint8_t* src, float* dst, int64_t n) {
    init_fp8_lut();
    for (int64_t i = 0; i < n; i++) {
        dst[i] = g_e4m3_to_float[src[i]];
    }
}

void float_to_fp8_e4m3_batch(const float* src, uint8_t* dst, int64_t n) {
    init_fp8_lut();
    for (int64_t i = 0; i < n; i++) {
        dst[i] = compute_float_to_e4m3(src[i]);
    }
}

// ============================================================
// INT4 dequantization
// ============================================================

// Unpack a single zero value from the packed qzeros tensor
// GPTQ zeros are packed the same way as weights: [num_groups, in_features // 2]
// For AWQ: zeros packed into int32, different layout — handled in loaders
static inline uint8_t unpack_zero(const PackedInt4Tensor& t, int32_t group, int32_t row) {
    // GPTQ layout: qzeros[group * (in_features/2) + row/2]
    // Each byte stores zeros for two consecutive rows
    int32_t idx = group * (t.in_features / 2) + row / 2;
    if (row % 2 == 0) {
        return t.qzeros[idx] & 0xF;
    } else {
        return (t.qzeros[idx] >> 4) & 0xF;
    }
}

std::vector<float> dequantize_int4_row(const PackedInt4Tensor& t, int32_t row) {
    std::vector<float> result(t.in_features, 0.0f);

    for (int32_t col = 0; col < t.in_features; col++) {
        // Determine which group this column belongs to
        int32_t group;
        if (t.desc_act && !t.g_idx.empty()) {
            group = t.g_idx[col];
        } else {
            group = col / t.group_size;
        }

        // Get scale for this (group, row)
        float scale = t.scales[(int64_t)group * t.out_features + row];

        // Unpack the INT4 weight value
        uint8_t w_int4 = unpack_int4(t, row, col);

        // Unpack the zero point for this (group, row)
        uint8_t zero_int4 = unpack_zero(t, group, row);

        // Dequantize: (w - z) * s
        result[col] = ((int)w_int4 - (int)zero_int4) * scale;
    }

    return result;
}

void dequantize_int4_row_range(const PackedInt4Tensor& t, int32_t row,
                                int32_t col_offset, int32_t count,
                                float* dst) {
    for (int32_t i = 0; i < count; i++) {
        int32_t col = col_offset + i;
        if (col >= t.in_features) { dst[i] = 0.0f; continue; }

        int32_t group;
        if (t.desc_act && !t.g_idx.empty()) {
            group = t.g_idx[col];
        } else {
            group = col / t.group_size;
        }

        float scale = t.scales[(int64_t)group * t.out_features + row];
        uint8_t w_int4 = unpack_int4(t, row, col);
        uint8_t zero_int4 = unpack_zero(t, group, row);

        dst[i] = ((int)w_int4 - (int)zero_int4) * scale;
    }
}

std::vector<float> dequantize_int4_full(const PackedInt4Tensor& t) {
    std::vector<float> result((int64_t)t.out_features * t.in_features, 0.0f);
    for (int32_t row = 0; row < t.out_features; row++) {
        auto row_data = dequantize_int4_row(t, row);
        std::copy(row_data.begin(), row_data.end(),
                  result.begin() + (int64_t)row * t.in_features);
    }
    return result;
}

// ============================================================
// FP8 tensor dequantization
// ============================================================

std::vector<float> FP8QuantizedTensor::dequantize_row(int32_t row) const {
    std::vector<float> result(in_features, 0.0f);
    const uint8_t* src = data.data() + (int64_t)row * in_features;
    float s = scale(row);
    for (int32_t i = 0; i < in_features; i++) {
        result[i] = fp8_e4m3_to_float(fp8_e4m3(src[i])) * s;
    }
    return result;
}

void FP8QuantizedTensor::dequantize_row_range(int32_t row, int32_t col_offset, int32_t count,
                                               float* dst) const {
    const uint8_t* src = data.data() + (int64_t)row * in_features + col_offset;
    float s = scale(row);
    for (int32_t i = 0; i < count; i++) {
        dst[i] = fp8_e4m3_to_float(fp8_e4m3(src[i])) * s;
    }
}

// ============================================================
// QuantizedTensor methods
// ============================================================

std::vector<float> QuantizedTensor::to_float_vector() const {
    switch (type) {
        case Type::FP32: {
            if (!fp32_data) return {};
            return std::vector<float>(fp32_data, fp32_data + numel);
        }
        case Type::FP16: {
            if (!fp16_data) return {};
            std::vector<float> result(numel);
            for (int64_t i = 0; i < numel; i++) {
                // FP16 → float conversion
                uint16_t h = fp16_data[i];
                uint32_t sign = (h >> 15) & 0x1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t frac = h & 0x3FF;
                if (exp == 0) {
                    if (frac == 0) { result[i] = sign ? -0.0f : 0.0f; }
                    else {
                        frac <<= 1;
                        while ((frac & 0x400) == 0) { frac <<= 1; exp--; }
                        frac &= 0x3FF;
                        uint32_t e = (exp > 0) ? (uint32_t)exp - 9 : 0;
                        uint32_t bits = (sign << 31) | ((127 + e) << 23) | (frac << 13);
                        result[i] = *(float*)&bits;
                    }
                } else if (exp == 31) {
                    result[i] = sign ? -INFINITY : INFINITY;
                } else {
                    uint32_t bits = (sign << 31) | ((127 - 15 + exp) << 23) | (frac << 13);
                    result[i] = *(float*)&bits;
                }
            }
            return result;
        }
        case Type::FP8: {
            return fp8_tensor.dequantize_row(0);  // for whole tensor
        }
        case Type::INT4_GPTQ:
        case Type::INT4_AWQ: {
            return dequantize_int4_full(int4_tensor);
        }
    }
    return {};
}

std::vector<float> QuantizedTensor::dequantize_row(int32_t row) const {
    switch (type) {
        case Type::FP32: {
            int64_t cols = numel / ((int64_t)shape[0]);
            std::vector<float> result(cols);
            std::memcpy(result.data(), fp32_data + row * cols, cols * sizeof(float));
            return result;
        }
        case Type::FP8: {
            return fp8_tensor.dequantize_row(row);
        }
        case Type::INT4_GPTQ:
        case Type::INT4_AWQ: {
            return dequantize_int4_row(int4_tensor, row);
        }
        default:
            return {};
    }
}

void QuantizedTensor::dequantize_row_range(int32_t row, int32_t col_offset,
                                            int32_t count, float* dst) const {
    switch (type) {
        case Type::FP32: {
            int64_t cols = numel / ((int64_t)shape[0]);
            std::memcpy(dst, fp32_data + row * cols + col_offset, count * sizeof(float));
            break;
        }
        case Type::FP8: {
            fp8_tensor.dequantize_row_range(row, col_offset, count, dst);
            break;
        }
        case Type::INT4_GPTQ:
        case Type::INT4_AWQ: {
            dequantize_int4_row_range(int4_tensor, row, col_offset, count, dst);
            break;
        }
        default:
            std::memset(dst, 0, count * sizeof(float));
            break;
    }
}

// ============================================================
// Fused quantized matmul: C = A @ Q^T
// ============================================================

void quantized_matmul_transpose(
    const float* a, int32_t m, int32_t k,
    const QuantizedTensor& q, int32_t n,
    float* c)
{
    // C[m, n] = A[m, k] @ Q[n, k]^T
    // For each output element C[i, j]:
    //   C[i, j] = sum_{p=0}^{k-1} A[i, p] * dequant(Q[j, p])

    const int32_t BLK = 64;  // dequant buffer for tiling

    for (int32_t i = 0; i < m; i++) {
        std::fill(c + i * n, c + (i + 1) * n, 0.0f);

        // Tile over k dimension to limit dequant buffer
        for (int32_t p_start = 0; p_start < k; p_start += BLK) {
            int32_t p_end = std::min(p_start + BLK, k);
            int32_t tile_size = p_end - p_start;

            // Dequantize a tile of Q: [n, tile_size]
            // We do it row by row for each output dimension
            for (int32_t j = 0; j < n; j++) {
                // Dequantize Q[j, p_start:p_end]
                float q_tile[BLK];
                q.dequantize_row_range(j, p_start, tile_size, q_tile);

                // Accumulate dot product
                float sum = 0.0f;
                const float* a_tile = a + i * k + p_start;
                for (int32_t t = 0; t < tile_size; t++) {
                    sum += a_tile[t] * q_tile[t];
                }
                c[i * n + j] += sum;
            }
        }
    }
}
