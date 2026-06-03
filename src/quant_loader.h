#pragma once
#include "config.h"
#include "quantization.h"
#include "model.h"
#include <string>
#include <map>
#include <vector>
#include <optional>

// ============================================================
// Quantized model loader — detects and loads GPTQ/AWQ/FP8 models
//
// Usage:
//   1. Parse quantization_config from model's config.json
//   2. For each safetensors file, if quantized:
//        - collect qweight/qzeros/scales tensors
//        - assemble into PackedInt4Tensor (GPTQ/AWQ) or FP8QuantizedTensor
//   3. Return QuantizedTensor for each weight matrix
// ============================================================

// Parse quantization_config from config.json content
QuantConfig parse_quant_config(const std::string& config_json);

// Load all quantized tensors for a model from safetensors directory.
// Returns a map: tensor_base_name -> QuantizedTensor
//   e.g., "model.layers.0.self_attn.q_proj" -> QuantizedTensor{INT4_GPTQ}
//
// For non-quantized models, returns empty map.
std::map<std::string, QuantizedTensor> load_quantized_tensors(
    const std::string& model_path,
    const QuantConfig& quant_config
);

// Dequantize all quantized tensors to FP32 (for CPU fallback inference).
// This uses more memory but allows the existing FP32 forward pass to work.
// Returns a map: tensor_base_name -> float* data (caller owns memory)
std::map<std::string, std::vector<float>> dequantize_all_tensors(
    const std::map<std::string, QuantizedTensor>& quant_tensors
);

// ============================================================
// Quantized matmul — used in forward pass for quantized weights
// ============================================================

// Compute: C = A @ W^T where W is quantized
// A: [m, k], W: [n, k] (quantized), C: [m, n]
// This fuses dequantization with the matmul.
void quantized_weight_matmul(
    const float* a, int32_t m, int32_t k,
    const QuantizedTensor& w_quant,  // [n, k] quantized
    int32_t n,
    float* c
);

// ============================================================
// FP8 activation quantization (online)
// ============================================================

// Quantize a float activation tensor to FP8 E4M3 with per-tensor scale.
// Returns the optimal scale to minimize quantization error.
struct FP8Activation {
    std::vector<uint8_t> data;  // FP8 E4M3 values
    float scale = 1.0f;
    int64_t numel = 0;
};

FP8Activation quantize_activation_fp8(const float* src, int64_t n);

// Dequantize FP8 activation back to float
void dequantize_activation_fp8(const FP8Activation& act, float* dst);
