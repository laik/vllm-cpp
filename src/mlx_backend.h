#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include "model.h"  // For TokenizerConfig

// Configuration defaults
constexpr int32_t MLX_MAX_NEW_TOKENS = 256;
constexpr float MLX_TEMPERATURE = 0.7f;
constexpr float MLX_TOP_P = 0.8f;

// ============================================================
// We store MLX arrays opaquely to keep MLX headers out of this file.
// Each void* points to an mlx::core::array (heap-allocated).
// ============================================================

struct MlxModel {
    // Architecture (detected from weights)
    int32_t num_layers = 0;
    int32_t hidden_size = 0;
    int32_t num_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;
    int32_t ffn_intermediate = 0;
    int32_t vocab_size = 0;
    float rope_base = 1000000.0f;
    float norm_eps = 1e-6f;

    // Weights: name → mlx::core::array*
    std::map<std::string, void*> weights;

    // KV cache per layer: vector of {k, v} pairs
    // Each entry: pair<void*, void*> for (k_cache, v_cache)
    std::vector<std::pair<void*, void*>> kv_cache;

    // Tokenizer
    TokenizerConfig tokenizer;
    std::string model_path;

    ~MlxModel();
};

// Loading
MlxModel* mlx_load_model(const std::string& model_path);

// Generation with streaming callback
std::vector<int32_t> mlx_generate(
    MlxModel* model,
    const std::vector<int32_t>& prompt_tokens,
    int32_t max_new_tokens = MLX_MAX_NEW_TOKENS,
    float temperature = MLX_TEMPERATURE,
    float top_p = MLX_TOP_P,
    std::function<void(int32_t)> on_token = nullptr,
    bool debug = false
);
