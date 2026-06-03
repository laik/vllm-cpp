#pragma once
#include "config.h"
#include "quantization.h"
#include <vector>
#include <string>
#include <functional>
#include <memory>

// ============================================================
// Tensor storage - supports FP32, FP16, BF16, and quantized
// ============================================================
enum class Precision : int8_t {
    F32, F16, BF16,
    FP8_E4M3,       // FP8 E4M3 (W8A8)
    INT4_GPTQ,      // GPTQ INT4
    INT4_AWQ        // AWQ INT4
};

struct Tensor {
    std::vector<int64_t> shape;
    float* data = nullptr;       // always fp32 for CPU (legacy)
    void* gpu_data = nullptr;    // fp16/bf16 on GPU
    Precision precision = Precision::F32;
    int64_t numel = 0;
    int64_t byte_size = 0;

    // Quantized tensor (for GPTQ/AWQ/FP8 models)
    // Owned by this Tensor; caller must delete in destructor or manage externally
    QuantizedTensor* quant = nullptr;

    int64_t count() const { return numel; }
    int64_t elem_size() const {
        if (quant && quant->is_quantized()) {
            return 1;  // approximate
        }
        switch (precision) {
            case Precision::F16:
            case Precision::BF16: return 2;
            case Precision::FP8_E4M3: return 1;
            default: return 4;
        }
    }
    int64_t allocated_bytes() const { return byte_size; }

    bool is_quantized() const {
        return quant && quant->is_quantized();
    }

    void allocate_float(int64_t n) {
        numel = n;
        byte_size = n * sizeof(float);
        data = new float[n]();
    }

    std::vector<float> to_vector() const {
        if (quant && quant->is_quantized()) {
            return quant->to_float_vector();
        }
        if (!data || numel == 0) return {};
        return std::vector<float>(data, data + numel);
    }

    // Dequantize one row → [in_features] floats (for row-wise access)
    std::vector<float> dequantize_row(int32_t row) const {
        if (quant && quant->is_quantized()) {
            return quant->dequantize_row(row);
        }
        // FP32 fallback: copy row
        int64_t cols = numel / (int64_t)shape[0];
        return std::vector<float>(data + row * cols, data + (row + 1) * cols);
    }
};

// ============================================================
// Model layer structures - Qwen3.6 specific
// ============================================================

struct RMSNormParams {
    Tensor weight;    // [hidden_size]
    float eps = 1e-6f;
};

struct Qwen36AttnLayer {
    Tensor q_proj;    // [hidden_size, hidden_size]
    Tensor k_proj;    // [hidden_size, num_kv_heads * head_dim]
    Tensor v_proj;    // [hidden_size, num_kv_heads * head_dim]
    Tensor o_proj;    // [hidden_size, hidden_size]
    Tensor q_norm;    // [head_dim] - Qwen3 head dim norm
    Tensor k_norm;    // [head_dim] - Qwen3 head dim norm
    Tensor q_proj_bias; // [hidden_size] - Qwen2 bias
    Tensor k_proj_bias; // [num_kv_heads * head_dim]
    Tensor v_proj_bias; // [num_kv_heads * head_dim]
};

struct Qwen36FFNLayer {
    Tensor gate_proj; // [hidden_size, ffn_intermediate]
    Tensor up_proj;   // [hidden_size, ffn_intermediate]
    Tensor down_proj; // [ffn_intermediate, hidden_size]
};

struct Qwen36Expert {
    Tensor gate_proj; // [hidden_size, ffn_intermediate]
    Tensor up_proj;   // [hidden_size, ffn_intermediate]
    Tensor down_proj; // [ffn_intermediate, hidden_size]
};

struct Qwen36MoELayer {
    Tensor gate;              // [hidden_size, num_experts]
    Tensor score_correction;  // [num_experts]
    Qwen36Expert* experts = nullptr;
    int32_t num_experts = 0;
};

struct Qwen36TransformerLayer {
    RMSNormParams input_norm;
    Qwen36AttnLayer attn;
    RMSNormParams attn_norm;
    Qwen36FFNLayer mlp;       // Dense variant
    Qwen36MoELayer moe;       // MoE variant (gate.data is nullptr if not MoE)
};

// ============================================================
// Full Qwen3.6 model
// ============================================================
struct Qwen36Model {
    // Architecture
    int32_t num_layers = NUM_LAYERS;
    int32_t hidden_size = HIDDEN_SIZE;
    int32_t num_heads = NUM_HEADS;
    int32_t num_kv_heads = NUM_KV_HEADS;
    int32_t head_dim = HEAD_DIM;
    int32_t ffn_intermediate = FFN_INTERMEDIATE;
    int32_t vocab_size = VOCAB_SIZE;
    int32_t max_context = MAX_CONTEXT;
    int32_t num_experts = NUM_EXPERTS;
    int32_t top_k = TOP_K;

    // RoPE
    float* rope_inv_freq = nullptr;
    int32_t rope_inv_freq_size = 0;
    double rope_base = ROPE_BASE;
    float rope_theta = 1000000.0f;

    // RMSNorm epsilon
    float norm_eps = 1e-6f;

    // Embeddings
    Tensor embeddings;           // [vocab_size, hidden_size]
    Tensor lm_head;             // [vocab_size, hidden_size]
    Tensor final_norm;          // [hidden_size]

    // Transformer layers
    std::vector<Qwen36TransformerLayer> layers;

    // Tokenizer
    TokenizerConfig tokenizer;

    // Quantization config
    QuantConfig quant_config;

    // Metadata
    std::string model_path;
    std::string architecture;
    bool is_moe = false;

    // Memory tracking
    int64_t total_params = 0;
    int64_t total_bytes = 0;
};

// ============================================================
// Model loading
// ============================================================
Qwen36Model load_model_safetensors(const std::string& model_path);

// ============================================================
// Inference
// ============================================================
std::vector<float> forward_cpu(
    const Qwen36Model& model,
    const std::vector<int32_t>& input_ids,
    int32_t batch_size = 1,
    bool debug = false
);

int32_t sample_token(
    const std::vector<float>& logits,
    float temperature = TEMPERATURE,
    float top_p = TOP_P,
    float repeat_penalty = REPEAT_PENALTY,
    const std::vector<int32_t>* history = nullptr
);

std::vector<int32_t> generate_cpu(
    const Qwen36Model& model,
    const std::vector<int32_t>& prompt_tokens,
    int32_t max_new_tokens = MAX_NEW_TOKENS,
    float temperature = TEMPERATURE,
    float top_p = TOP_P,
    std::function<void(int32_t)> on_token = nullptr
);

// ============================================================
// Tokenizer (minimal)
// ============================================================
void load_vocab(const std::string& path,
                std::map<std::string, int32_t>& vocab,
                std::map<int32_t, std::string>& inv_vocab);

void load_added_tokens(const std::string& model_path,
                       std::map<std::string, int32_t>& vocab,
                       std::map<int32_t, std::string>& inv_vocab);

std::vector<int32_t> tokenize(const TokenizerConfig& tok, const std::string& text);
std::vector<std::string> decode_tokens(const TokenizerConfig& tok, const std::vector<int32_t>& ids);
std::string decode_token(const TokenizerConfig& tok, int32_t id);

// PagedAttention structures moved to kv_cache_manager.h
