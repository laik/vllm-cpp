#pragma once
#include <cstdint>
#include <string>
#include <map>
#include <vector>

// ============================================================
// Qwen3.6 model configuration - hardcoded architecture params
// ============================================================
//
// Supports Qwen3.6-35B-A3B (MoE) and Qwen3.6-8B (Dense) variants.
// Actual values from safetensors override these defaults at load time.
//
// Qwen3.6-35B-A3B (MoE variant):
//   - Total parameters: ~35B, Active: ~3.5B per forward pass
//   - 64 transformer layers
//   - Hidden dim: 5120
//   - 64 attention heads, 80-dim per head
//   - FFN intermediate: 12288 (SwiGLU)
//   - 36 experts, top-8 routing
//   - RMSNorm + RoPE + Yarn scaling
//
// Qwen3.6-8B (Dense variant):
//   - 36 layers, hidden: 5120, heads: 64, head_dim: 80
//   - FFN intermediate: 12288, no MoE
// ============================================================

// ----- Model architecture -----
constexpr int32_t    NUM_LAYERS          = 64;
constexpr int32_t    HIDDEN_SIZE         = 5120;
constexpr int32_t    NUM_HEADS           = 64;
constexpr int32_t    NUM_KV_HEADS        = 8;   // GQA: 8 groups
constexpr int32_t    HEAD_DIM            = 80;   // 5120 / 64
constexpr int32_t    FFN_INTERMEDIATE    = 12288;
constexpr int32_t    MAX_CONTEXT         = 32768;
constexpr int32_t    VOCAB_SIZE          = 151936;

// ----- MoE configuration (Qwen3.6-35B-A3B) -----
constexpr int32_t    NUM_EXPERTS         = 36;
constexpr int32_t    TOP_K               = 8;

// ----- Positional encoding -----
constexpr double     ROPE_BASE           = 1000000.0;
constexpr double     YARN_SCALE          = 8.0;
constexpr double     YARN_ORIG_CTX       = 32768.0;

// ----- Block management (PagedAttention) -----
constexpr int32_t    BLOCK_SIZE          = 16;
constexpr int32_t    NUM_BLOCKS          = MAX_CONTEXT / BLOCK_SIZE;
constexpr int32_t    MAX_SEQ_PER_BLOCK   = BLOCK_SIZE;

// ----- Quantization -----
constexpr bool       QUANT_ENABLED       = false;   // fp16/bf16 weight only
constexpr int32_t    QUANT_GROUP_SIZE    = 64;

// ----- Generation -----
constexpr float      TEMPERATURE         = 0.7f;
constexpr float      TOP_P               = 0.8f;
constexpr float      REPEAT_PENALTY      = 1.1f;
constexpr int32_t    MAX_NEW_TOKENS      = 2048;
constexpr int32_t    BOS_TOKEN_ID        = 151643;  // <|begin_of_text|>
constexpr int32_t    EOS_TOKEN_ID        = 151645;  // <|end_of_text|>

// ----- Runtime -----
constexpr int32_t    MAX_BATCH_SIZE      = 4;
constexpr int32_t    CUDA_DEVICE         = 0;

// ============================================================
// Tokenizer (minimal BPE-like, production should use tiktoken)
// ============================================================
struct TokenizerConfig {
    std::string vocab_path;                // path to merged vocab file
    std::map<std::string, int32_t> vocab;  // string -> token id
    std::map<int32_t, std::string> inv_vocab; // token id -> string
    std::vector<std::pair<std::string, std::string>> merges; // BPE merges
    int32_t unk_token_id = 151644;         // <|unk|>
    int32_t bos_token_id = BOS_TOKEN_ID;
    int32_t eos_token_id = EOS_TOKEN_ID;
};

// ============================================================
// safetensors header format
// ============================================================
struct TensorMeta {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::string dtype;     // "F32", "F16", "BF16"
    int64_t data_offsets[2]; // [start, end] byte offsets in file
};
