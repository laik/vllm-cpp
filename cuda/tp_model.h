#pragma once
#include <cuda_runtime.h>
#include <half.h>
#include <vector>
#include <string>
#include <cstdint>

// ============================================================
// GPU Model with Tensor Parallelism
// ============================================================
//
// Each GPU holds its own TP slice of the weights.
// All weights are stored as half* on the GPU.
//
// TP partitioning rules:
// - Column-parallel (Q, K, V, gate, up): split along output dim
// - Row-parallel (O, down): split along input dim
// - Shared across all ranks (embeddings, norms, lm_head, gate router)
// ============================================================

struct TPModelLayer {
    // RMSNorm weights (shared, not split)
    half* input_norm_w = nullptr;
    half* attn_norm_w = nullptr;

    // Attention projections
    half* q_proj_w = nullptr;  // [local_q_out, hidden]
    half* k_proj_w = nullptr;  // [local_k_out, hidden]
    half* v_proj_w = nullptr;  // [local_v_out, hidden]
    half* o_proj_w = nullptr;  // [hidden, local_o_in]
    half* q_proj_bias = nullptr;
    half* k_proj_bias = nullptr;
    half* v_proj_bias = nullptr;
    half* q_norm_w = nullptr;  // [head_dim]
    half* k_norm_w = nullptr;  // [head_dim]

    // Dense MLP
    half* gate_proj_w = nullptr;  // [local_ffn, hidden]
    half* up_proj_w = nullptr;    // [local_ffn, hidden]
    half* down_proj_w = nullptr;  // [hidden, local_ffn]

    // MoE (only if is_moe)
    half* moe_gate_w = nullptr;    // [hidden, num_experts] - shared
    half* moe_score_corr = nullptr; // [num_experts] - shared
    half** experts_gate_w = nullptr; // [num_experts][local_ffn, hidden]
    half** experts_up_w = nullptr;   // [num_experts][local_ffn, hidden]
    half** experts_down_w = nullptr; // [num_experts][hidden, local_ffn]
    int32_t num_experts = 0;
};

struct TPModel {
    // Architecture
    int32_t num_layers = 0;
    int32_t hidden_size = 0;
    int32_t num_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;
    int32_t ffn_intermediate = 0;
    int32_t vocab_size = 0;
    int32_t top_k = 2;

    // TP-local dimensions
    int32_t local_num_heads = 0;
    int32_t local_num_kv_heads = 0;
    int32_t local_q_out = 0;    // local_num_heads * head_dim
    int32_t local_k_out = 0;    // local_num_kv_heads * head_dim
    int32_t local_v_out = 0;    // local_num_kv_heads * head_dim
    int32_t local_o_in = 0;     // local_num_heads * head_dim
    int32_t local_ffn = 0;      // ffn_intermediate / tp_size

    // Shared weights (full size on each GPU)
    half* embeddings = nullptr;    // [vocab_size, hidden_size]
    half* lm_head = nullptr;       // [vocab_size, hidden_size]
    half* final_norm_w = nullptr;  // [hidden_size]

    // RoPE
    float* rope_inv_freq = nullptr;
    int32_t rope_inv_freq_size = 0;
    float rope_theta = 1000000.0f;
    float norm_eps = 1e-6f;

    // Layers
    std::vector<TPModelLayer> layers;

    // KV cache (per-layer, grows dynamically)
    half** k_cache = nullptr;  // [num_layers][max_seq * local_k_out]
    half** v_cache = nullptr;  // [num_layers][max_seq * local_v_out]
    int32_t* seq_lens = nullptr;  // [batch_size]

    bool is_moe = false;
    std::string model_path;
};

// Load model weights from safetensors, partitioned for TP
TPModel* tp_load_model(const std::string& model_path);

// Free all GPU memory
void tp_free_model(TPModel* model);

// TP-aware forward pass
void tp_forward_pass(
    TPModel* model,
    const int32_t* input_ids,  // [batch_size, seq_len]
    float* logits,             // [batch_size, vocab_size] - only on rank 0 or gathered
    int32_t batch_size,
    int32_t seq_len,
    cudaStream_t stream);

// TP-aware attention kernel (with all-gather for cross-GPU heads)
void tp_attention_forward(
    const half* q_proj_w, const half* k_proj_w, const half* v_proj_w,
    const half* o_proj_w,
    const half* q_bias, const half* k_bias, const half* v_bias,
    const half* q_norm_w, const half* k_norm_w,
    const float* rope_inv_freq, int32_t rope_inv_freq_size,
    const half* k_cache, const half* v_cache,
    float* attn_output,
    int32_t hidden_size,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t local_num_heads, int32_t local_num_kv_heads,
    int32_t local_q_out, int32_t local_k_out, int32_t local_v_out,
    int32_t local_o_in,
    int32_t seq_len, int32_t batch_size,
    const float* input,         // [batch_size, hidden_size] (normed)
    float* output,              // [batch_size, hidden_size]
    float* temp_buf,
    cudaStream_t stream);

// TP-aware dense MLP forward
void tp_dense_mlp_forward(
    const half* gate_proj_w, const half* up_proj_w, const half* down_proj_w,
    float* output,
    int32_t hidden_size,
    int32_t ffn_intermediate,
    int32_t local_ffn,
    const float* input,         // [batch_size, hidden_size] (normed)
    float* temp_buf,
    cudaStream_t stream);

// TP-aware MoE forward
void tp_moe_forward(
    const half* gate_w,
    const half** experts_gate, const half** experts_up, const half** experts_down,
    float* output,
    int32_t hidden_size,
    int32_t ffn_intermediate,
    int32_t local_ffn,
    int32_t num_experts,
    int32_t top_k,
    int32_t num_tokens,
    const float* input,
    float* temp_buf,
    cudaStream_t stream);
