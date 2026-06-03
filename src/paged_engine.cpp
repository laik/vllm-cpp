#include "paged_engine.h"
#include "paged_attention.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>

// ============================================================
// Inline math helpers
// ============================================================

static inline void rms_norm(const float* input, const float* weight, float* out,
                            int32_t dim, float eps) {
    float sum_sq = 0.0f;
    for (int32_t i = 0; i < dim; i++) sum_sq += input[i] * input[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / dim + eps);
    for (int32_t i = 0; i < dim; i++) out[i] = weight[i] * input[i] * inv_rms;
}

static inline void matmul_transpose(const float* a, int32_t m, int32_t k,
                                    const float* b, int32_t n, float* c) {
    for (int32_t i = 0; i < m; i++) {
        for (int32_t j = 0; j < n; j++) {
            float sum = 0.0f;
            const float* a_row = a + i * k;
            const float* b_row = b + j * k;
            for (int32_t p = 0; p < k; p++) sum += a_row[p] * b_row[p];
            c[i * n + j] = sum;
        }
    }
}

static inline void add_vectors(float* out, const float* a, const float* b, int32_t n) {
    for (int32_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static inline void add_bias(float* out, const float* bias, int32_t n) {
    for (int32_t i = 0; i < n; i++) out[i] += bias[i];
}

static inline float swiglu_func(float x) {
    return x * (1.0f / (1.0f + std::exp(-x)));
}

static inline void rope_single(float* x, int32_t dim, int32_t head_dim, int32_t pos,
                               const float* inv_freq, int32_t inv_freq_size) {
    int32_t num_heads = dim / head_dim;
    for (int32_t h = 0; h < num_heads; h++) {
        float* x_head = x + h * head_dim;
        for (int32_t i = 0; i < head_dim / 2; i++) {
            int32_t fi = std::min(i, inv_freq_size - 1);
            float freq = inv_freq[fi] * (float)pos;
            float cos_v = std::cos(freq);
            float sin_v = std::sin(freq);
            float x0 = x_head[2 * i], x1 = x_head[2 * i + 1];
            x_head[2 * i]     = x0 * cos_v - x1 * sin_v;
            x_head[2 * i + 1] = x0 * sin_v + x1 * cos_v;
        }
    }
}

static inline void compute_attention(
    float* out,
    const float* q, const float* k_cache, const float* v_cache,
    int32_t seq_len, int32_t num_heads, int32_t head_dim,
    int32_t num_kv_heads)
{
    int32_t heads_per_group = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt((float)head_dim);

    float* logits = new float[num_heads * seq_len];
    float* weights = new float[num_heads * seq_len];

    for (int32_t h = 0; h < num_heads; h++) {
        int32_t kv_h = h / heads_per_group;
        const float* q_h = q + h * head_dim;
        for (int32_t s = 0; s < seq_len; s++) {
            float sum = 0.0f;
            const float* k_h = k_cache + (s * num_kv_heads + kv_h) * head_dim;
            for (int32_t d = 0; d < head_dim; d++) sum += q_h[d] * k_h[d];
            logits[h * seq_len + s] = sum * scale;
        }
    }

    for (int32_t h = 0; h < num_heads; h++) {
        float max_l = -1e9f;
        for (int32_t s = 0; s < seq_len; s++)
            max_l = std::max(max_l, logits[h * seq_len + s]);
        float sum_exp = 0.0f;
        for (int32_t s = 0; s < seq_len; s++) {
            weights[h * seq_len + s] = std::exp(logits[h * seq_len + s] - max_l);
            sum_exp += weights[h * seq_len + s];
        }
        for (int32_t s = 0; s < seq_len; s++)
            weights[h * seq_len + s] /= (sum_exp + 1e-9f);
    }

    for (int32_t h = 0; h < num_heads; h++) {
        int32_t kv_h = h / heads_per_group;
        float* o_h = out + h * head_dim;
        std::fill(o_h, o_h + head_dim, 0.0f);
        for (int32_t s = 0; s < seq_len; s++) {
            float w = weights[h * seq_len + s];
            const float* v_h = v_cache + (s * num_kv_heads + kv_h) * head_dim;
            for (int32_t d = 0; d < head_dim; d++) o_h[d] += w * v_h[d];
        }
    }

    delete[] logits;
    delete[] weights;
}

static inline void moe_forward(
    float* out,
    const float* hidden, int32_t hidden_size,
    const Qwen36MoELayer& moe, int32_t top_k)
{
    int32_t ffn_dim = (int32_t)moe.experts[0].gate_proj.shape[0];
    float* gate_scores = new float[moe.num_experts]();
    for (int32_t e = 0; e < moe.num_experts; e++) {
        float sum = 0.0f;
        for (int32_t d = 0; d < hidden_size; d++)
            sum += moe.gate.data[e * hidden_size + d] * hidden[d];
        gate_scores[e] = sum;
    }
    float max_s = gate_scores[0];
    for (int32_t e = 1; e < moe.num_experts; e++) max_s = std::max(max_s, gate_scores[e]);
    float sum_e = 0.0f;
    for (int32_t e = 0; e < moe.num_experts; e++) {
        gate_scores[e] = std::exp(gate_scores[e] - max_s);
        sum_e += gate_scores[e];
    }
    for (int32_t e = 0; e < moe.num_experts; e++) gate_scores[e] /= (sum_e + 1e-9f);

    std::vector<std::pair<float, int32_t>> scored(moe.num_experts);
    for (int32_t i = 0; i < moe.num_experts; i++) scored[i] = {gate_scores[i], i};
    std::sort(scored.begin(), scored.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    std::fill(out, out + hidden_size, 0.0f);
    float* expert_out = new float[ffn_dim]();
    float* gate_act = new float[ffn_dim]();
    float* up_act = new float[ffn_dim]();

    for (int32_t k = 0; k < std::min(top_k, moe.num_experts); k++) {
        int32_t eid = scored[k].second;
        float score = scored[k].first;
        const auto& expert = moe.experts[eid];
        for (int32_t i = 0; i < ffn_dim; i++) {
            float gs = 0.0f, us = 0.0f;
            for (int32_t d = 0; d < hidden_size; d++) {
                gs += expert.gate_proj.data[i * hidden_size + d] * hidden[d];
                us += expert.up_proj.data[i * hidden_size + d] * hidden[d];
            }
            gate_act[i] = gs;
            up_act[i] = us;
        }
        for (int32_t i = 0; i < ffn_dim; i++)
            expert_out[i] = swiglu_func(gate_act[i]) * up_act[i];
        for (int32_t d = 0; d < hidden_size; d++) {
            float sum = 0.0f;
            for (int32_t i = 0; i < ffn_dim; i++)
                sum += expert.down_proj.data[d * ffn_dim + i] * expert_out[i];
            out[d] += sum * score;
        }
    }
    delete[] gate_scores;
    delete[] expert_out;
    delete[] gate_act;
    delete[] up_act;
}

// ============================================================
// PagedEngine implementation
// ============================================================

PagedEngine::PagedEngine(const Qwen36Model& model,
                         int32_t max_num_seqs,
                         int32_t max_total_tokens,
                         int32_t num_gpu_blocks,
                         int32_t num_cpu_blocks)
    : model_(model), scheduler_(nullptr)
{
    if (num_gpu_blocks < 0) {
        num_gpu_blocks = (MAX_CONTEXT + BLOCK_SIZE - 1) / BLOCK_SIZE * max_num_seqs;
    }

    int32_t kv_dim = model.num_kv_heads * model.head_dim;
    KVDtype kv_dtype = model.quant_config.kv_dtype;
    scheduler_ = new Scheduler(max_num_seqs, max_total_tokens,
                               model.num_layers, kv_dim,
                               num_gpu_blocks, num_cpu_blocks, kv_dtype);

    int32_t hidden = model.hidden_size;
    int32_t total_q_dim = model.num_heads * model.head_dim;
    int32_t kv_heads_dim = model.num_kv_heads * model.head_dim;
    int32_t ffn_dim = model.ffn_intermediate;

    q_buf_ = new float[total_q_dim]();
    k_buf_ = new float[kv_heads_dim]();
    v_buf_ = new float[kv_heads_dim]();
    attn_out_ = new float[total_q_dim]();
    hidden_out_ = new float[hidden]();
    norm_buf_ = new float[std::max(hidden, ffn_dim)]();
    mlp_buf_ = new float[ffn_dim]();
    gate_buf_ = new float[ffn_dim]();
    up_buf_ = new float[ffn_dim]();
    k_contiguous_ = new float[MAX_CONTEXT * kv_heads_dim]();
    v_contiguous_ = new float[MAX_CONTEXT * kv_heads_dim]();
}

PagedEngine::~PagedEngine() {
    delete scheduler_;
    delete[] q_buf_;
    delete[] k_buf_;
    delete[] v_buf_;
    delete[] attn_out_;
    delete[] hidden_out_;
    delete[] norm_buf_;
    delete[] mlp_buf_;
    delete[] gate_buf_;
    delete[] up_buf_;
    delete[] k_contiguous_;
    delete[] v_contiguous_;
}

int32_t PagedEngine::submit(const std::string& prompt,
                            int32_t max_new_tokens,
                            float temperature, float top_p) {
    auto tokens = tokenize(model_.tokenizer, prompt);
    if (tokens.empty()) tokens.insert(tokens.begin(), model_.tokenizer.bos_token_id);
    return scheduler_->submit_request(prompt, tokens,
                                      max_new_tokens, temperature, top_p);
}

int32_t PagedEngine::submit_tokens(const std::vector<int32_t>& prompt_tokens,
                                    int32_t max_new_tokens,
                                    float temperature, float top_p) {
    return scheduler_->submit_request("", prompt_tokens,
                                      max_new_tokens, temperature, top_p);
}

EngineOutput PagedEngine::step() {
    EngineOutput output;
    ScheduleOutput sched = scheduler_->schedule();
    if (sched.batch_ids.empty()) return output;

    std::vector<int32_t> all_new_tokens;
    std::vector<int32_t> all_rids;

    // Process each request in the batch
    for (size_t i = 0; i < sched.batch_ids.size(); i++) {
        int32_t rid = sched.batch_ids[i];
        auto req_opt = scheduler_->get_request(rid);
        if (!req_opt) continue;

        if (sched.is_prefill[i] || req_opt->block_table.empty()) {
            // We need mutable access - get from scheduler directly
            run_prefill(*req_opt);
            std::cerr << "[PAGED] Prefill done, req=" << rid
                      << " seq_len=" << req_opt->get_seq_len() << std::endl;
        } else {
            int32_t new_token = run_decode(*req_opt);
            all_new_tokens.push_back(new_token);
            all_rids.push_back(rid);
        }
    }

    // Report outputs back to scheduler
    if (!all_new_tokens.empty()) {
        std::vector<std::vector<int32_t>> token_lists;
        for (int32_t t : all_new_tokens) token_lists.push_back({t});
        scheduler_->report_output(all_rids, token_lists);
    }

    // Collect results
    for (size_t i = 0; i < all_rids.size(); i++) {
        auto req_opt = scheduler_->get_request(all_rids[i]);
        if (!req_opt) continue;
        int32_t tok = all_new_tokens[i];
        output.request_ids.push_back(all_rids[i]);
        output.new_token_ids.push_back(tok);
        output.new_texts.push_back(decode_token(model_.tokenizer, tok));
        output.finished.push_back(req_opt->state == RequestState::FINISHED || tok == EOS_TOKEN_ID);
    }

    for (int32_t rid : sched.finished_requests) {
        auto req_opt = scheduler_->get_request(rid);
        if (req_opt) {
            output.request_ids.push_back(rid);
            output.finished.push_back(true);
        }
    }

    output.num_prompt_tokens = sched.num_prompt_tokens;
    output.num_generation_tokens = (int32_t)sched.new_token_ids.size();
    return output;
}

std::optional<Request> PagedEngine::get_request(int32_t request_id) const {
    return scheduler_->get_request(request_id);
}

void PagedEngine::remove_request(int32_t request_id) {
    scheduler_->remove_request(request_id);
}

bool PagedEngine::has_requests() const {
    return scheduler_->num_waiting() > 0 || scheduler_->num_running() > 0;
}

// ============================================================
// Prefill: process all prompt tokens, fill KV blocks
// ============================================================

void PagedEngine::run_prefill(Request& req) {
    int32_t seq_len = (int32_t)req.prompt_tokens.size();
    if (seq_len <= 0) return;

    int32_t hidden = model_.hidden_size;
    int32_t num_heads = model_.num_heads;
    int32_t num_kv_heads = model_.num_kv_heads;
    int32_t head_dim = model_.head_dim;
    int32_t total_q_dim = num_heads * head_dim;
    int32_t kv_dim = num_kv_heads * head_dim;
    int32_t ffn_dim = model_.ffn_intermediate;
    auto& kv_mgr = scheduler_->kv_manager();

    // Allocate blocks if not already done
    if (req.block_table.empty()) {
        int32_t needed = req.get_num_blocks();
        kv_mgr.alloc_blocks(needed, req.block_table);
    }

    // Load embeddings for all positions
    float* all_hidden = new float[seq_len * hidden]();
    for (int32_t pos = 0; pos < seq_len; pos++) {
        int32_t id = req.prompt_tokens[pos];
        if (id >= 0 && id < model_.vocab_size) {
            const float* emb = model_.embeddings.data + id * hidden;
            std::copy(emb, emb + hidden, all_hidden + pos * hidden);
        }
    }

    // Process layer by layer, position by position
    for (int32_t l = 0; l < model_.num_layers; l++) {
        const auto& layer = model_.layers[l];

        for (int32_t pos = 0; pos < seq_len; pos++) {
            // Input norm
            rms_norm(all_hidden + pos * hidden, layer.input_norm.weight.data,
                     norm_buf_, hidden, layer.input_norm.eps);

            // K projection + RoPE + cache
            matmul_transpose(layer.attn.k_proj.data, kv_dim, hidden,
                             norm_buf_, 1, k_buf_);
            if (layer.attn.k_proj_bias.data)
                add_bias(k_buf_, layer.attn.k_proj_bias.data, kv_dim);
            rope_single(k_buf_, kv_dim, head_dim, pos,
                        model_.rope_inv_freq, model_.rope_inv_freq_size);
            write_kv_token(kv_mgr, req.block_table, pos + 1, l, k_buf_, v_buf_);

            // V projection + cache
            matmul_transpose(layer.attn.v_proj.data, kv_dim, hidden,
                             norm_buf_, 1, v_buf_);
            if (layer.attn.v_proj_bias.data)
                add_bias(v_buf_, layer.attn.v_proj_bias.data, kv_dim);
            write_kv_token(kv_mgr, req.block_table, pos + 1, l, k_buf_, v_buf_);

            // Q projection + RoPE
            matmul_transpose(layer.attn.q_proj.data, total_q_dim, hidden,
                             norm_buf_, 1, q_buf_);
            if (layer.attn.q_proj_bias.data)
                add_bias(q_buf_, layer.attn.q_proj_bias.data, total_q_dim);
            rope_single(q_buf_, total_q_dim, head_dim, pos,
                        model_.rope_inv_freq, model_.rope_inv_freq_size);

            // Attention: read from blocks into contiguous, compute
            read_kv_prefill(kv_mgr.block_pool(), req.block_table,
                            pos + 1, l, num_kv_heads, head_dim,
                            k_contiguous_, v_contiguous_);
            compute_attention(attn_out_, q_buf_, k_contiguous_, v_contiguous_,
                              pos + 1, num_heads, head_dim, num_kv_heads);

            // O projection + residual
            matmul_transpose(layer.attn.o_proj.data, hidden, total_q_dim,
                             attn_out_, 1, hidden_out_);
            add_vectors(all_hidden + pos * hidden,
                        all_hidden + pos * hidden, hidden_out_, hidden);

            // Post-attention norm
            rms_norm(all_hidden + pos * hidden, layer.attn_norm.weight.data,
                     norm_buf_, hidden, layer.attn_norm.eps);

            // MLP
            if (layer.moe.experts && layer.moe.gate.data) {
                moe_forward(mlp_buf_, norm_buf_, hidden, layer.moe, model_.top_k);
            } else if (layer.mlp.gate_proj.data) {
                matmul_transpose(layer.mlp.gate_proj.data, ffn_dim, hidden,
                                 norm_buf_, 1, gate_buf_);
                matmul_transpose(layer.mlp.up_proj.data, ffn_dim, hidden,
                                 norm_buf_, 1, up_buf_);
                for (int32_t i = 0; i < ffn_dim; i++)
                    mlp_buf_[i] = swiglu_func(gate_buf_[i]) * up_buf_[i];
                matmul_transpose(layer.mlp.down_proj.data, hidden, ffn_dim,
                                 mlp_buf_, 1, mlp_buf_);
            }

            // Residual
            add_vectors(all_hidden + pos * hidden,
                        all_hidden + pos * hidden, mlp_buf_, hidden);
        }
    }

    // Final norm on last position
    rms_norm(all_hidden + (seq_len - 1) * hidden, model_.final_norm.data,
             norm_buf_, hidden, model_.norm_eps);

    // LM head
    std::vector<float> logits(model_.vocab_size);
    matmul_transpose(model_.lm_head.data, model_.vocab_size, hidden,
                     norm_buf_, 1, logits.data());

    // Sample first generated token
    auto req_opt = scheduler_->get_request(req.request_id);
    if (req_opt) {
        req_opt->generated_tokens.push_back(logits.data() ? 
            sample_token(logits, TEMPERATURE, TOP_P, REPEAT_PENALTY, &req_opt->prompt_tokens) : -1);

        // Allocate new block if needed
        int32_t needed = req_opt->get_num_blocks();
        if ((int32_t)req_opt->block_table.size() < needed) {
            int32_t additional = needed - (int32_t)req_opt->block_table.size();
            std::vector<int32_t> new_blocks;
            if (kv_mgr.alloc_blocks(additional, new_blocks)) {
                req_opt->block_table.insert(req_opt->block_table.end(),
                                          new_blocks.begin(), new_blocks.end());
            }
        }
    }

    delete[] all_hidden;
}

// ============================================================
// Decode: generate one new token using PagedAttention
// ============================================================

int32_t PagedEngine::run_decode(Request& req) {
    int32_t seq_len = req.get_seq_len();
    int32_t hidden = model_.hidden_size;
    int32_t num_heads = model_.num_heads;
    int32_t num_kv_heads = model_.num_kv_heads;
    int32_t head_dim = model_.head_dim;
    int32_t total_q_dim = num_heads * head_dim;
    int32_t kv_dim = num_kv_heads * head_dim;
    int32_t ffn_dim = model_.ffn_intermediate;
    auto& kv_mgr = scheduler_->kv_manager();

    int32_t input_id;
    if (!req.generated_tokens.empty()) {
        input_id = req.generated_tokens.back();
    } else {
        input_id = req.prompt_tokens.empty() ? 0 : req.prompt_tokens.back();
    }

    const float* emb = model_.embeddings.data + input_id * hidden;
    std::copy(emb, emb + hidden, hidden_out_);

    for (int32_t l = 0; l < model_.num_layers; l++) {
        const auto& layer = model_.layers[l];

        // Input norm
        rms_norm(hidden_out_, layer.input_norm.weight.data,
                 norm_buf_, hidden, layer.input_norm.eps);

        // K projection + RoPE + write to block
        matmul_transpose(layer.attn.k_proj.data, kv_dim, hidden,
                         norm_buf_, 1, k_buf_);
        if (layer.attn.k_proj_bias.data)
            add_bias(k_buf_, layer.attn.k_proj_bias.data, kv_dim);
        rope_single(k_buf_, kv_dim, head_dim, seq_len - 1,
                    model_.rope_inv_freq, model_.rope_inv_freq_size);
        write_kv_token(kv_mgr, req.block_table, seq_len, l, k_buf_, v_buf_);

        // V projection + write to block
        matmul_transpose(layer.attn.v_proj.data, kv_dim, hidden,
                         norm_buf_, 1, v_buf_);
        if (layer.attn.v_proj_bias.data)
            add_bias(v_buf_, layer.attn.v_proj_bias.data, kv_dim);
        write_kv_token(kv_mgr, req.block_table, seq_len, l, k_buf_, v_buf_);

        // Q projection + RoPE
        matmul_transpose(layer.attn.q_proj.data, total_q_dim, hidden,
                         norm_buf_, 1, q_buf_);
        if (layer.attn.q_proj_bias.data)
            add_bias(q_buf_, layer.attn.q_proj_bias.data, total_q_dim);
        rope_single(q_buf_, total_q_dim, head_dim, seq_len - 1,
                    model_.rope_inv_freq, model_.rope_inv_freq_size);

        // >>> PagedAttention <<<
        std::fill(attn_out_, attn_out_ + total_q_dim, 0.0f);
        paged_attention(q_buf_, req.block_table, seq_len,
                        kv_mgr.block_pool(),
                        num_heads, num_kv_heads, head_dim, l,
                        attn_out_);

        // O projection + residual
        matmul_transpose(layer.attn.o_proj.data, hidden, total_q_dim,
                         attn_out_, 1, hidden_out_);
        add_vectors(hidden_out_, hidden_out_, hidden_out_, hidden);

        // Post-attention norm
        rms_norm(hidden_out_, layer.attn_norm.weight.data,
                 norm_buf_, hidden, layer.attn_norm.eps);

        // MLP
        if (layer.moe.experts && layer.moe.gate.data) {
            moe_forward(mlp_buf_, norm_buf_, hidden, layer.moe, model_.top_k);
        } else if (layer.mlp.gate_proj.data) {
            matmul_transpose(layer.mlp.gate_proj.data, ffn_dim, hidden,
                             norm_buf_, 1, gate_buf_);
            matmul_transpose(layer.mlp.up_proj.data, ffn_dim, hidden,
                             norm_buf_, 1, up_buf_);
            for (int32_t i = 0; i < ffn_dim; i++)
                mlp_buf_[i] = swiglu_func(gate_buf_[i]) * up_buf_[i];
            matmul_transpose(layer.mlp.down_proj.data, hidden, ffn_dim,
                             mlp_buf_, 1, mlp_buf_);
        }

        add_vectors(hidden_out_, hidden_out_, mlp_buf_, hidden);
    }

    // Final norm + LM head
    rms_norm(hidden_out_, model_.final_norm.data,
             norm_buf_, hidden, model_.norm_eps);

    std::vector<float> logits(model_.vocab_size);
    matmul_transpose(model_.lm_head.data, model_.vocab_size, hidden,
                     norm_buf_, 1, logits.data());

    std::vector<int32_t> full_history = req.prompt_tokens;
    full_history.insert(full_history.end(),
                        req.generated_tokens.begin(), req.generated_tokens.end());

    return sample_token(logits, req.temperature, req.top_p, REPEAT_PENALTY,
                        &full_history);
}
