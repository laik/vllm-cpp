#pragma once
#include "model.h"
#include "scheduler.h"
#include "paged_attention.h"
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>

// ============================================================
// PagedAttention Engine - ties together scheduler + KV manager
// + model forward pass for multi-request serving
// ============================================================

struct EngineOutput {
    std::vector<int32_t> request_ids;
    std::vector<int32_t> new_token_ids;
    std::vector<std::string> new_texts;
    std::vector<bool> finished;
    int32_t num_prompt_tokens = 0;
    int32_t num_generation_tokens = 0;
};

class PagedEngine {
public:
    explicit PagedEngine(const Qwen36Model& model,
                int32_t max_num_seqs,
                int32_t max_total_tokens,
                int32_t num_gpu_blocks = -1,
                int32_t num_cpu_blocks = 0);

    ~PagedEngine();

    int32_t submit(const std::string& prompt,
                   int32_t max_new_tokens = MAX_NEW_TOKENS,
                   float temperature = TEMPERATURE,
                   float top_p = TOP_P);
    int32_t submit_tokens(const std::vector<int32_t>& prompt_tokens,
                          int32_t max_new_tokens = MAX_NEW_TOKENS,
                          float temperature = TEMPERATURE,
                          float top_p = TOP_P);
    EngineOutput step();
    std::optional<Request> get_request(int32_t request_id) const;
    void remove_request(int32_t request_id);
    bool has_requests() const;
    Scheduler& scheduler() { return *scheduler_; }
    const Scheduler& scheduler() const { return *scheduler_; }

private:
    void run_prefill(Request& req);
    int32_t run_decode(Request& req);

    const Qwen36Model& model_;
    Scheduler* scheduler_{nullptr};

    float* q_buf_ = nullptr;
    float* k_buf_ = nullptr;
    float* v_buf_ = nullptr;
    float* attn_out_ = nullptr;
    float* hidden_out_ = nullptr;
    float* norm_buf_ = nullptr;
    float* mlp_buf_ = nullptr;
    float* gate_buf_ = nullptr;
    float* up_buf_ = nullptr;
    float* k_contiguous_ = nullptr;
    float* v_contiguous_ = nullptr;
};
