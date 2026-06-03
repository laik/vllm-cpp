#pragma once
#include "request.h"
#include "kv_cache_manager.h"
#include <vector>
#include <queue>
#include <map>
#include <mutex>
#include <optional>
#include <cstdint>

// ============================================================
// Scheduling result for one iteration
// ============================================================
struct ScheduleOutput {
    std::vector<int32_t> batch_ids;           // Request IDs in the batch
    std::vector<int32_t> batch_seq_lens;      // Sequence lengths for each
    std::vector<const std::vector<int32_t>*> batch_block_tables; // Block tables
    std::vector<int32_t> new_token_ids;       // Newly generated tokens (one per running request)
    std::vector<bool> is_prefill;             // true = prefill phase, false = decode phase
    std::vector<int32_t> finished_requests;   // Request IDs that finished this iteration
    int32_t num_prompt_tokens = 0;            // Total prompt tokens processed
    int32_t num_generation_tokens = 0;        // Total generated tokens
};

// ============================================================
// Scheduler: implements continuous batching with preemption
// ============================================================
class Scheduler {
public:
    Scheduler(int32_t max_num_seqs, int32_t max_total_tokens,
              int32_t num_layers, int32_t kv_dim,
              int32_t num_gpu_blocks, int32_t num_cpu_blocks,
              KVDtype kv_dtype = KVDtype::FP32);

    // Submit a new request
    int32_t submit_request(const std::string& prompt,
                           const std::vector<int32_t>& prompt_tokens,
                           int32_t max_new_tokens = MAX_NEW_TOKENS,
                           float temperature = TEMPERATURE,
                           float top_p = TOP_P);

    // Remove a request (cancel)
    void remove_request(int32_t request_id);

    // Run one scheduling iteration
    ScheduleOutput schedule();

    // Report generation results (new tokens) back to scheduler
    void report_output(const std::vector<int32_t>& request_ids,
                       const std::vector<std::vector<int32_t>>& new_token_ids);

    // Get request by ID
    std::optional<Request> get_request(int32_t request_id) const;

    // Get all requests
    const std::map<int32_t, Request>& get_all_requests() const { return requests_; }

    // KV cache manager access
    KVCacheManager& kv_manager() { return kv_manager_; }
    const KVCacheManager& kv_manager() const { return kv_manager_; }

    // Statistics
    int32_t num_waiting() const;
    int32_t num_running() const;
    int32_t num_swapped() const;
    int32_t num_finished() const;

private:
    // Admission control: can we fit these requests?
    bool can_allocate(const Request& req) const;

    // Allocate blocks for a request (moves WAITING -> RUNNING)
    bool allocate_request(Request& req);

    // Preemption: evict requests to make room
    void preempt_requests(int32_t needed_blocks);

    // Free blocks for finished/preempted requests
    void free_request_blocks(Request& req);

    // Checkpoint a running request (save blocks for preemption)
    void checkpoint_request(Request& req);

    // Restore a preempted request from checkpoint
    bool restore_request(Request& req);

    // Max number of simultaneous sequences
    int32_t max_num_seqs_;

    // Max total tokens across all sequences (limits batch size)
    int32_t max_total_tokens_;

    // Requests by ID
    std::map<int32_t, Request> requests_;
    int32_t next_request_id_ = 0;

    // KV cache management (GPU + CPU swap)
    KVCacheManager kv_manager_;

    // Running request order (FIFO)
    std::vector<int32_t> running_;

    // Waiting queue (FIFO)
    std::vector<int32_t> waiting_;

    mutable std::mutex mtx_;
};
