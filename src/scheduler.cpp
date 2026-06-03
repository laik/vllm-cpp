#include "scheduler.h"
#include <algorithm>
#include <cstdio>
#include <numeric>

// ============================================================
// Scheduler
// ============================================================

Scheduler::Scheduler(int32_t max_num_seqs,
                     int32_t max_total_tokens,
                     int32_t num_layers,
                     int32_t kv_dim,
                     int32_t num_gpu_blocks,
                     int32_t num_cpu_blocks,
                     KVDtype kv_dtype)
    : max_num_seqs_(max_num_seqs),
      max_total_tokens_(max_total_tokens),
      kv_manager_(num_layers, kv_dim, num_gpu_blocks + num_cpu_blocks, kv_dtype)
{
}

int32_t Scheduler::submit_request(const std::string& prompt,
                                  const std::vector<int32_t>& prompt_tokens,
                                  int32_t max_new_tokens,
                                  float temperature,
                                  float top_p)
{
    std::lock_guard<std::mutex> lock(mtx_);

    Request req;
    req.request_id = next_request_id_++;
    req.prompt = prompt;
    req.prompt_tokens = prompt_tokens;
    req.max_new_tokens = max_new_tokens;
    req.temperature = temperature;
    req.top_p = top_p;
    req.state = RequestState::WAITING;
    req.preempt_count = 0;

    requests_[req.request_id] = std::move(req);
    waiting_.push_back(req.request_id);
    return req.request_id;
}

void Scheduler::remove_request(int32_t request_id) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = requests_.find(request_id);
    if (it == requests_.end()) return;

    Request& req = it->second;
    free_request_blocks(req);
    req.state = RequestState::FINISHED;

    // Remove from queues
    waiting_.erase(
        std::remove(waiting_.begin(), waiting_.end(), request_id),
        waiting_.end());
    running_.erase(
        std::remove(running_.begin(), running_.end(), request_id),
        running_.end());

    requests_.erase(it);
}

bool Scheduler::can_allocate(const Request& req) const {
    int32_t needed = req.get_num_blocks();
    int32_t free_blocks = kv_manager_.free_blocks_count();
    return needed <= free_blocks &&
           (int32_t)running_.size() + (req.state == RequestState::WAITING ? 1 : 0) <= max_num_seqs_;
}

bool Scheduler::allocate_request(Request& req) {
    int32_t needed = req.get_num_blocks();
    if (needed <= 0) return true;

    auto blocks = std::vector<int32_t>();
    if (!kv_manager_.alloc_blocks(needed, blocks)) {
        return false;
    }
    req.block_table = std::move(blocks);
    req.state = RequestState::RUNNING;
    return true;
}

void Scheduler::preempt_requests(int32_t needed_blocks) {
    // Evict from the end of running queue (youngest first) until we have enough blocks
    int32_t freed = 0;
    std::vector<int32_t> to_preempt;
    for (int32_t i = (int32_t)running_.size() - 1; i >= 0 && freed < needed_blocks; i--) {
        int32_t rid = running_[i];
        auto it = requests_.find(rid);
        if (it == requests_.end()) continue;

        Request& req = it->second;
        if (req.state != RequestState::RUNNING) continue;

        // Checkpoint before preempting
        checkpoint_request(req);
        free_request_blocks(req);
        req.state = RequestState::PREEMPTED;
        req.preempt_count++;

        to_preempt.push_back(rid);
        freed += req.get_num_blocks();
    }

    // Remove preempted from running_
    for (int32_t rid : to_preempt) {
        running_.erase(
            std::remove(running_.begin(), running_.end(), rid),
            running_.end());
        // Add to front of waiting for priority restore
        waiting_.insert(waiting_.begin(), rid);
    }
}

void Scheduler::free_request_blocks(Request& req) {
    if (!req.block_table.empty()) {
        kv_manager_.free_blocks(req.block_table);
        req.block_table.clear();
    }
    if (!req.checkpoint_blocks.empty()) {
        // CPU checkpoint blocks are freed
        req.checkpoint_blocks.clear();
    }
}

void Scheduler::checkpoint_request(Request& req) {
    // For CPU implementation, we copy blocks to CPU storage
    // For GPU, this would swap to CPU memory
    // Simple approach: save the block ids as checkpoint
    req.checkpoint_blocks = req.block_table;
}

bool Scheduler::restore_request(Request& req) {
    if (req.checkpoint_blocks.empty()) {
        return false;
    }

    // Re-allocate blocks
    if (!allocate_request(req)) {
        return false;
    }

    // Copy checkpoint blocks to new blocks
    if (!req.checkpoint_blocks.empty() && req.block_table.size() == req.checkpoint_blocks.size()) {
        kv_manager_.copy_blocks(req.checkpoint_blocks, req.block_table);
    }

    return true;
}

ScheduleOutput Scheduler::schedule() {
    std::lock_guard<std::mutex> lock(mtx_);
    ScheduleOutput output;

    // Phase 1: Admit waiting requests that can fit
    std::vector<int32_t> newly_admitted;
    {
        std::vector<int32_t> next_waiting;
        for (int32_t rid : waiting_) {
            auto it = requests_.find(rid);
            if (it == requests_.end()) continue;
            Request& req = it->second;

            if (req.state == RequestState::PREEMPTED) {
                // Try to restore
                if (can_allocate(req)) {
                    if (restore_request(req)) {
                        running_.push_back(rid);
                        newly_admitted.push_back(rid);
                        continue;
                    }
                }
                next_waiting.push_back(rid);
            } else if (req.state == RequestState::WAITING) {
                if (can_allocate(req)) {
                    if (allocate_request(req)) {
                        running_.push_back(rid);
                        newly_admitted.push_back(rid);
                        continue;
                    }
                }
                next_waiting.push_back(rid);
            } else {
                next_waiting.push_back(rid);
            }
        }
        waiting_ = std::move(next_waiting);
    }

    // Phase 2: If not enough room, try preempting running requests
    int32_t total_needed = 0;
    for (int32_t rid : waiting_) {
        auto it = requests_.find(rid);
        if (it != requests_.end()) {
            total_needed += it->second.get_num_blocks();
        }
    }
    if (total_needed > 0 && !waiting_.empty()) {
        preempt_requests(total_needed);
        // Retry admission after preemption
        std::vector<int32_t> next_waiting;
        for (int32_t rid : waiting_) {
            auto it = requests_.find(rid);
            if (it == requests_.end()) continue;
            Request& req = it->second;
            if (can_allocate(req) && allocate_request(req)) {
                running_.push_back(rid);
                newly_admitted.push_back(rid);
            } else {
                next_waiting.push_back(rid);
            }
        }
        waiting_ = std::move(next_waiting);
    }

    // Phase 3: Build batch from running requests
    int32_t total_tokens = 0;
    for (int32_t rid : running_) {
        auto it = requests_.find(rid);
        if (it == requests_.end()) continue;
        Request& req = it->second;
        if (req.state != RequestState::RUNNING) continue;

        int32_t seq_len = req.get_seq_len();
        total_tokens += seq_len;

        // Check total tokens limit
        if (total_tokens > max_total_tokens_) {
            // Don't include this one in the batch
            continue;
        }

        output.batch_ids.push_back(rid);
        output.batch_seq_lens.push_back(seq_len);
        output.batch_block_tables.push_back(&req.block_table);
        output.is_prefill.push_back(
            newly_admitted.size() > 0 &&
            std::find(newly_admitted.begin(), newly_admitted.end(), rid) != newly_admitted.end());
    }

    // Phase 4: Free finished requests
    std::vector<int32_t> next_running;
    for (int32_t rid : running_) {
        auto it = requests_.find(rid);
        if (it == requests_.end()) continue;
        Request& req = it->second;

        if (req.state == RequestState::FINISHED ||
            (int32_t)req.generated_tokens.size() >= req.max_new_tokens) {
            // Mark as finished
            if (req.state != RequestState::FINISHED) {
                req.state = RequestState::FINISHED;
                output.finished_requests.push_back(rid);
            }
            free_request_blocks(req);
        } else {
            next_running.push_back(rid);
        }
    }
    running_ = std::move(next_running);

    // Statistics
    output.num_prompt_tokens = 0;
    output.num_generation_tokens = (int32_t)output.batch_ids.size();
    for (size_t i = 0; i < output.batch_ids.size(); i++) {
        if (output.is_prefill[i]) {
            auto it = requests_.find(output.batch_ids[i]);
            if (it != requests_.end()) {
                output.num_prompt_tokens += (int32_t)it->second.prompt_tokens.size();
            }
        }
    }

    return output;
}

void Scheduler::report_output(const std::vector<int32_t>& request_ids,
                              const std::vector<std::vector<int32_t>>& new_token_ids) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < request_ids.size(); i++) {
        int32_t rid = request_ids[i];
        auto it = requests_.find(rid);
        if (it == requests_.end()) continue;
        Request& req = it->second;
        req.generated_tokens.insert(req.generated_tokens.end(),
                                    new_token_ids[i].begin(),
                                    new_token_ids[i].end());

        // Check EOS
        for (int32_t tok : new_token_ids[i]) {
            if (tok == EOS_TOKEN_ID) {
                req.state = RequestState::FINISHED;
            }
        }

        // Allocate new blocks if needed
        int32_t needed = req.get_num_blocks();
        if ((int32_t)req.block_table.size() < needed) {
            int32_t additional = needed - (int32_t)req.block_table.size();
            auto new_blocks = std::vector<int32_t>();
            if (kv_manager_.alloc_blocks(additional, new_blocks)) {
                req.block_table.insert(req.block_table.end(),
                                       new_blocks.begin(),
                                       new_blocks.end());
            }
        }
    }
}

std::optional<Request> Scheduler::get_request(int32_t request_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = requests_.find(request_id);
    if (it == requests_.end()) return std::nullopt;
    return it->second;
}

int32_t Scheduler::num_waiting() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return (int32_t)waiting_.size();
}

int32_t Scheduler::num_running() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return (int32_t)running_.size();
}

int32_t Scheduler::num_swapped() const {
    std::lock_guard<std::mutex> lock(mtx_);
    int32_t count = 0;
    for (auto& [id, req] : requests_) {
        if (req.state == RequestState::PREEMPTED) count++;
    }
    return count;
}

int32_t Scheduler::num_finished() const {
    std::lock_guard<std::mutex> lock(mtx_);
    int32_t count = 0;
    for (auto& [id, req] : requests_) {
        if (req.state == RequestState::FINISHED) count++;
    }
    return count;
}
