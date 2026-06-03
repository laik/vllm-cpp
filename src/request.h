#include "config.h"
#include <cstdint>
#include <vector>
#include <string>

// ============================================================
// Request states
// ============================================================
enum class RequestState {
    WAITING,     // Waiting for enough blocks to be allocated
    RUNNING,     // Currently in the batch
    FINISHED,    // Generated EOS or max tokens
    PREEMPTED    // Preempted due to memory pressure (can be restored)
};

// ============================================================
// Single request (one prompt + generation)
// ============================================================
struct Request {
    int32_t request_id = -1;
    std::string prompt;
    std::vector<int32_t> prompt_tokens;   // Tokenized prompt
    std::vector<int32_t> generated_tokens; // Generated so far

    // Block table: maps logical block index -> physical block id
    std::vector<int32_t> block_table;

    // Checkpoint block tables for preemption restore
    std::vector<int32_t> checkpoint_blocks;

    RequestState state = RequestState::WAITING;
    int32_t preempt_count = 0;  // How many times this request has been preempted

    // Generation params
    int32_t max_new_tokens = MAX_NEW_TOKENS;
    float temperature = TEMPERATURE;
    float top_p = TOP_P;
    int32_t seed = 0;

    // Computed properties
    int32_t get_seq_len() const {
        return (int32_t)(prompt_tokens.size() + generated_tokens.size());
    }
    int32_t get_num_blocks() const {
        int32_t total_tokens = get_seq_len();
        return (total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
    int32_t get_last_block_offset() const {
        int32_t total_tokens = get_seq_len();
        return total_tokens % BLOCK_SIZE;
    }
    bool is_prefix_shared() const {
        return block_table.size() > 0;
    }
};
