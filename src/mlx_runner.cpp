#include "mlx_backend.h"
#include "config.h"
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

// ============================================================
// Structured logging to stderr
// ============================================================
static void log_info(const std::string& msg) {
    std::cerr << "[INFO] " << msg << std::endl;
}
static void log_warn(const std::string& msg) {
    std::cerr << "[WARN] " << msg << std::endl;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_path> [options]\n"
              << "\nOptions:\n"
              << "  <model_path>     Path to model directory\n"
              << "  --prompt TEXT    Single prompt (non-interactive)\n"
              << "  --max-new N      Max new tokens (default: " << MLX_MAX_NEW_TOKENS << ")\n"
              << "  --temp T         Temperature (default: " << MLX_TEMPERATURE << ")\n"
              << "  --top-p P        Top-p (default: " << MLX_TOP_P << ")\n"
              << "  --system TEXT    System prompt\n"
              << "\nInteractive mode: run without --prompt\n"
              << "  Type your message, Enter to submit\n"
              << "  'quit' or 'exit' to leave"
              << std::endl;
}

static std::string read_line() {
    std::string line;
    std::getline(std::cin, line);
    return line;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string single_prompt;
    std::string system_prompt;
    int32_t max_new = MLX_MAX_NEW_TOKENS;
    float temperature = MLX_TEMPERATURE;
    float top_p = MLX_TOP_P;
    bool debug = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            single_prompt = argv[++i];
        } else if (arg == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (arg == "--max-new" && i + 1 < argc) {
            max_new = std::stoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg[0] == '-') {
            log_warn("unknown flag: " + arg);
        }
    }

    // Load model
    log_info("loading MLX model from " + model_path + "...");
    auto start_load = std::chrono::steady_clock::now();
    MlxModel* model = mlx_load_model(model_path);
    auto end_load = std::chrono::steady_clock::now();
    int64_t load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count();

    log_info("loaded " + std::to_string(load_ms) + "ms  layers=" + std::to_string(model->num_layers) +
             " heads=" + std::to_string(model->num_heads) +
             " kv_heads=" + std::to_string(model->num_kv_heads) +
             " hidden=" + std::to_string(model->hidden_size) +
             " vocab=" + std::to_string(model->vocab_size));

    // Conversation state
    std::vector<std::string> user_msgs;
    std::vector<std::string> assistant_msgs;

    auto run_turn = [&](const std::string& user_input) {
        // Build prompt using token IDs with <|im_start|>/<|im_end|> format
        std::vector<int32_t> tokens;
        tokens.push_back(model->tokenizer.bos_token_id);

        // System message
        tokens.push_back(IM_START_TOKEN_ID);  // <|im_start|>
        std::string sys_text = system_prompt.empty()
            ? "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            : "system\n" + system_prompt;
        auto sys_tok = tokenize(model->tokenizer, sys_text);
        tokens.insert(tokens.end(), sys_tok.begin(), sys_tok.end());
        tokens.push_back(IM_END_TOKEN_ID);  // <|im_end|>
        auto nl = tokenize(model->tokenizer, "\n");
        tokens.insert(tokens.end(), nl.begin(), nl.end());

        // Conversation history
        for (size_t i = 0; i < user_msgs.size(); i++) {
            tokens.push_back(IM_START_TOKEN_ID);
            auto ut = tokenize(model->tokenizer, "user\n" + user_msgs[i]);
            tokens.insert(tokens.end(), ut.begin(), ut.end());
            tokens.push_back(IM_END_TOKEN_ID);
            tokens.insert(tokens.end(), nl.begin(), nl.end());

            if (i < assistant_msgs.size()) {
                tokens.push_back(IM_START_TOKEN_ID);
                auto at = tokenize(model->tokenizer, "assistant\n" + assistant_msgs[i]);
                tokens.insert(tokens.end(), at.begin(), at.end());
                tokens.push_back(IM_END_TOKEN_ID);
                tokens.insert(tokens.end(), nl.begin(), nl.end());
            }
        }

        // Current user message
        tokens.push_back(IM_START_TOKEN_ID);
        auto ut = tokenize(model->tokenizer, "user\n" + user_input);
        tokens.insert(tokens.end(), ut.begin(), ut.end());
        tokens.push_back(IM_END_TOKEN_ID);
        tokens.insert(tokens.end(), nl.begin(), nl.end());

        // Assistant start
        tokens.push_back(IM_START_TOKEN_ID);
        auto at = tokenize(model->tokenizer, "assistant\n");
        tokens.insert(tokens.end(), at.begin(), at.end());

        // Generate
        std::string reply;
        int32_t token_count = 0;
        auto start_gen = std::chrono::steady_clock::now();

        auto generated = mlx_generate(
            model, tokens, max_new, temperature, top_p,
            [&](int32_t token_id) {
                token_count++;
                std::string decoded = decode_token(model->tokenizer, token_id);
                if (debug) std::cerr << "[DEBUG] token=" << token_id << " decoded=\"" << decoded << "\"" << std::endl;
                reply += decoded;
                std::cout << decoded;
                std::cout.flush();
            },
            debug
        );

        auto end_gen = std::chrono::steady_clock::now();
        int64_t gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();

        std::cout << std::endl;
        log_info(std::to_string(token_count) + " tokens / " + std::to_string(gen_ms) + "ms");

        // Save to history
        user_msgs.push_back(user_input);
        assistant_msgs.push_back(reply);

        // Keep last 20 turns
        while (user_msgs.size() > 20) {
            user_msgs.erase(user_msgs.begin());
            assistant_msgs.erase(assistant_msgs.begin());
        }
    };

    if (!single_prompt.empty()) {
        // Non-interactive: one turn, exit
        run_turn(single_prompt);
    } else {
        // Interactive loop
        log_info("interactive mode — type 'quit' or 'exit' to leave");

        while (true) {
            std::cout << "> ";
            std::cout.flush();

            std::string line = read_line();
            if (std::cin.eof()) break;

            // Trim trailing whitespace
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
                line.pop_back();

            if (line.empty()) continue;
            if (line == "quit" || line == "exit") break;

            run_turn(line);
        }
    }

    delete model;
    log_info("bye");
    return 0;
}
