#include "mlx_backend.h"
#include "config.h"
#include "device.h"
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

static void log_info(const std::string& msg) { std::cerr << "[INFO] " << msg << std::endl; }
static void log_warn(const std::string& msg) { std::cerr << "[WARN] " << msg << std::endl; }

static std::string read_line() { std::string line; std::getline(std::cin, line); return line; }

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [options]\n"
                  << "  --prompt TEXT   Single prompt\n"
                  << "  --max-new N     Max tokens (default: " << MLX_MAX_NEW_TOKENS << ")\n"
                  << "  --temp T        Temperature (default: " << MLX_TEMPERATURE << ")\n"
                  << "  --top-p P       Top-p (default: " << MLX_TOP_P << ")\n"
                  << "  --system TEXT   System prompt\n"
                  << "  --debug         Debug output\n";
        return 1;
    }

    std::string model_path = argv[1], single_prompt, system_prompt;
    int32_t max_new = MLX_MAX_NEW_TOKENS;
    float temperature = MLX_TEMPERATURE, top_p = MLX_TOP_P;
    bool debug = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) single_prompt = argv[++i];
        else if (arg == "--system" && i + 1 < argc) system_prompt = argv[++i];
        else if (arg == "--max-new" && i + 1 < argc) max_new = std::stoi(argv[++i]);
        else if (arg == "--temp" && i + 1 < argc) temperature = std::stof(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc) top_p = std::stof(argv[++i]);
        else if (arg == "--debug") debug = true;
        else if (arg[0] == '-') log_warn("unknown: " + arg);
    }

    DeviceInfo::print_info();

    log_info("loading model...");
    auto t0 = std::chrono::steady_clock::now();
    MlxModel* model = mlx_load_model(model_path);
    auto t1 = std::chrono::steady_clock::now();
    log_info("loaded in " + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()) + "ms  "
        "layers=" + std::to_string(model->num_layers) +
        " heads=" + std::to_string(model->num_heads) +
        " hidden=" + std::to_string(model->hidden_size));

    std::vector<std::string> user_msgs, assistant_msgs;

    auto run_turn = [&](const std::string& user_input) {
        // Build prompt with Qwen2.5 chat template
        std::vector<int32_t> tokens;

        // System message
        tokens.push_back(IM_START_TOKEN_ID);
        std::string sys = system_prompt.empty()
            ? "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            : system_prompt;
        auto sys_tok = tokenize(model->tokenizer, "system\n" + sys);
        tokens.insert(tokens.end(), sys_tok.begin(), sys_tok.end());
        tokens.push_back(IM_END_TOKEN_ID);
        auto nl = tokenize(model->tokenizer, "\n");
        tokens.insert(tokens.end(), nl.begin(), nl.end());

        // History
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
        auto t_gen = std::chrono::steady_clock::now();

        mlx_generate(model, tokens, max_new, temperature, top_p,
            [&](int32_t token_id) {
                token_count++;
                std::string decoded = decode_token(model->tokenizer, token_id);
                reply += decoded;
                std::cout << decoded;
                std::cout.flush();
            }, debug);

        auto t_done = std::chrono::steady_clock::now();
        int64_t gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_done - t_gen).count();

        std::cout << std::endl;
        log_info(std::to_string(token_count) + " tokens / " + std::to_string(gen_ms) + "ms");

        user_msgs.push_back(user_input);
        assistant_msgs.push_back(reply);
        while (user_msgs.size() > 20) { user_msgs.erase(user_msgs.begin()); assistant_msgs.erase(assistant_msgs.begin()); }
    };

    if (!single_prompt.empty()) {
        run_turn(single_prompt);
    } else {
        log_info("interactive mode — type 'quit' to exit");
        while (true) {
            std::cout << "\n> " << std::flush;
            std::string line = read_line();
            if (std::cin.eof()) break;
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line.empty()) continue;
            if (line == "quit" || line == "exit") break;
            run_turn(line);
        }
    }

    delete model;
    log_info("bye");
    return 0;
}
