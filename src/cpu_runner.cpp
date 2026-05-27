#include "model.h"
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>

// ============================================================
// CPU inference runner - validates the forward pass
// ============================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <model_path> [prompt]\n"
              << "\nOptions:\n"
              << "  <model_path>     Path to model directory (with .safetensors files)\n"
              << "  [prompt]         Input text (default: \"Hello\")\n"
              << "  --tokens IDs     Input as token IDs (comma-separated)\n"
              << "  --max-new N      Max new tokens (default: " << MAX_NEW_TOKENS << ")\n"
              << "  --temp T         Temperature (default: " << TEMPERATURE << ")\n"
              << "  --top-p P        Top-p (default: " << TOP_P << ")\n"
              << "  --debug          Print debug info per token\n"
              << "\nExample:\n"
              << "  " << prog << " /path/to/Qwen3-8B \"Hello world\"\n"
              << std::endl;
}

static std::vector<int32_t> parse_token_ids(const std::string& s) {
    std::vector<int32_t> ids;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (!token.empty()) ids.push_back(std::stoi(token));
    }
    return ids;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = "Hello";
    std::vector<int32_t> input_tokens;
    bool use_raw_tokens = false;
    int32_t max_new_tokens = MAX_NEW_TOKENS;
    float temperature = TEMPERATURE;
    float top_p = TOP_P;
    bool debug = false;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tokens" && i + 1 < argc) {
            input_tokens = parse_token_ids(argv[++i]);
            use_raw_tokens = true;
        } else if (arg == "--max-new" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg[0] != '-') {
            prompt = arg;
        }
    }

    // Load model
    std::cout << "=== vllm-cpp CPU Runner ===\n" << std::endl;
    auto start_load = std::chrono::steady_clock::now();
    Qwen36Model model = load_model_safetensors(model_path);
    auto end_load = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
    std::cout << "\nModel loaded in " << (int)load_ms << " ms\n" << std::endl;

    // Tokenize input
    std::vector<int32_t> tokens;
    if (use_raw_tokens) {
        tokens = input_tokens;
        std::cout << "Input tokens (raw): " << tokens.size() << " tokens" << std::endl;
    } else {
        // Add BOS token
        tokens.push_back(model.tokenizer.bos_token_id);
        auto word_tokens = tokenize(model.tokenizer, prompt);
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
        std::cout << "Input text: \"" << prompt << "\"" << std::endl;
        std::cout << "Tokenized: " << tokens.size() << " tokens" << std::endl;
        for (auto id : tokens) std::cout << "  " << id << std::endl;
    }

    // Run generation
    std::cout << "\nGenerating (max=" << max_new_tokens
              << ", temp=" << temperature
              << ", top_p=" << top_p << ")..." << std::endl;

    std::string generated_text;
    int32_t token_count = 0;
    auto start_gen = std::chrono::steady_clock::now();

    auto generated = generate_cpu(
        model, tokens, max_new_tokens, temperature, top_p,
        [&](int32_t token_id) {
            token_count++;
            auto tok_end = std::chrono::steady_clock::now();
            double tok_ms = std::chrono::duration<double, std::milli>(tok_end - start_gen).count();
            start_gen = tok_end;

            // Decode token
            std::string decoded;
            if (model.tokenizer.inv_vocab.count(token_id)) {
                decoded = model.tokenizer.inv_vocab[token_id];
            } else {
                decoded = "<" + std::to_string(token_id) + ">";
            }
            generated_text += decoded;

            if (debug) {
                std::cout << "  Token " << token_count
                          << ": id=" << token_id
                          << " text=\"" << decoded
                          << "\" time=" << (int)tok_ms << "ms"
                          << std::endl;
            } else {
                std::cout << decoded;
                std::cout.flush();
            }
        }
    );

    auto end_gen = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();

    std::cout << "\n\n--- Generation complete ---\n"
              << "Tokens generated: " << token_count << "\n"
              << "Total time: " << (int)total_ms << " ms\n";
    if (token_count > 0)
        std::cout << "Speed: " << (int)(total_ms / token_count) << " ms/token\n";

    // Cleanup model memory
    for (int32_t l = 0; l < model.num_layers; l++) {
        if (model.layers[l].moe.experts) {
            for (int32_t e = 0; e < model.layers[l].moe.num_experts; e++) {
                delete[] model.layers[l].moe.experts[e].gate_proj.data;
                delete[] model.layers[l].moe.experts[e].up_proj.data;
                delete[] model.layers[l].moe.experts[e].down_proj.data;
            }
            delete[] model.layers[l].moe.experts;
        }
    }
    delete[] model.embeddings.data;
    // Only delete lm_head if it doesn't share embeddings (tie_word_embeddings)
    if (model.lm_head.data != model.embeddings.data)
        delete[] model.lm_head.data;
    delete[] model.rope_inv_freq;

    return 0;
}
