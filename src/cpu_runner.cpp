#include "model.h"
#include "paged_engine.h"
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include <thread>

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <model_path> [prompt]\n"
              << "\nOptions:\n"
              << "  <model_path>     Path to model directory (with .safetensors files)\n"
              << "  [prompt]         Input text (default: \"Hello\")\n"
              << "  --tokens IDs     Input as raw token IDs (comma-separated)\n"
              << "  --max-new N      Max new tokens (default: " << MAX_NEW_TOKENS << ")\n"
              << "  --temp T         Temperature (default: " << TEMPERATURE << ")\n"
              << "  --top-p P        Top-p (default: " << TOP_P << ")\n"
              << "  --paged          Use PagedAttention engine (block-based KV cache)\n"
              << "  --max-batch N    Max concurrent requests (default: 4)\n"
              << "  --max-tokens N   Max total tokens in batch (default: 4096)\n"
              << "  --debug          Print debug info per token\n"
              << "\nExample:\n"
              << "  " << prog << " /path/to/Qwen3-8B \"Hello world\"\n"
              << "  " << prog << " --paged /path/to/Qwen3-8B \"Hello world\"\n"
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
    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::string model_path;
    std::string prompt = "Hello";
    std::vector<int32_t> input_tokens;
    bool use_raw_tokens = false;
    int32_t max_new_tokens = MAX_NEW_TOKENS;
    float temperature = TEMPERATURE, top_p = TOP_P;
    bool use_paged = false;
    int32_t max_batch = 4, max_total_tokens = 8192;
    bool debug = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tokens" && i+1 < argc) { input_tokens = parse_token_ids(argv[++i]); use_raw_tokens = true; }
        else if (arg == "--max-new" && i+1 < argc) { max_new_tokens = std::stoi(argv[++i]); }
        else if (arg == "--temp" && i+1 < argc) { temperature = std::stof(argv[++i]); }
        else if (arg == "--top-p" && i+1 < argc) { top_p = std::stof(argv[++i]); }
        else if (arg == "--paged") { use_paged = true; }
        else if (arg == "--max-batch" && i+1 < argc) { max_batch = std::stoi(argv[++i]); }
        else if (arg == "--max-tokens" && i+1 < argc) { max_total_tokens = std::stoi(argv[++i]); }
        else if (arg == "--debug") { debug = true; }
        else {
            if (model_path.empty()) model_path = arg; else prompt = arg;
        }
    }
    if (model_path.empty()) { print_usage(argv[0]); return 1; }

    std::cout << "=== vllm-cpp CPU Runner ===\n" << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    Qwen36Model model = load_model_safetensors(model_path);
    std::cout << "\nModel loaded in " << (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count() << " ms\n" << std::endl;

    std::vector<int32_t> tokens;
    if (use_raw_tokens) {
        tokens = input_tokens;
        std::cout << "Input tokens (raw): " << tokens.size() << " tokens\n";
    } else {
        tokens.push_back(model.tokenizer.bos_token_id);
        auto wt = tokenize(model.tokenizer, prompt);
        tokens.insert(tokens.end(), wt.begin(), wt.end());
        std::cout << "Input text: \"" << prompt << "\"\n"
                  << "Tokenized: " << tokens.size() << " tokens\n";
        for (auto id : tokens) std::cout << "  " << id << "\n";
    }

    if (use_paged) {
        std::cerr << "\n=== PagedAttention Engine ===\n  max_batch=" << max_batch
                  << " max_tokens=" << max_total_tokens << "\n";

        PagedEngine engine(model, max_batch, max_total_tokens);
        int32_t rid = engine.submit_tokens(tokens, max_new_tokens, temperature, top_p);
        int32_t gc = 0; auto sg = std::chrono::steady_clock::now();

        for (int32_t i = 0; i < max_new_tokens; i++) {
            auto out = engine.step();
            if (out.new_token_ids.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue; }
            for (size_t k = 0; k < out.request_ids.size(); k++) {
                if (out.request_ids[k] != rid) continue;
                if (k < out.new_texts.size()) {
                    gc++; auto te = std::chrono::steady_clock::now();
                    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(te-sg).count(); sg = te;
                    if (debug) { std::cerr << "  Token " << gc << ": id=" << out.new_token_ids[k]
                                << " text=\"" << out.new_texts[k] << "\" time=" << (int)ms << "ms\n"; }
                    else { std::cout << out.new_texts[k]; std::cout.flush(); }
                }
                if (k < out.finished.size() && out.finished[k]) goto pdone;
            }
            auto ro = engine.get_request(rid);
            if (ro && ro->state == RequestState::FINISHED) goto pdone;
        }
    pdone:;
        std::cout << "\n\n--- PagedAttention generation complete ---\n"
                  << "Tokens generated: " << gc << "\n"
                  << "Total time: " << (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-sg).count() << " ms\n";
        if (gc > 0) std::cout << "Speed: ... ms/token\n";
    } else {
        std::cout << "\nGenerating (max=" << max_new_tokens << ", temp=" << temperature
                  << ", top_p=" << top_p << ")..." << std::endl;
        int32_t gc = 0; auto sg = std::chrono::steady_clock::now();
        generate_cpu(model, tokens, max_new_tokens, temperature, top_p,
            [&](int32_t tid) {
                gc++; auto te = std::chrono::steady_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::milliseconds>(te-sg).count(); sg = te;
                std::string decoded;
                if (model.tokenizer.inv_vocab.count(tid)) decoded = model.tokenizer.inv_vocab[tid];
                else decoded = "<" + std::to_string(tid) + ">";
                if (debug) { std::cout << "  Token " << gc << ": id=" << tid << " text=\"" << decoded
                            << "\" time=" << (int)ms << "ms\n"; }
                else { std::cout << decoded; std::cout.flush(); }
            });
        std::cout << "\n\n--- Generation complete ---\nTokens generated: " << gc << "\n";
    }

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
    if (model.lm_head.data != model.embeddings.data) delete[] model.lm_head.data;
    delete[] model.rope_inv_freq;
    return 0;
}
