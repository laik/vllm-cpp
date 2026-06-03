// ============================================================
// Unified HTTP server — OpenAI-compatible API
//
// Platform backends:
//   macOS (Metal/MLX):   uses mlx_backend.h, GPU-accelerated
//   Linux (CUDA/Paged):  uses paged_engine.h, continuous batching
//   CPU fallback:        uses model.h forward_cpu()
//
// Endpoints:
//   /v1/chat/completions  — Chat API (Qwen template)
//   /v1/completions       — Text completion API
//   /v1/models            — Model list
//   /health               — Health check
// ============================================================

#include "config.h"
#include "device.h"
#include "model.h"
#include "json_utils.h"
#include "http_utils.h"
#include "chat_prompt.h"

#ifdef HAS_MLX
#include "mlx_backend.h"
#define HAS_GPU_BACKEND 1
#else
#include "paged_engine.h"
#endif

#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <thread>
#include <atomic>
#include <ctime>
#include <vector>
#include <map>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

static void log_info(const std::string& msg) { std::cerr << "[INFO] " << msg << std::endl; }

// ============================================================
// Shared handlers
// ============================================================
static std::string handle_models(const std::string& model_id) {
    JsonValue m = JsonValue::object(); m.set("id", JsonValue::string(model_id)); m.set("object", JsonValue::string("model")); m.set("owned_by", JsonValue::string("vllm-cpp"));
    JsonValue arr = JsonValue::array(); arr.push(m);
    JsonValue resp = JsonValue::object(); resp.set("object", JsonValue::string("list")); resp.set("data", arr);
    return http_resp(200, json_render(resp), "application/json");
}

static std::string handle_health() {
    auto info = DeviceInfo::detect();
    JsonValue r = JsonValue::object();
    r.set("status", JsonValue::string("ok"));
    r.set("backend", JsonValue::string(info.name));
    r.set("gpu", JsonValue::boolean(info.gpu_available));
#ifdef HAS_MLX
    r.set("platform", JsonValue::string("macOS/MLX"));
#else
    r.set("platform", JsonValue::string("Linux/CUDA-CPU"));
#endif
    return http_resp(200, json_render(r), "application/json");
}

// ============================================================
// Backend-specific inference
// ============================================================

#ifdef HAS_MLX
// ----- macOS MLX backend -----
static std::string handle_completion_mlx(
    MlxModel* model, const HttpRequest& req, const std::string& model_id)
{
    bool stream = false;
    size_t sp = req.body.find("\"stream\"");
    if (sp != std::string::npos) {
        sp = req.body.find(':', sp + 8);
        if (sp != std::string::npos) {
            sp++; while (sp < req.body.size() && (req.body[sp] == ' ' || req.body[sp] == '\t')) sp++;
            if (sp < req.body.size() && req.body.compare(sp, 4, "true") == 0) stream = true;
        }
    }
    float temp = json_get_number(req.body, "temperature", 0.7f);
    float top_p = json_get_number(req.body, "top_p", 0.8f);
    int32_t max_tok = json_get_int(req.body, "max_tokens", 512);

    std::vector<int32_t> tokens;
    if (req.path.find("/v1/chat/completions") != std::string::npos)
        tokens = build_chat_prompt_tokens(model->tokenizer, req.body);
    else
        tokens = build_text_prompt_tokens(model->tokenizer, req.body);

    if (tokens.empty())
        return http_resp(400, R"({"error":{"message":"No valid prompt","type":"invalid_request_error"}})", "application/json");

    std::string id = req.path.find("chat") != std::string::npos
        ? "chatcmpl-" + std::to_string(time(nullptr))
        : "cmpl-" + std::to_string(time(nullptr));

    if (stream) {
        std::string response;
        bool first_chunk = true;

        mlx_generate(model, tokens, max_tok, temp, top_p,
            [&](int32_t token_id) {
                std::string decoded = decode_token(model->tokenizer, token_id);
                JsonValue chunk = JsonValue::object();
                chunk.set("id", JsonValue::string(id));
                chunk.set("object", JsonValue::string("chat.completion.chunk"));
                chunk.set("created", JsonValue::integer(time(nullptr)));
                chunk.set("model", JsonValue::string(model_id));
                JsonValue delta = JsonValue::object();
                if (first_chunk) { delta.set("role", JsonValue::string("assistant")); first_chunk = false; }
                delta.set("content", JsonValue::string(decoded));
                JsonValue choice = JsonValue::object();
                choice.set("index", JsonValue::integer(0));
                choice.set("delta", delta);
                JsonValue choices = JsonValue::array(); choices.push(choice);
                chunk.set("choices", choices);
                response += sse(json_render(chunk));
            }, false);

        JsonValue fin = JsonValue::object();
        fin.set("id", JsonValue::string(id));
        fin.set("object", JsonValue::string("chat.completion.chunk"));
        fin.set("created", JsonValue::integer(time(nullptr)));
        fin.set("model", JsonValue::string(model_id));
        JsonValue fd = JsonValue::object(); fd.set("content", JsonValue::string(""));
        JsonValue fc = JsonValue::object(); fc.set("index", JsonValue::integer(0)); fc.set("delta", fd); fc.set("finish_reason", JsonValue::string("stop"));
        JsonValue fcs = JsonValue::array(); fcs.push(fc); fin.set("choices", fcs);
        response += sse(json_render(fin));
        response += sse("[DONE]");
        return http_resp(200, response, "text/event-stream");
    } else {
        std::string full_text;
        auto gen = mlx_generate(model, tokens, max_tok, temp, top_p,
            [&](int32_t token_id) { full_text += decode_token(model->tokenizer, token_id); }, false);

        JsonValue choice = JsonValue::object(); choice.set("index", JsonValue::integer(0));
        JsonValue msg = JsonValue::object(); msg.set("role", JsonValue::string("assistant")); msg.set("content", JsonValue::string(full_text));
        choice.set("message", msg); choice.set("finish_reason", JsonValue::string("stop"));

        JsonValue usage = JsonValue::object();
        int64_t pt = (int64_t)tokens.size(), ct = (int64_t)(gen.size() - tokens.size());
        usage.set("prompt_tokens", JsonValue::integer(pt)); usage.set("completion_tokens", JsonValue::integer(ct)); usage.set("total_tokens", JsonValue::integer(pt + ct));

        JsonValue resp = JsonValue::object();
        resp.set("id", JsonValue::string(id));
        resp.set("object", JsonValue::string("chat.completion"));
        resp.set("created", JsonValue::integer(time(nullptr)));
        resp.set("model", JsonValue::string(model_id));
        JsonValue choices = JsonValue::array(); choices.push(choice);
        resp.set("choices", choices); resp.set("usage", usage);

        return http_resp(200, json_render(resp), "application/json");
    }
}

#else
// ----- Linux PagedAttention backend -----
struct PendingRequest {
    int32_t server_id = 0;
    int32_t engine_request_id = -1;
    int32_t fd = -1;
    bool stream = false;
    bool is_chat = false;
    int32_t max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.8f;
    std::string prompt_text;
    std::vector<int32_t> prompt_tokens;
    std::string completion_id;
    std::string model_id;
    bool sent_header = false;
};

static std::string handle_completion_paged(
    PagedEngine* engine, const TokenizerConfig& tokenizer,
    const HttpRequest& req, const std::string& model_id)
{
    bool stream = false;
    size_t sp = req.body.find("\"stream\"");
    if (sp != std::string::npos) {
        sp = req.body.find(':', sp + 8);
        if (sp != std::string::npos) {
            sp++; while (sp < req.body.size() && (req.body[sp] == ' ' || req.body[sp] == '\t')) sp++;
            if (sp < req.body.size() && req.body.compare(sp, 4, "true") == 0) stream = true;
        }
    }

    float temp = json_get_number(req.body, "temperature", 0.7f);
    float top_p = json_get_number(req.body, "top_p", 0.8f);
    int32_t max_tok = json_get_int(req.body, "max_tokens", 512);

    std::vector<int32_t> tokens;
    if (req.path.find("/v1/chat/completions") != std::string::npos)
        tokens = build_chat_prompt_tokens(tokenizer, req.body);
    else
        tokens = build_text_prompt_tokens(tokenizer, req.body);

    if (tokens.empty())
        return http_resp(400, R"({"error":{"message":"No valid prompt","type":"invalid_request_error"}})", "application/json");

    int32_t rid = engine->submit_tokens(tokens, max_tok, temp, top_p);

    std::string id = req.path.find("chat") != std::string::npos
        ? "chatcmpl-" + std::to_string(time(nullptr))
        : "cmpl-" + std::to_string(time(nullptr));

    if (stream) {
        std::string response;
        response += sse("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"model\":\"" + model_id + "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}");
        bool done = false;
        while (!done) {
            auto output = engine->step();
            for (size_t i = 0; i < output.request_ids.size(); i++) {
                if (output.request_ids[i] == rid) {
                    if (output.finished[i]) done = true;
                    int32_t tid = output.new_token_ids[i];
                    if (tid != EOS_TOKEN_ID) {
                        std::string decoded = decode_token(tokenizer, tid);
                        response += sse("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"model\":\"" + model_id + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + decoded + "\"}}]}");
                    }
                }
            }
        }
        response += sse("[DONE]");
        return http_resp(200, response, "text/event-stream");
    } else {
        std::string full_text;
        bool done = false;
        while (!done) {
            auto output = engine->step();
            for (size_t i = 0; i < output.request_ids.size(); i++) {
                if (output.request_ids[i] == rid) {
                    int32_t tid = output.new_token_ids[i];
                    if (tid == EOS_TOKEN_ID) {
                        done = true;
                    } else {
                        full_text += decode_token(tokenizer, tid);
                    }
                    if (output.finished[i]) done = true;
                }
            }
        }

        JsonValue choice = JsonValue::object(); choice.set("index", JsonValue::integer(0));
        JsonValue msg = JsonValue::object(); msg.set("role", JsonValue::string("assistant")); msg.set("content", JsonValue::string(full_text));
        choice.set("message", msg); choice.set("finish_reason", JsonValue::string("stop"));

        JsonValue usage = JsonValue::object();
        usage.set("prompt_tokens", JsonValue::integer((int64_t)tokens.size()));
        usage.set("completion_tokens", JsonValue::integer((int64_t)full_text.size()));
        usage.set("total_tokens", JsonValue::integer((int64_t)(tokens.size() + full_text.size())));

        JsonValue resp = JsonValue::object();
        resp.set("id", JsonValue::string(id));
        resp.set("object", JsonValue::string("chat.completion"));
        resp.set("created", JsonValue::integer(time(nullptr)));
        resp.set("model", JsonValue::string(model_id));
        JsonValue choices = JsonValue::array(); choices.push(choice);
        resp.set("choices", choices); resp.set("usage", usage);

        return http_resp(200, json_render(resp), "application/json");
    }
}
#endif

// ============================================================
// Client handler (per-connection thread)
// ============================================================

#ifdef HAS_MLX
static void client_thread(int32_t fd, MlxModel* model, const std::string& model_id) {
    char buf[65536]; int32_t total = 0;
    while (total < 65535) {
        int32_t n = recv(fd, buf + total, 65535 - total, 0);
        if (n <= 0) break; total += n; buf[total] = '\0';
        std::string r(buf, total);
        auto crlf = r.find("\r\n\r\n");
        if (crlf == std::string::npos) continue;
        int32_t body_start = (int32_t)crlf + 4;
        std::string hdr = r.substr(0, crlf);
        auto clp = hdr.find("Content-Length:");
        if (clp != std::string::npos) {
            std::string cl = hdr.substr(clp + 15); auto nls = cl.find('\r'); if (nls != std::string::npos) cl = cl.substr(0, nls);
            while (!cl.empty() && cl[0] == ' ') cl = cl.substr(1);
            if ((int32_t)total >= body_start + std::stoi(cl)) break;
            continue;
        }
        break;
    }
    if (total <= 0) { close(fd); return; }

    std::string raw(buf, total);
    HttpRequest req = parse_http(raw);
    std::string response;

    if (req.path.find("/v1/chat/completions") == 0 || req.path.find("/v1/completions") == 0)
        response = handle_completion_mlx(model, req, model_id);
    else if (req.path.find("/v1/models") == 0)
        response = handle_models(model_id);
    else if (req.path.find("/health") == 0)
        response = handle_health();
    else
        response = http_resp(200, R"({"message":"vllm-cpp server","endpoints":["/v1/chat/completions","/v1/completions","/v1/models","/health"]})", "application/json");

    if (!response.empty()) send(fd, response.c_str(), response.size(), 0);
    close(fd);
}
#else
static void client_thread(int32_t fd, PagedEngine* engine, const TokenizerConfig& tokenizer,
                          const std::string& model_id) {
    char buf[65536]; int32_t total = 0;
    while (total < 65535) {
        int32_t n = recv(fd, buf + total, 65535 - total, 0);
        if (n <= 0) break; total += n; buf[total] = '\0';
        std::string r(buf, total);
        auto crlf = r.find("\r\n\r\n");
        if (crlf == std::string::npos) continue;
        int32_t body_start = (int32_t)crlf + 4;
        std::string hdr = r.substr(0, crlf);
        auto clp = hdr.find("Content-Length:");
        if (clp != std::string::npos) {
            std::string cl = hdr.substr(clp + 15); auto nls = cl.find('\r'); if (nls != std::string::npos) cl = cl.substr(0, nls);
            while (!cl.empty() && cl[0] == ' ') cl = cl.substr(1);
            if ((int32_t)total >= body_start + std::stoi(cl)) break;
            continue;
        }
        break;
    }
    if (total <= 0) { close(fd); return; }

    std::string raw(buf, total);
    HttpRequest req = parse_http(raw);
    std::string response;

    if (req.path.find("/v1/chat/completions") == 0 || req.path.find("/v1/completions") == 0) {
        // Build prompt tokens based on request type
        std::vector<int32_t> tokens;
        if (req.path.find("/v1/chat/completions") != std::string::npos)
            tokens = build_chat_prompt_tokens(tokenizer, req.body);
        else
            tokens = build_text_prompt_tokens(tokenizer, req.body);

        float temp = json_get_number(req.body, "temperature", 0.7f);
        float top_p = json_get_number(req.body, "top_p", 0.8f);
        int32_t max_tok = json_get_int(req.body, "max_tokens", 512);

        int32_t rid = engine->submit_tokens(tokens, max_tok, temp, top_p);

        // Simple non-streaming generation loop
        std::string full_text;
        bool done = false;
        while (!done) {
            auto output = engine->step();
            for (size_t i = 0; i < output.request_ids.size(); i++) {
                if (output.request_ids[i] == rid) {
                    int32_t tid = output.new_token_ids[i];
                    if (tid == EOS_TOKEN_ID) {
                        done = true;
                    } else {
                        full_text += decode_token(tokenizer, tid);
                    }
                    if (output.finished[i]) done = true;
                }
            }
        }

        std::string id = "cmpl-" + std::to_string(time(nullptr));
        JsonValue choice = JsonValue::object(); choice.set("index", JsonValue::integer(0));
        JsonValue msg = JsonValue::object(); msg.set("role", JsonValue::string("assistant")); msg.set("content", JsonValue::string(full_text));
        choice.set("message", msg); choice.set("finish_reason", JsonValue::string("stop"));

        JsonValue resp = JsonValue::object();
        resp.set("id", JsonValue::string(id));
        resp.set("object", JsonValue::string("chat.completion"));
        resp.set("created", JsonValue::integer(time(nullptr)));
        resp.set("model", JsonValue::string(model_id));
        JsonValue choices = JsonValue::array(); choices.push(choice);
        resp.set("choices", choices);

        response = http_resp(200, json_render(resp), "application/json");
    } else if (req.path.find("/v1/models") == 0) {
        response = handle_models(model_id);
    } else if (req.path.find("/health") == 0) {
        response = handle_health();
    } else {
        response = http_resp(200, R"({"message":"vllm-cpp server","endpoints":["/v1/chat/completions","/v1/completions","/v1/models","/health"]})", "application/json");
    }

    if (!response.empty()) send(fd, response.c_str(), response.size(), 0);
    close(fd);
}
#endif

// ============================================================
// Main — platform-aware initialization
// ============================================================

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_path> [options]\n"
              << "Options:\n"
              << "  --port PORT                  HTTP port (default: 8080)\n"
              << "  --gpu-memory-utilization F   GPU memory fraction (default: 0.9)\n"
              << "  --max-model-len N            Max sequence length (default: auto)\n"
              << "  --kv-cache-dtype TYPE        KV cache dtype: fp32, fp16, fp8 (default: fp32)\n"
              << "  --debug                      Enable debug output\n"
              << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    int32_t port = 8080;
    float gpu_mem_util = 0.9f;
    int32_t max_model_len = 0;
    bool debug = false;
    std::string kv_dtype_str = "fp32";

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (arg == "--gpu-memory-utilization" && i + 1 < argc) gpu_mem_util = std::stof(argv[++i]);
        else if (arg == "--max-model-len" && i + 1 < argc) max_model_len = std::stoi(argv[++i]);
        else if (arg == "--kv-cache-dtype" && i + 1 < argc) kv_dtype_str = argv[++i];
        else if (arg == "--debug") debug = true;
    }

    DeviceInfo::print_info();

#ifdef HAS_MLX
    // ============================================================
    // macOS MLX path
    // ============================================================
    log_info("platform: macOS (MLX Metal backend)");

    log_info("loading model (" + model_path + ")...");
    auto t0 = std::chrono::steady_clock::now();
    MlxModel* mlx_model = mlx_load_model(model_path);
    auto t1 = std::chrono::steady_clock::now();
    int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    log_info("loaded in " + std::to_string(ms) + "ms  layers=" + std::to_string(mlx_model->num_layers) +
             " hidden=" + std::to_string(mlx_model->hidden_size) + " vocab=" + std::to_string(mlx_model->vocab_size));

    size_t slash = model_path.rfind('/');
    std::string model_id = (slash != std::string::npos) ? model_path.substr(slash + 1) : model_path;

    int32_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int32_t opt = 1; setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) { std::cerr << "[ERROR] bind failed on " << port << std::endl; return 1; }
    if (listen(server_fd, 10) < 0) { std::cerr << "[ERROR] listen failed\n"; return 1; }

    log_info("listening on ::" + std::to_string(port));
    log_info("endpoints: /v1/chat/completions /v1/completions /v1/models /health");

    while (true) {
        sockaddr_in ca{}; socklen_t cl = sizeof(ca);
        int32_t fd = accept(server_fd, (sockaddr*)&ca, &cl);
        if (fd < 0) continue;
        std::thread(client_thread, fd, mlx_model, model_id).detach();
    }

    delete mlx_model; close(server_fd);

#else
    // ============================================================
    // Linux CPU / CUDA PagedAttention path
    // ============================================================
    log_info("platform: Linux (PagedAttention engine)");

    log_info("loading model (" + model_path + ")...");
    auto t0 = std::chrono::steady_clock::now();
    Qwen36Model model = load_model_safetensors(model_path);

    // Apply KV cache dtype
    if (kv_dtype_str == "fp8" || kv_dtype_str == "fp8_e4m3") {
        model.quant_config.kv_dtype = KVDtype::FP8_E4M3;
        log_info("KV cache dtype: FP8_E4M3 (4x memory reduction)");
    } else if (kv_dtype_str == "fp16") {
        model.quant_config.kv_dtype = KVDtype::FP16;
        log_info("KV cache dtype: FP16 (2x memory reduction)");
    } else {
        model.quant_config.kv_dtype = KVDtype::FP32;
        log_info("KV cache dtype: FP32");
    }

    auto t1 = std::chrono::steady_clock::now();
    int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    log_info("loaded in " + std::to_string(ms) + "ms  layers=" + std::to_string(model.num_layers) +
             " hidden=" + std::to_string(model.hidden_size) + " vocab=" + std::to_string(model.vocab_size));

    size_t slash = model_path.rfind('/');
    std::string model_id = (slash != std::string::npos) ? model_path.substr(slash + 1) : model_path;

    // Create PagedAttention engine
    int32_t max_seqs = 8;
    int32_t max_total_tokens = MAX_CONTEXT;
    int32_t num_gpu_blocks = (max_total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * max_seqs;
    PagedEngine engine(model, max_seqs, max_total_tokens, num_gpu_blocks, 0);

    int32_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int32_t opt = 1; setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) { std::cerr << "[ERROR] bind failed on " << port << std::endl; return 1; }
    if (listen(server_fd, 10) < 0) { std::cerr << "[ERROR] listen failed\n"; return 1; }

    log_info("listening on ::" + std::to_string(port));
    log_info("endpoints: /v1/chat/completions /v1/completions /v1/models /health");

    while (true) {
        sockaddr_in ca{}; socklen_t cl = sizeof(ca);
        int32_t fd = accept(server_fd, (sockaddr*)&ca, &cl);
        if (fd < 0) continue;
        std::thread(client_thread, fd, &engine, model.tokenizer, model_id).detach();
    }

    close(server_fd);
#endif

    return 0;
}
