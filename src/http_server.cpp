#include "mlx_backend.h"
#include "config.h"
#include "device.h"
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <thread>
#include <atomic>
#include <ctime>
#include <vector>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

// OpenAI-compatible v1/chat/completions + v1/completions + v1/models
// Uses MLX GPU backend (Metal) when available, CPU fallback otherwise.

static void log_info(const std::string& msg) { std::cerr << "[INFO] " << msg << std::endl; }

// --- JSON helpers ---
struct JsonValue {
    enum Type { Null, Bool, Int, Float, Str, Arr, Obj };
    Type type = Null;
    bool bval = false; int64_t ival = 0; double fval = 0.0;
    std::string sval; std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    static JsonValue null() { JsonValue v; v.type = Null; return v; }
    static JsonValue boolean(bool v) { JsonValue j; j.type = Bool; j.bval = v; return j; }
    static JsonValue integer(int64_t v) { JsonValue j; j.type = Int; j.ival = v; return j; }
    static JsonValue number(double v) { JsonValue j; j.type = Float; j.fval = v; return j; }
    static JsonValue string(const std::string& v) { JsonValue j; j.type = Str; j.sval = v; return j; }
    static JsonValue array() { JsonValue j; j.type = Arr; return j; }
    static JsonValue object() { JsonValue j; j.type = Obj; return j; }

    void push(const JsonValue& v) { arr.push_back(v); }
    void set(const std::string& key, const JsonValue& val) { obj.emplace_back(key, val); }
};

static std::string json_render(const JsonValue& v) {
    std::ostringstream o;
    switch (v.type) {
    case JsonValue::Null: o << "null"; break;
    case JsonValue::Bool: o << (v.bval ? "true" : "false"); break;
    case JsonValue::Int: o << v.ival; break;
    case JsonValue::Float: o << v.fval; break;
    case JsonValue::Str:
        o << '"';
        for (char c : v.sval) {
            if (c == '"') o << "\\\""; else if (c == '\\') o << "\\\\";
            else if (c == '\n') o << "\\n"; else if (c == '\r') o << "\\r";
            else if (c == '\t') o << "\\t"; else o << c;
        }
        o << '"';
        break;
    case JsonValue::Arr:
        o << "["; for (size_t i = 0; i < v.arr.size(); i++) { if (i) o << ","; o << json_render(v.arr[i]); } o << "]"; break;
    case JsonValue::Obj:
        o << "{"; for (size_t i = 0; i < v.obj.size(); i++) { if (i) o << ","; o << '"' << v.obj[i].first << "\":" << json_render(v.obj[i].second); } o << "}"; break;
    }
    return o.str();
}

static std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + search.size() + 2);
    if (pos == std::string::npos) return "";
    std::string result; pos++;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) { pos++; switch (json[pos]) { case '"': result += '"'; break; case '\\': result += '\\'; break; case 'n': result += '\n'; break; case 't': result += '\t'; break; case 'r': result += '\r'; break; default: result += json[pos]; break; } }
        else result += json[pos];
        pos++;
    }
    return result;
}

static float json_get_number(const std::string& json, const std::string& key, float def) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    pos++; while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.size()) return def;
    std::string num;
    while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '.' || json[pos] == '-')) num += json[pos++];
    return num.empty() ? def : std::stof(num);
}

static int32_t json_get_int(const std::string& json, const std::string& key, int32_t def) {
    return (int32_t)json_get_number(json, key, (float)def);
}

// Extract chat messages → token IDs using Qwen2.5 template
static std::vector<int32_t> build_chat_prompt(MlxModel* model, const std::string& json) {
    // Walk messages array, collect roles + content
    struct Msg { std::string role, content; };
    std::vector<Msg> messages;

    size_t pos = json.find("\"messages\"");
    if (pos == std::string::npos) return {};
    pos = json.find('[', pos);
    if (pos == std::string::npos) return {};

    while (pos < json.size()) {
        pos = json.find('{', pos);
        if (pos == std::string::npos || json[pos] == ']') break;
        size_t end_pos = json.find('}', pos);
        if (end_pos == std::string::npos) break;
        std::string piece = json.substr(pos, end_pos - pos + 1);

        std::string role = json_get_string(piece, "role");
        std::string content = json_get_string(piece, "content");
        if (!role.empty() && !content.empty())
            messages.push_back({role, content});

        pos = end_pos + 1;
    }

    if (messages.empty()) return {};

    // Build token ids with Qwen2.5 template
    std::vector<int32_t> tokens;

    for (auto& m : messages) {
        tokens.push_back(IM_START_TOKEN_ID);
        std::string role_str;
        if (m.role == "system") role_str = "system\n";
        else if (m.role == "user") role_str = "user\n";
        else if (m.role == "assistant") role_str = "assistant\n";
        else role_str = m.role + "\n";

        auto role_toks = tokenize(model->tokenizer, role_str + m.content);
        tokens.insert(tokens.end(), role_toks.begin(), role_toks.end());
        tokens.push_back(IM_END_TOKEN_ID);
        auto nl = tokenize(model->tokenizer, "\n");
        tokens.insert(tokens.end(), nl.begin(), nl.end());
    }

    // Assistant prefix
    tokens.push_back(IM_START_TOKEN_ID);
    auto asst = tokenize(model->tokenizer, "assistant\n");
    tokens.insert(tokens.end(), asst.begin(), asst.end());

    return tokens;
}

// Extract prompt text for /v1/completions
static std::vector<int32_t> build_text_prompt(MlxModel* model, const std::string& json) {
    std::string prompt = json_get_string(json, "prompt");
    if (prompt.empty()) {
        // Try system + user
        std::string sys = json_get_string(json, "system");
        std::string usr = json_get_string(json, "user");
        if (!sys.empty() && !usr.empty())
            prompt = sys + "\n" + usr;
        else if (!usr.empty())
            prompt = usr;
    }
    if (prompt.empty()) return {};
    return tokenize(model->tokenizer, prompt);
}

// HTTP
struct HttpRequest { std::string method, path, body; std::map<std::string, std::string> headers; };

static HttpRequest parse_http(const std::string& raw) {
    HttpRequest req;
    std::istringstream iss(raw);
    std::string first; std::getline(iss, first);
    auto sp = first.find(' '); if (sp != std::string::npos) req.method = first.substr(0, sp);
    auto sp2 = first.find(' ', sp + 1); if (sp2 != std::string::npos) req.path = first.substr(sp + 1, sp2 - sp - 1);

    std::string line;
    while (std::getline(iss, line)) {
        if (line == "\r" || line.empty()) break;
        auto hs = line.find(':'); if (hs != std::string::npos) {
            std::string k = line.substr(0, hs), v = line.substr(hs + 1);
            if (!v.empty() && v[0] == ' ') v = v.substr(1);
            if (!v.empty() && v.back() == '\r') v.pop_back();
            req.headers[k] = v;
        }
    }
    auto it = req.headers.find("Content-Length"); if (it == req.headers.end()) it = req.headers.find("content-length");
    if (it != req.headers.end()) { int32_t len = std::stoi(it->second); req.body.resize(len); iss.read(&req.body[0], len); }
    return req;
}

static std::string http_resp(int32_t code, const std::string& body, const std::string& ct) {
    std::ostringstream o;
    o << "HTTP/1.1 " << code << " " << (code == 200 ? "OK" : code == 400 ? "Bad Request" : "Not Found") << "\r\n";
    o << "Content-Type: " << ct << "\r\nContent-Length: " << body.size() << "\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n" << body;
    return o.str();
}

static std::string sse(const std::string& json) { return "data: " + json + "\n\n"; }

// --- Handlers ---
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
    return http_resp(200, json_render(r), "application/json");
}

// --- Main handler ---
static std::string handle_openai_mlx(MlxModel* model, const HttpRequest& req, const std::string& model_id) {
    // Check stream flag (handle "stream":true and "stream": true)
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
        tokens = build_chat_prompt(model, req.body);
    else
        tokens = build_text_prompt(model, req.body);

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

        // Final chunk
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

static void client_thread(int32_t fd, MlxModel* model, const std::string& model_id) {
    char buf[65536]; int32_t total = 0;
    while (total < 65535) {
        int32_t n = recv(fd, buf + total, 65535 - total, 0);
        if (n <= 0) break; total += n; buf[total] = '\0';
        std::string r(buf, total);
        auto crlf = r.find("\r\n\r\n");
        if (crlf == std::string::npos) continue;
        // Check content-length
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
        response = handle_openai_mlx(model, req, model_id);
    else if (req.path.find("/v1/models") == 0)
        response = handle_models(model_id);
    else if (req.path.find("/health") == 0)
        response = handle_health();
    else
        response = http_resp(200, R"({"message":"vllm-cpp MLX server","endpoints":["/v1/chat/completions","/v1/completions","/v1/models","/health"]})", "application/json");

    if (!response.empty()) send(fd, response.c_str(), response.size(), 0);
    close(fd);
}

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <model_path> [--port PORT] [--debug]\n"; return 1; }

    std::string model_path = argv[1]; int32_t port = 8080; bool debug = false;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (arg == "--debug") debug = true;
    }

    DeviceInfo::print_info();

    log_info("loading model (" + model_path + ")...");
    auto t0 = std::chrono::steady_clock::now();
    MlxModel* model = mlx_load_model(model_path);
    auto t1 = std::chrono::steady_clock::now();
    int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    log_info("loaded in " + std::to_string(ms) + "ms  layers=" + std::to_string(model->num_layers) +
             " hidden=" + std::to_string(model->hidden_size) + " vocab=" + std::to_string(model->vocab_size));

    // Derive model ID from path
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
        std::thread(client_thread, fd, model, model_id).detach();
    }

    delete model; close(server_fd);
    return 0;
}
