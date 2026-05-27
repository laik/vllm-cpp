#include "model.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <thread>
#include <atomic>
#include <ctime>
#include <vector>
#include <chrono>

// ============================================================
// Minimal HTTP server using raw sockets (no dependencies)
// OpenAI-compatible /v1/chat/completions endpoint
// ============================================================

// Minimal JSON value for building responses
struct JsonValue {
    enum Type { Null, Bool, Int, Float, Str, Arr, Obj };
    Type type = Null;
    bool bval = false;
    int64_t ival = 0;
    double fval = 0.0;
    std::string sval;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    static JsonValue null_val() { JsonValue v; v.type = Null; return v; }
    static JsonValue bool_val(bool v) { JsonValue j; j.type = Bool; j.bval = v; return j; }
    static JsonValue int_val(int64_t v) { JsonValue j; j.type = Int; j.ival = v; return j; }
    static JsonValue float_val(double v) { JsonValue j; j.type = Float; j.fval = v; return j; }
    static JsonValue str_val(const std::string& v) { JsonValue j; j.type = Str; j.sval = v; return j; }
    static JsonValue arr_val() { JsonValue j; j.type = Arr; return j; }
    static JsonValue obj_val() { JsonValue j; j.type = Obj; return j; }

    void push(const JsonValue& v) { arr.push_back(v); }
    void set(const std::string& key, const JsonValue& val) { obj.emplace_back(key, val); }
};

static std::string json_encode(const JsonValue& v) {
    std::ostringstream o;
    switch (v.type) {
        case JsonValue::Null: o << "null"; break;
        case JsonValue::Bool: o << (v.bval ? "true" : "false"); break;
        case JsonValue::Int: o << v.ival; break;
        case JsonValue::Float: o << v.fval; break;
        case JsonValue::Str:
            o << '"';
            for (char c : v.sval) {
                if (c == '"') o << "\\\"";
                else if (c == '\\') o << "\\\\";
                else if (c == '\n') o << "\\n";
                else if (c == '\r') o << "\\r";
                else if (c == '\t') o << "\\t";
                else if (c >= 0x20 && c < 0x7F) o << c;
                else o << "\\u" << std::hex << (int)(unsigned char)c;
            }
            o << '"';
            break;
        case JsonValue::Arr:
            o << "[";
            for (size_t i = 0; i < v.arr.size(); i++) {
                if (i) o << ",";
                o << json_encode(v.arr[i]);
            }
            o << "]";
            break;
        case JsonValue::Obj:
            o << "{";
            for (size_t i = 0; i < v.obj.size(); i++) {
                if (i) o << ",";
                o << '"' << v.obj[i].first << "\":" << json_encode(v.obj[i].second);
            }
            o << "}";
            break;
    }
    return o.str();
}

// Minimal JSON reader for incoming requests
static std::string parse_json_string_field(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + search.size() + 2); // skip ": "
    if (pos == std::string::npos) return "";
    // Read until closing quote
    std::string result;
    pos++; // skip opening "
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                case 'n': result += '\n'; break;
                case 't': result += '\t'; break;
                case 'r': result += '\r'; break;
                default: result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    return result;
}

static float parse_json_float_field(const std::string& json, const std::string& key, float def) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) pos++;
    if (pos >= json.size()) return def;
    std::string num;
    if (json[pos] == '-') num += json[pos++];
    while (pos < json.size() && (json[pos] >= '0' && json[pos] <= '9' || json[pos] == '.')) num += json[pos++];
    return num.empty() ? def : std::stof(num);
}

static int32_t parse_json_int_field(const std::string& json, const std::string& key, int32_t def) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) pos++;
    if (pos >= json.size()) return def;
    std::string num;
    if (json[pos] == '-') num += json[pos++];
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') num += json[pos++];
    return num.empty() ? def : std::stoi(num);
}

// Extract messages array from JSON: [{"role":"user","content":"..."}]
static std::string extract_prompt(const std::string& json) {
    // Look for messages array, find the last user content
    size_t pos = json.find("\"messages\"");
    if (pos == std::string::npos) return "";
    pos = json.find('[', pos);
    if (pos == std::string::npos) return "";

    std::string prompt;
    while (pos < json.size()) {
        // Find a message object
        pos = json.find('{', pos);
        if (pos == std::string::npos || json[pos] == ']') break;

        std::string role = parse_json_string_field(json.substr(pos), "role");
        std::string content = parse_json_string_field(json.substr(pos), "content");

        if (role == "user" || role == "assistant" || role == "system") {
            // Qwen3 uses special message format
            std::string prefix;
            if (role == "system") prefix = "<|im_start|>system\n";
            else if (role == "user") prefix = "<|im_start|>user\n";
            else prefix = "<|im_start|>assistant\n";

            if (!prompt.empty() && role != "user") prompt += "<|im_end|>\n";
            prompt += prefix + content;
        }

        // Move to next object or end of array
        pos = json.find('}', pos);
        if (pos == std::string::npos) break;
        pos++;
    }

    if (!prompt.empty()) prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

// HTTP request parser
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::map<std::string, std::string> headers;
};

static HttpRequest parse_request(const std::string& raw) {
    HttpRequest req;
    std::istringstream iss(raw);
    std::string first_line;
    std::getline(iss, first_line);

    // Parse "METHOD /path HTTP/1.1"
    auto sp = first_line.find(' ');
    if (sp != std::string::npos) {
        req.method = first_line.substr(0, sp);
        sp = first_line.find(' ', sp + 1);
        if (sp != std::string::npos) req.path = first_line.substr(sp + 1);
    }

    // Parse headers
    std::string line;
    while (std::getline(iss, line)) {
        if (line == "\r" || line.empty()) break;
        auto hs = line.find(':');
        if (hs != std::string::npos) {
            std::string key = line.substr(0, hs);
            std::string val = (hs + 1 < line.size()) ? line.substr(hs + 1) : "";
            // Trim leading space
            if (!val.empty() && val[0] == ' ') val = val.substr(1);
            // Trim \r
            if (!val.empty() && val.back() == '\r') val.pop_back();
            req.headers[key] = val;
        }
    }

    // Read body
    auto it_len = req.headers.find("Content-Length");
    if (it_len == req.headers.end()) it_len = req.headers.find("content-length");
    if (it_len != req.headers.end()) {
        int32_t len = std::stoi(it_len->second);
        req.body.resize(len);
        iss.read(&req.body[0], len);
    }

    return req;
}

// SSE event formatter
static std::string sse_event(const std::string& json) {
    return "data: " + json + "\n\n";
}

// Build chat completion response
static std::string build_choice_json(const std::string& content, bool finish) {
    JsonValue choice = JsonValue::obj_val();
    choice.set("index", JsonValue::int_val(0));

    JsonValue delta = JsonValue::obj_val();
    delta.set("role", JsonValue::str_val("assistant"));
    delta.set("content", JsonValue::str_val(content));
    choice.set("delta", delta);

    if (finish) {
        // Final chunk with finish_reason
        JsonValue final_delta = JsonValue::obj_val();
        final_delta.set("role", JsonValue::str_val("assistant"));
        final_delta.set("content", JsonValue::str_val(""));
        choice.set("delta", final_delta);
        choice.set("finish_reason", JsonValue::str_val("stop"));
    }

    return json_encode(choice);
}

static std::string build_response_json(
    const std::string& id, const std::string& model,
    const std::string& content, int64_t prompt_tokens, int64_t gen_tokens,
    bool stream) {

    if (stream) {
        // Return first SSE chunk
        JsonValue choice = JsonValue::obj_val();
        choice.set("index", JsonValue::int_val(0));
        JsonValue delta = JsonValue::obj_val();
        delta.set("role", JsonValue::str_val("assistant"));
        delta.set("content", JsonValue::str_val(content));
        choice.set("delta", delta);

        JsonValue resp = JsonValue::obj_val();
        resp.set("id", JsonValue::str_val(id));
        resp.set("object", JsonValue::str_val("chat.completion.chunk"));
        resp.set("created", JsonValue::int_val(time(nullptr)));
        resp.set("model", JsonValue::str_val(model));

        JsonValue choices = JsonValue::arr_val();
        choices.push(choice);
        resp.set("choices", choices);

        return sse_event(json_encode(resp));
    } else {
        // Non-streaming
        JsonValue choice = JsonValue::obj_val();
        choice.set("index", JsonValue::int_val(0));
        JsonValue message = JsonValue::obj_val();
        message.set("role", JsonValue::str_val("assistant"));
        message.set("content", JsonValue::str_val(content));
        choice.set("message", message);
        choice.set("finish_reason", JsonValue::str_val("stop"));

        JsonValue usage = JsonValue::obj_val();
        usage.set("prompt_tokens", JsonValue::int_val(prompt_tokens));
        usage.set("completion_tokens", JsonValue::int_val(gen_tokens));
        usage.set("total_tokens", JsonValue::int_val(prompt_tokens + gen_tokens));

        JsonValue resp = JsonValue::obj_val();
        resp.set("id", JsonValue::str_val(id));
        resp.set("object", JsonValue::str_val("chat.completion"));
        resp.set("created", JsonValue::int_val(time(nullptr)));
        resp.set("model", JsonValue::str_val(model));

        JsonValue choices = JsonValue::arr_val();
        choices.push(choice);
        resp.set("choices", choices);
        resp.set("usage", usage);

        return json_encode(resp);
    }
}

// HTTP response builder
static std::string build_http_response(int32_t status, const std::string& body,
                                        const std::string& content_type) {
    std::ostringstream o;
    o << "HTTP/1.1 " << status << " "
      << (status == 200 ? "OK" : status == 400 ? "Bad Request" : "Not Found")
      << "\r\n";
    o << "Content-Type: " << content_type << "\r\n";
    o << "Content-Length: " << body.size() << "\r\n";
    o << "Connection: keep-alive\r\n";
    o << "Access-Control-Allow-Origin: *\r\n";
    o << "X-Content-Type-Options: nosniff\r\n";
    o << "Cache-Control: no-cache\r\n";
    o << "\r\n";
    o << body;
    return o.str();
}

// Handle /v1/models endpoint
static std::string handle_models() {
    JsonValue data = JsonValue::obj_val();
    data.set("id", JsonValue::str_val("qwen3.6-35b-a3b"));
    data.set("object", JsonValue::str_val("model"));
    data.set("owned_by", JsonValue::str_val("vllm-cpp"));

    JsonValue arr = JsonValue::arr_val();
    arr.push(data);

    JsonValue resp = JsonValue::obj_val();
    resp.set("object", JsonValue::str_val("list"));
    resp.set("data", arr);

    return build_http_response(200, json_encode(resp), "application/json");
}

// Handle /v1/chat/completions endpoint
static void handle_chat_completion(
    const HttpRequest& req, const Qwen36Model& model,
    std::string& response_out) {

    auto prompt = extract_prompt(req.body);
    if (prompt.empty()) {
        response_out = build_http_response(400,
            R"({"error":{"message":"No user messages found","type":"invalid_request_error"}})",
            "application/json");
        return;
    }

    bool stream = req.body.find("\"stream\"") != std::string::npos &&
                  req.body.find("\"stream\": true") != std::string::npos;

    float temperature = parse_json_float_field(req.body, "temperature", TEMPERATURE);
    float top_p = parse_json_float_field(req.body, "top_p", TOP_P);
    int32_t max_tokens = parse_json_int_field(req.body, "max_tokens", MAX_NEW_TOKENS);

    // Tokenize
    auto input_tokens = tokenize(model.tokenizer, prompt);
    if (input_tokens.empty()) {
        // Fallback: use BOS token
        input_tokens.push_back(model.tokenizer.bos_token_id);
    }

    std::string id = "chatcmpl-" + std::to_string(time(nullptr));

    if (stream) {
        // Streaming response
        std::string response;

        // Generate and send chunks
        auto generated = generate_cpu(model, input_tokens, max_tokens, temperature, top_p,
            [&](int32_t token_id) {
                auto decoded_arr = decode_tokens(model.tokenizer, {token_id});
                std::string decoded;
                for (auto& d : decoded_arr) decoded += d;

                JsonValue chunk = JsonValue::obj_val();
                chunk.set("id", JsonValue::str_val(id));
                chunk.set("object", JsonValue::str_val("chat.completion.chunk"));
                chunk.set("created", JsonValue::int_val(time(nullptr)));
                chunk.set("model", JsonValue::str_val("qwen3.6-35b-a3b"));

                JsonValue delta = JsonValue::obj_val();
                delta.set("content", JsonValue::str_val(decoded));
                JsonValue choice = JsonValue::obj_val();
                choice.set("index", JsonValue::int_val(0));
                choice.set("delta", delta);

                JsonValue choices = JsonValue::arr_val();
                choices.push(choice);
                chunk.set("choices", choices);

                response += sse_event(json_encode(chunk));
            }
        );

        // Final chunk
        JsonValue final_chunk = JsonValue::obj_val();
        final_chunk.set("id", JsonValue::str_val(id));
        final_chunk.set("object", JsonValue::str_val("chat.completion.chunk"));
        final_chunk.set("created", JsonValue::int_val(time(nullptr)));
        final_chunk.set("model", JsonValue::str_val("qwen3.6-35b-a3b"));

        JsonValue final_delta = JsonValue::obj_val();
        final_delta.set("content", JsonValue::str_val(""));
        JsonValue final_choice = JsonValue::obj_val();
        final_choice.set("index", JsonValue::int_val(0));
        final_choice.set("delta", final_delta);
        final_choice.set("finish_reason", JsonValue::str_val("stop"));

        JsonValue final_choices = JsonValue::arr_val();
        final_choices.push(final_choice);
        final_chunk.set("choices", final_choices);

        response += sse_event(json_encode(final_chunk));
        response += sse_event("[DONE]");

        response_out = build_http_response(200, response, "text/event-stream");
    } else {
        // Non-streaming: generate all then return
        std::string full_text;
        auto generated = generate_cpu(model, input_tokens, max_tokens, temperature, top_p,
            [&](int32_t token_id) {
                auto decoded_arr = decode_tokens(model.tokenizer, {token_id});
                for (auto& d : decoded_arr) full_text += d;
            }
        );

        response_out = build_response_json(id, "qwen3.6-35b-a3b", full_text,
                                           (int64_t)input_tokens.size(),
                                           (int64_t)(generated.size() - input_tokens.size()),
                                           false);
    }
}

// Handle /v1/completions endpoint (legacy)
static void handle_completions(
    const HttpRequest& req, const Qwen36Model& model,
    std::string& response_out) {

    std::string prompt = parse_json_string_field(req.body, "prompt");
    if (prompt.empty()) {
        response_out = build_http_response(400,
            R"({"error":{"message":"No prompt provided","type":"invalid_request_error"}})",
            "application/json");
        return;
    }

    float temperature = parse_json_float_field(req.body, "temperature", TEMPERATURE);
    float top_p = parse_json_float_field(req.body, "top_p", TOP_P);
    int32_t max_tokens = parse_json_int_field(req.body, "max_tokens", 256);

    auto input_tokens = tokenize(model.tokenizer, prompt);
    if (input_tokens.empty()) {
        input_tokens.push_back(model.tokenizer.bos_token_id);
    }

    std::string full_text;
    auto generated = generate_cpu(model, input_tokens, max_tokens, temperature, top_p,
        [&](int32_t token_id) {
            auto decoded_arr = decode_tokens(model.tokenizer, {token_id});
            for (auto& d : decoded_arr) full_text += d;
        }
    );

    std::string id = "cmpl-" + std::to_string(time(nullptr));

    JsonValue choice = JsonValue::obj_val();
    choice.set("index", JsonValue::int_val(0));
    choice.set("text", JsonValue::str_val(full_text));
    choice.set("finish_reason", JsonValue::str_val("stop"));

    JsonValue usage = JsonValue::obj_val();
    usage.set("prompt_tokens", JsonValue::int_val(input_tokens.size()));
    usage.set("completion_tokens", JsonValue::int_val(generated.size() - input_tokens.size()));
    usage.set("total_tokens", JsonValue::int_val(generated.size()));

    JsonValue resp = JsonValue::obj_val();
    resp.set("id", JsonValue::str_val(id));
    resp.set("object", JsonValue::str_val("text_completion"));
    resp.set("created", JsonValue::int_val(time(nullptr)));
    resp.set("model", JsonValue::str_val("qwen3.6-35b-a3b"));

    JsonValue choices = JsonValue::arr_val();
    choices.push(choice);
    resp.set("choices", choices);
    resp.set("usage", usage);

    response_out = build_http_response(200, json_encode(resp), "application/json");
}

// Handle a single client connection
static void handle_client(int32_t client_fd, Qwen36Model& model) {
    const int32_t BUF_SIZE = 65536;
    char buf[BUF_SIZE];
    int32_t total_read = 0;

    // Read until we have a complete HTTP request
    while (total_read < BUF_SIZE - 1) {
        int32_t n = recv(client_fd, buf + total_read, BUF_SIZE - total_read - 1, 0);
        if (n <= 0) break;
        total_read += n;
        buf[total_read] = '\0';

        // Check if we have the full headers (look for \r\n\r\n)
        std::string received(buf, total_read);
        if (received.find("\r\n\r\n") != std::string::npos) {
            // Check if we need more for the body
            auto crlf_pos = received.find("\r\n\r\n");
            std::string header_part = received.substr(0, crlf_pos);
            auto content_len_pos = header_part.find("Content-Length:");
            if (content_len_pos != std::string::npos) {
                std::string cl_line = header_part.substr(content_len_pos);
                auto nl = cl_line.find('\n');
                if (nl != std::string::npos) cl_line = cl_line.substr(0, nl);
                cl_line.erase(0, cl_line.find(':') + 1);
                // Trim
                while (!cl_line.empty() && cl_line[0] == ' ') cl_line = cl_line.substr(1);
                int32_t cl = std::stoi(cl_line);
                int32_t body_start = (int32_t)crlf_pos + 4;
                if (total_read >= body_start + cl) break; // Have full body
                // Need to read more
                continue;
            } else {
                break; // No body expected
            }
        }
    }

    if (total_read <= 0) {
        close(client_fd);
        return;
    }

    std::string raw(buf, total_read);
    HttpRequest req = parse_request(raw);

    std::string response;

    if (req.path == "/v1/chat/completions" || req.path == "/v1/chat/completions/") {
        handle_chat_completion(req, model, response);
    } else if (req.path == "/v1/completions" || req.path == "/v1/completions/") {
        handle_completions(req, model, response);
    } else if (req.path == "/v1/models" || req.path == "/v1/models/") {
        response = handle_models();
    } else if (req.path == "/health") {
        response = build_http_response(200, R"({"status":"ok"})", "application/json");
    } else if (req.path == "/" || req.path == "/") {
        response = build_http_response(200,
            R"({"message":"vllm-cpp Qwen3.6 inference server","endpoints":["/v1/chat/completions","/v1/completions","/v1/models","/health"]})",
            "application/json");
    } else {
        response = build_http_response(404,
            R"({"error":{"message":"Not found","type":"invalid_request_error"}})",
            "application/json");
    }

    if (!response.empty()) {
        send(client_fd, response.c_str(), response.size(), 0);
    }

    close(client_fd);
}

// ============================================================
// Main: start HTTP server
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_path> [--port PORT]\n"
                  << "  <model_path>  Path to model directory\n"
                  << "  --port PORT   Listen port (default: 8080)\n"
                  << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    int32_t port = 8080;

    for (int32_t i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[i + 1]);
    }

    // Load model
    std::cout << "Loading model..." << std::endl;
    Qwen36Model model = load_model_safetensors(model_path);
    std::cout << "Model loaded: " << model.num_layers << " layers, "
              << model.hidden_size << " hidden" << std::endl;

    // Create server socket
    int32_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    // Allow address reuse
    int32_t opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Bind
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "Failed to bind on port " << port << std::endl;
        close(server_fd);
        return 1;
    }

    // Listen
    if (listen(server_fd, 10) < 0) {
        std::cerr << "Failed to listen" << std::endl;
        close(server_fd);
        return 1;
    }

    std::cout << "vllm-cpp server listening on port " << port << std::endl;
    std::cout << "Endpoints: /v1/chat/completions /v1/completions /v1/models /health" << std::endl;

    // Accept loop
    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int32_t client_fd = accept(server_fd, (sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) continue;

        std::cout << "Connection from "
                  << inet_ntoa(client_addr.sin_addr) << std::endl;

        // Handle in a thread
        std::thread t(handle_client, client_fd, std::ref(model));
        t.detach();
    }

    close(server_fd);
    return 0;
}
