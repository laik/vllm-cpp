#include "model.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <filesystem>

// ============================================================
// Minimal JSON parser for safetensors header
// ============================================================

static std::string skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t'))
        pos++;
    return s;
}

static bool parse_json_string(const std::string& s, size_t& pos, std::string& out) {
    if (pos >= s.size() || s[pos] != '"') return false;
    pos++;
    out.clear();
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            pos++;
            switch (s[pos]) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '/': out += '/'; break;
                case 'u':
                    if (pos + 4 < s.size()) {
                        int cp = 0;
                        for (int i = 0; i < 4; i++) {
                            cp <<= 4;
                            char c = s[pos + 1 + i];
                            cp += (c >= '0' && c <= '9') ? c - '0' :
                                  ((c >= 'a' && c <= 'f') ? c - 'a' + 10 :
                                  ((c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0));
                        }
                        if (cp < 0x80) out += (char)cp;
                        else if (cp < 0x800) {
                            out += (char)(0xC0 | (cp >> 6));
                            out += (char)(0x80 | (cp & 0x3F));
                        } else {
                            out += (char)(0xE0 | (cp >> 12));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F));
                            out += (char)(0x80 | (cp & 0x3F));
                        }
                        pos += 4;
                    }
                    break;
                default: out += s[pos]; break;
            }
        } else {
            out += s[pos];
        }
        pos++;
    }
    if (pos < s.size()) pos++;
    return true;
}

static bool parse_json_number(const std::string& s, size_t& pos, double& out) {
    if (pos >= s.size()) return false;
    std::string num;
    bool neg = false;
    if (s[pos] == '-') { neg = true; pos++; }
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') num += s[pos++];
    if (pos < s.size() && s[pos] == '.') {
        num += '.'; pos++;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') num += s[pos++];
    }
    if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
        num += s[pos++];
        if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) num += s[pos++];
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') num += s[pos++];
    }
    if (num.empty()) return false;
    out = std::stod(num) * (neg ? -1.0 : 1.0);
    return true;
}

static bool parse_json_array_int64(const std::string& s, size_t& pos, std::vector<int64_t>& out) {
    if (pos >= s.size() || s[pos] != '[') return false;
    pos++;
    out.clear();
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ']') { pos++; return true; }
    while (pos < s.size()) {
        skip_ws(s, pos);
        double val;
        if (!parse_json_number(s, pos, val)) return false;
        out.push_back((int64_t)val);
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') pos++;
        else if (pos < s.size() && s[pos] == ']') { pos++; return true; }
        else return false;
    }
    return true;
}

static bool parse_json_object(const std::string& s, size_t& pos,
                               std::map<std::string, std::string>& str_fields,
                               std::map<std::string, std::vector<int64_t>>& arr_fields) {
    if (pos >= s.size() || s[pos] != '{') return false;
    pos++;
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') { pos++; return true; }
    while (pos < s.size()) {
        skip_ws(s, pos);
        std::string key;
        if (!parse_json_string(s, pos, key)) return false;
        skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ':') return false;
        pos++;
        skip_ws(s, pos);
        if (pos >= s.size()) return false;
        if (s[pos] == '"') {
            std::string val;
            if (!parse_json_string(s, pos, val)) return false;
            str_fields[key] = val;
        } else if (s[pos] == '[') {
            std::vector<int64_t> val;
            if (!parse_json_array_int64(s, pos, val)) return false;
            arr_fields[key] = val;
        } else if (s[pos] == '{') {
            // Nested object: parse recursively with dotted keys
            std::map<std::string, std::string> nested_str;
            std::map<std::string, std::vector<int64_t>> nested_arr;
            parse_json_object(s, pos, nested_str, nested_arr);
            for (auto& [k, v] : nested_str) str_fields[key + "." + k] = v;
            for (auto& [k, v] : nested_arr) arr_fields[key + "." + k] = v;
        } else {
            double val;
            if (parse_json_number(s, pos, val)) {
                str_fields[key] = std::to_string((int64_t)val);
            } else if (s.substr(pos, 4) == "true") {
                str_fields[key] = "true"; pos += 4;
            } else if (s.substr(pos, 5) == "false") {
                str_fields[key] = "false"; pos += 5;
            } else if (s.substr(pos, 4) == "null") {
                str_fields[key] = "null"; pos += 4;
            } else {
                return false;
            }
        }
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') pos++;
        else if (pos < s.size() && s[pos] == '}') { pos++; return true; }
        else return false;
    }
    return true;
}

// ============================================================
// safetensors header parsing
// ============================================================

static std::optional<std::vector<TensorMeta>> parse_safetensors_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;

    uint64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    if (!f || header_len > 100 * 1024 * 1024) return std::nullopt;

    std::string header(header_len, '\0');
    f.read(&header[0], header_len);
    if (!f) return std::nullopt;

    std::map<std::string, std::string> str_fields;
    std::map<std::string, std::vector<int64_t>> arr_fields;

    size_t pos = 0;
    skip_ws(header, pos);
    if (!parse_json_object(header, pos, str_fields, arr_fields)) return std::nullopt;

    // Each tensor: key is "tensor_name" with value = dtype string
    // Then "tensor_name.shape", "tensor_name.data_offsets" as arrays
    std::vector<TensorMeta> tensors;

    // First, find all tensor keys (those that have .shape entries)
    std::vector<std::string> tensor_names;
    for (auto& [key, val] : arr_fields) {
        if (key == "shape" || key == "data_offsets") continue;
        auto dot_pos = key.rfind('.');
        if (dot_pos != std::string::npos) {
            std::string base = key.substr(0, dot_pos);
            std::string suffix = key.substr(dot_pos + 1);
            if ((suffix == "shape" || suffix == "data_offsets") &&
                std::find(tensor_names.begin(), tensor_names.end(), base) == tensor_names.end()) {
                tensor_names.push_back(base);
            }
        }
    }

    for (auto& name : tensor_names) {
        TensorMeta meta;
        meta.name = name;
        meta.dtype = str_fields[name + ".dtype"];
        meta.shape = arr_fields[name + ".shape"];
        auto dof = arr_fields[name + ".data_offsets"];
        meta.data_offsets[0] = (dof.size() > 0) ? dof[0] : 0;
        meta.data_offsets[1] = (dof.size() > 1) ? dof[1] : 0;
        tensors.push_back(meta);
    }

    return tensors;
}

// ============================================================
// Tensor loading
// ============================================================

static bool load_tensor_from_file(const std::string& path, const TensorMeta& meta, Tensor& out) {
    out.shape = meta.shape;
    out.precision = (meta.dtype == "F16") ? Precision::F16 :
                    (meta.dtype == "BF16") ? Precision::BF16 : Precision::F32;

    int64_t numel = 1;
    for (auto d : meta.shape) numel *= d;
    out.numel = numel;
    out.byte_size = numel * sizeof(float);
    out.data = new float[numel]();

    std::ifstream f(path, std::ios::binary);
    if (!f) { delete[] out.data; out.data = nullptr; return false; }

    f.seekg(meta.data_offsets[0]);
    if (!f) { delete[] out.data; out.data = nullptr; return false; }

    if (meta.dtype == "F32") {
        f.read(reinterpret_cast<char*>(out.data), numel * sizeof(float));
    } else if (meta.dtype == "F16") {
        std::vector<uint16_t> raw(numel);
        f.read(reinterpret_cast<char*>(raw.data()), numel * sizeof(uint16_t));
        for (int64_t i = 0; i < numel; i++) {
            uint16_t h = raw[i];
            uint32_t sign = (h >> 15) & 0x1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t frac = h & 0x3FF;
            if (exp == 0) {
                if (frac == 0) { out.data[i] = sign ? -0.0f : 0.0f; }
                else {
                    frac <<= 1;
                    while ((frac & 0x400) == 0) { frac <<= 1; exp--; }
                    frac &= 0x3FF;
                    uint32_t e = (exp > 0) ? (uint32_t)exp - 9 : 0;
                    uint32_t bits = (sign << 31) | ((127 + e) << 23) | (frac << 13);
                    out.data[i] = *(float*)&bits;
                }
            } else if (exp == 31) {
                out.data[i] = sign ? -INFINITY : INFINITY;
            } else {
                uint32_t bits = (sign << 31) | ((127 - 15 + exp) << 23) | (frac << 13);
                out.data[i] = *(float*)&bits;
            }
        }
    } else if (meta.dtype == "BF16") {
        std::vector<uint16_t> raw(numel);
        f.read(reinterpret_cast<char*>(raw.data()), numel * sizeof(uint16_t));
        for (int64_t i = 0; i < numel; i++) {
            uint32_t bits = (uint32_t)raw[i] << 16;
            out.data[i] = *(float*)&bits;
        }
    } else {
        std::cerr << "Unsupported dtype: " << meta.dtype << std::endl;
        delete[] out.data; out.data = nullptr;
        return false;
    }

    return f.good();
}

// ============================================================
// Load all tensors from safetensors files
// ============================================================

static std::map<std::string, Tensor> load_all_tensors(const std::string& model_path) {
    std::map<std::string, Tensor> tensors;
    std::vector<std::string> st_files;

    for (auto& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.path().extension() == ".safetensors") {
            st_files.push_back(entry.path().string());
        }
    }

    if (st_files.empty()) {
        std::cerr << "No safetensors files found in " << model_path << std::endl;
        return tensors;
    }

    std::sort(st_files.begin(), st_files.end());

    for (auto& path : st_files) {
        std::cout << "  Loading " << path.substr(path.find_last_of("/") + 1) << "..." << std::endl;
        auto metas = parse_safetensors_header(path);
        if (!metas) { std::cerr << "  Failed to parse header" << std::endl; continue; }

        for (auto& meta : *metas) {
            if (load_tensor_from_file(path, meta, tensors[meta.name])) {
                // success
            } else {
                std::cerr << "  Failed to load tensor: " << meta.name << std::endl;
                tensors.erase(meta.name);
            }
        }
    }

    return tensors;
}

// ============================================================
// Model loading from safetensors
// ============================================================

Qwen36Model load_model_safetensors(const std::string& model_path) {
    std::cout << "Loading model from: " << model_path << std::endl;
    auto tensors = load_all_tensors(model_path);
    Qwen36Model model;

    auto find_tensor = [&](const std::vector<std::string>& names) -> Tensor* {
        for (auto& name : names) {
            if (tensors.count(name)) return &tensors[name];
            for (auto& [k, v] : tensors) {
                if (k == name || k.find(name) != std::string::npos) return &tensors[k];
            }
        }
        return nullptr;
    };

    // --- Embeddings ---
    auto* embed = find_tensor({"embed_tokens.weight", "model.embed_tokens.weight"});
    if (embed) {
        model.vocab_size = (int32_t)embed->shape[0];
        model.hidden_size = (int32_t)embed->shape[1];
        model.embeddings = *embed;
        std::cout << "  Embeddings: " << model.vocab_size << " x " << model.hidden_size << std::endl;
    }

    // --- Detect heads from q_proj ---
    auto* q_proj = find_tensor({".q_proj"});
    if (q_proj) {
        model.hidden_size = (int32_t)q_proj->shape[1];
        // Try to get the actual q_proj for layer 0
        auto* q0 = find_tensor({"model.layers.0.self_attn.q_proj.weight"});
        if (q0) {
            model.num_heads = (int32_t)(q0->shape[0] / (q0->shape[0] / q0->shape[1]));
            model.head_dim = model.hidden_size / model.num_heads;
        } else {
            model.num_heads = NUM_HEADS;
            model.head_dim = (int32_t)(q_proj->shape[0] / NUM_HEADS);
        }
    }

    auto* k0 = find_tensor({"model.layers.0.self_attn.k_proj.weight"});
    if (k0) {
        model.num_kv_heads = (int32_t)(k0->shape[0] / model.head_dim);
    }

    // --- Detect layer count ---
    int32_t max_layer = -1;
    for (auto& [k, v] : tensors) {
        auto pos = k.find("model.layers.");
        if (pos != std::string::npos) {
            auto end = k.find('.', pos + 13);
            if (end != std::string::npos) {
                int32_t layer = std::stoi(k.substr(pos + 13, end - pos - 13));
                max_layer = std::max(max_layer, layer);
            }
        }
    }
    model.num_layers = max_layer + 1;
    std::cout << "  Layers: " << model.num_layers << std::endl;

    model.layers.resize(model.num_layers);

    // --- Load each layer ---
    for (int32_t l = 0; l < model.num_layers; l++) {
        auto prefix = "model.layers." + std::to_string(l) + ".";

        auto gt = [&](const std::string& suffix) -> Tensor* {
            auto key = prefix + suffix;
            if (tensors.count(key)) return &tensors[key];
            return nullptr;
        };

        // Input norm
        auto* inp_norm = gt("input_layernorm.weight");
        if (inp_norm) { model.layers[l].input_norm.weight = *inp_norm; }

        // Attention
        auto* qp = gt("self_attn.q_proj.weight");
        if (qp) model.layers[l].attn.q_proj = *qp;
        auto* kp = gt("self_attn.k_proj.weight");
        if (kp) model.layers[l].attn.k_proj = *kp;
        auto* vp = gt("self_attn.v_proj.weight");
        if (vp) model.layers[l].attn.v_proj = *vp;
        auto* op = gt("self_attn.o_proj.weight");
        if (op) model.layers[l].attn.o_proj = *op;

        // Post-attention norm
        auto* atn_norm = gt("post_attention_layernorm.weight");
        if (atn_norm) { model.layers[l].attn_norm.weight = *atn_norm; }

        // MLP (dense)
        auto* gate_w = gt("mlp.gate_proj.weight");
        auto* up_w = gt("mlp.up_proj.weight");
        auto* down_w = gt("mlp.down_proj.weight");
        if (gate_w) {
            model.layers[l].mlp.gate_proj = *gate_w;
            model.layers[l].mlp.up_proj = *up_w;
            model.layers[l].mlp.down_proj = *down_w;
            model.ffn_intermediate = (int32_t)gate_w->shape[0];
        }

        // MoE
        auto* moe_router = gt("mlp.gate.weight");
        if (moe_router) {
            model.layers[l].moe.gate = *moe_router;
            model.is_moe = true;

            // Count experts
            int32_t num_experts = 0;
            for (auto& [k, v] : tensors) {
                if (k.find(prefix + "mlp.experts.") != std::string::npos) {
                    auto es = k.find(".experts.") + 9;
                    auto ee = k.find('.', es);
                    if (es <= (size_t)ee) {
                        int32_t e = std::stoi(k.substr(es, ee - es));
                        num_experts = std::max(num_experts, e + 1);
                    }
                }
            }
            model.layers[l].moe.num_experts = num_experts;
            model.layers[l].moe.experts = new Qwen36Expert[num_experts]();
            model.num_experts = num_experts;

            for (int32_t e = 0; e < num_experts; e++) {
                auto ep = "mlp.experts." + std::to_string(e) + ".";
                auto* eg = gt(ep + "gate_proj.weight");
                if (eg) model.layers[l].moe.experts[e].gate_proj = *eg;
                auto* eu = gt(ep + "up_proj.weight");
                if (eu) model.layers[l].moe.experts[e].up_proj = *eu;
                auto* ed = gt(ep + "down_proj.weight");
                if (ed) model.layers[l].moe.experts[e].down_proj = *ed;
            }
        }
    }

    // --- LM head ---
    auto* lm_head = find_tensor({"lm_head.weight"});
    if (lm_head) {
        model.lm_head = *lm_head;
    } else {
        // Shared with embeddings
        model.lm_head = model.embeddings;
    }

    // --- Final norm ---
    auto* norm_f = find_tensor({"norm.weight", "model.norm.weight"});
    if (norm_f) {
        model.final_norm = *norm_f;
    }

    // --- RoPE frequencies ---
    auto* freqs = find_tensor({"rope_freqs.inv_freq", "model.rope_freqs_tensor"});
    if (freqs) {
        model.rope_inv_freq = freqs->data;
        model.rope_inv_freq_size = (int32_t)freqs->numel;
    }

    // --- Compute RoPE if not loaded ---
    if (!model.rope_inv_freq) {
        int32_t rope_dim = model.head_dim;
        model.rope_inv_freq = new float[rope_dim / 2]();
        model.rope_inv_freq_size = rope_dim / 2;
        for (int32_t i = 0; i < rope_dim / 2; i++) {
            model.rope_inv_freq[i] = 1.0f / std::pow((float)ROPE_BASE, (2.0f * i) / rope_dim);
        }
    }

    // --- Load tokenizer ---
    std::string vocab_path = model_path + "/vocab.json";
    std::string merges_path = model_path + "/merges.txt";
    if (std::filesystem::exists(vocab_path)) {
        load_vocab(vocab_path, model.tokenizer.vocab, model.tokenizer.inv_vocab);
        std::cout << "  Vocab: " << model.tokenizer.vocab.size() << " tokens" << std::endl;
    }
    if (std::filesystem::exists(merges_path)) {
        std::ifstream mf(merges_path);
        std::string line;
        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto sp = line.find(' ');
            if (sp != std::string::npos && sp + 2 < line.size()) {
                model.tokenizer.merges.emplace_back(line.substr(0, sp), line.substr(sp + 1));
            }
        }
        std::cout << "  Merges: " << model.tokenizer.merges.size() << std::endl;
    }

    // Report
    size_t total_bytes = 0;
    for (auto& [k, v] : tensors) total_bytes += v.byte_size;
    std::cout << "  Total memory: " << (total_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Architecture: " << (model.is_moe ? "MoE" : "Dense")
              << ", " << model.num_layers << "L, h=" << model.hidden_size
              << ", heads=" << model.num_heads << ", head_dim=" << model.head_dim
              << std::endl;

    return model;
}

// ============================================================
// GGUF loading stub
// ============================================================

Qwen36Model load_model_gguf(const std::string& path) {
    std::cerr << "GGUF loading not yet implemented. Use safetensors format." << std::endl;
    return Qwen36Model();
}

// ============================================================
// Vocab loading
// ============================================================

void load_vocab(const std::string& path,
                std::map<std::string, int32_t>& vocab,
                std::map<int32_t, std::string>& inv_vocab) {
    std::ifstream f(path);
    if (!f) return;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    size_t pos = 1;
    while (pos < content.size()) {
        std::string key;
        if (!parse_json_string(content, pos, key)) break;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == ':')) pos++;
        double val;
        if (!parse_json_number(content, pos, val)) break;
        int32_t id = (int32_t)val;
        vocab[key] = id;
        inv_vocab[id] = key;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == ',' ||
               content[pos] == '\n' || content[pos] == '\r')) pos++;
    }
}

// ============================================================
// Tokenizer
// ============================================================

std::vector<int32_t> tokenize(const TokenizerConfig& tok, const std::string& text) {
    if (tok.vocab.empty()) {
        // Fallback: whitespace splitting with dummy token IDs
        std::vector<int32_t> tokens;
        std::istringstream iss(text);
        std::string word;
        int32_t id = 0;
        while (iss >> word) {
            auto it = tok.vocab.find(word);
            id = (it != tok.vocab.end()) ? it->second : tok.unk_token_id;
            tokens.push_back(id);
        }
        return tokens;
    }

    // BPE tokenization
    auto bpe_encode = [&](const std::string& word) -> std::vector<std::string> {
        // Split into UTF-8 chars
        std::vector<std::string> tokens;
        for (size_t i = 0; i < word.size(); ) {
            size_t j = i;
            uint8_t c = (uint8_t)word[i];
            if (c < 0x80) j = i + 1;
            else if (c < 0xE0) j = i + 2;
            else if (c < 0xF0) j = i + 3;
            else j = i + 4;
            if (j > word.size()) j = word.size();
            tokens.push_back(word.substr(i, j - i));
            i = j;
        }

        // Apply merges
        for (int iter = 0; iter < 200 && tokens.size() > 1; iter++) {
            int best_rank = INT32_MAX;
            int best_pos = -1;
            for (int i = 0; i < (int)tokens.size() - 1; i++) {
                std::string pair = tokens[i] + tokens[i + 1];
                for (int m = 0; m < (int)tok.merges.size(); m++) {
                    if (tok.merges[m].first + tok.merges[m].second == pair) {
                        if (m < best_rank) { best_rank = m; best_pos = i; }
                        break;
                    }
                }
            }
            if (best_pos < 0) break;
            tokens[best_pos] = tokens[best_pos] + tokens[best_pos + 1];
            tokens.erase(tokens.begin() + best_pos + 1);
        }
        return tokens;
    };

    // Split text into words (whitespace-separated)
    std::vector<std::string> words;
    {
        std::string current;
        for (char c : text) {
            if (std::isspace((unsigned char)c)) {
                if (!current.empty()) words.push_back(current);
                words.push_back(std::string(1, c));
                current.clear();
            } else {
                current += c;
            }
        }
        if (!current.empty()) words.push_back(current);
    }

    std::vector<int32_t> result;
    for (auto& w : words) {
        auto subtokens = bpe_encode(w);
        for (auto& st : subtokens) {
            auto it = tok.vocab.find(st);
            if (it != tok.vocab.end()) result.push_back(it->second);
            else result.push_back(tok.unk_token_id);
        }
    }
    return result;
}

std::vector<std::string> decode_tokens(const TokenizerConfig& tok, const std::vector<int32_t>& ids) {
    std::vector<std::string> result;
    for (auto id : ids) {
        auto it = tok.inv_vocab.find(id);
        if (it != tok.inv_vocab.end()) result.push_back(it->second);
        else result.push_back("<" + std::to_string(id) + ">");
    }
    return result;
}

// ============================================================
// Core math operations
// ============================================================

static void rms_norm(float* out, const float* input, const float* weight,
                     int32_t dim, float eps) {
    float sum_sq = 0.0f;
    for (int32_t i = 0; i < dim; i++) sum_sq += input[i] * input[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / dim + eps);
    for (int32_t i = 0; i < dim; i++) out[i] = weight[i] * input[i] * inv_rms;
}

static void matmul_transpose(const float* a, int32_t m, int32_t k,
                              const float* b, int32_t n, float* c) {
    // C = A @ B^T: A[m,k] B[n,k] -> C[m,n]
    for (int32_t i = 0; i < m; i++) {
        for (int32_t j = 0; j < n; j++) {
            float sum = 0.0f;
            const float* a_row = a + i * k;
            const float* b_row = b + j * k;
            for (int32_t p = 0; p < k; p++) sum += a_row[p] * b_row[p];
            c[i * n + j] = sum;
        }
    }
}

static void add_vectors(float* out, const float* a, const float* b, int32_t n) {
    for (int32_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static float swiglu(float x) {
    return x * (1.0f / (1.0f + std::exp(-x)));
}

// ============================================================
// RoPE (Rotary Position Embedding)
// ============================================================

static void apply_rope(float* q, float* k, int32_t dim, int32_t head_dim,
                       int32_t pos, const float* inv_freq, int32_t inv_freq_size) {
    int32_t num_heads = dim / head_dim;
    for (int32_t h = 0; h < num_heads; h++) {
        float* q_head = q + h * head_dim;
        float* k_head = k + h * head_dim;
        for (int32_t i = 0; i < head_dim / 2; i++) {
            int32_t fi = std::min(i, inv_freq_size - 1);
            float freq = inv_freq[fi] * pos;
            float cos_v = std::cos(freq);
            float sin_v = std::sin(freq);
            float q0 = q_head[2 * i], q1 = q_head[2 * i + 1];
            float k0 = k_head[2 * i], k1 = k_head[2 * i + 1];
            q_head[2 * i]     = q0 * cos_v - q1 * sin_v;
            q_head[2 * i + 1] = q0 * sin_v + q1 * cos_v;
            k_head[2 * i]     = k0 * cos_v - k1 * sin_v;
            k_head[2 * i + 1] = k0 * sin_v + k1 * cos_v;
        }
    }
}

// ============================================================
// Attention forward
// ============================================================

static void compute_attention(
    float* out,
    const float* q, const float* k_cache, const float* v_cache,
    int32_t seq_len, int32_t num_heads, int32_t head_dim,
    int32_t num_kv_heads) {

    int32_t heads_per_group = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt((float)head_dim);

    float* logits = new float[num_heads * seq_len];
    float* weights = new float[num_heads * seq_len];

    // Q @ K^T
    for (int32_t h = 0; h < num_heads; h++) {
        int32_t kv_h = h / heads_per_group;
        const float* q_h = q + h * head_dim;
        for (int32_t s = 0; s < seq_len; s++) {
            float sum = 0.0f;
            const float* k_h = k_cache + (kv_h * seq_len + s) * head_dim;
            for (int32_t d = 0; d < head_dim; d++) sum += q_h[d] * k_h[d];
            logits[h * seq_len + s] = sum * scale;
        }
    }

    // Softmax per head
    for (int32_t h = 0; h < num_heads; h++) {
        float max_l = -1e9f;
        for (int32_t s = 0; s < seq_len; s++)
            max_l = std::max(max_l, logits[h * seq_len + s]);
        float sum_exp = 0.0f;
        for (int32_t s = 0; s < seq_len; s++) {
            weights[h * seq_len + s] = std::exp(logits[h * seq_len + s] - max_l);
            sum_exp += weights[h * seq_len + s];
        }
        for (int32_t s = 0; s < seq_len; s++)
            weights[h * seq_len + s] /= (sum_exp + 1e-9f);
    }

    // Weighted V
    for (int32_t h = 0; h < num_heads; h++) {
        int32_t kv_h = h / heads_per_group;
        float* o_h = out + h * head_dim;
        std::fill(o_h, o_h + head_dim, 0.0f);
        for (int32_t s = 0; s < seq_len; s++) {
            float w = weights[h * seq_len + s];
            const float* v_h = v_cache + (kv_h * seq_len + s) * head_dim;
            for (int32_t d = 0; d < head_dim; d++) o_h[d] += w * v_h[d];
        }
    }

    delete[] logits;
    delete[] weights;
}

// ============================================================
// MoE forward (CPU)
// ============================================================

static void moe_forward(
    float* out,
    const float* hidden, int32_t hidden_size,
    const Qwen36MoELayer& moe, int32_t top_k) {

    int32_t ffn_dim = (int32_t)moe.experts[0].gate_proj.shape[0];

    // Gate scores
    float* gate_scores = new float[moe.num_experts]();
    for (int32_t e = 0; e < moe.num_experts; e++) {
        float sum = 0.0f;
        for (int32_t d = 0; d < hidden_size; d++)
            sum += moe.gate.data[e * hidden_size + d] * hidden[d];
        gate_scores[e] = sum;
    }

    // Softmax
    float max_s = gate_scores[0];
    for (int32_t e = 1; e < moe.num_experts; e++) max_s = std::max(max_s, gate_scores[e]);
    float sum_e = 0.0f;
    for (int32_t e = 0; e < moe.num_experts; e++) {
        gate_scores[e] = std::exp(gate_scores[e] - max_s);
        sum_e += gate_scores[e];
    }
    for (int32_t e = 0; e < moe.num_experts; e++) gate_scores[e] /= (sum_e + 1e-9f);

    // Top-k selection
    std::vector<std::pair<float, int32_t>> scored(moe.num_experts);
    for (int32_t i = 0; i < moe.num_experts; i++) scored[i] = {gate_scores[i], i};
    std::sort(scored.begin(), scored.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    // Compute + combine
    std::fill(out, out + hidden_size, 0.0f);
    float* expert_out = new float[ffn_dim]();
    float* gate_act = new float[ffn_dim]();
    float* up_act = new float[ffn_dim]();

    for (int32_t k = 0; k < top_k && k < moe.num_experts; k++) {
        int32_t eid = scored[k].second;
        float score = scored[k].first;
        const auto& expert = moe.experts[eid];

        // Gate + Up projections
        for (int32_t i = 0; i < ffn_dim; i++) {
            float gs = 0.0f, us = 0.0f;
            for (int32_t d = 0; d < hidden_size; d++) {
                gs += expert.gate_proj.data[i * hidden_size + d] * hidden[d];
                us += expert.up_proj.data[i * hidden_size + d] * hidden[d];
            }
            gate_act[i] = gs;
            up_act[i] = us;
        }

        // SwiGLU
        for (int32_t i = 0; i < ffn_dim; i++)
            expert_out[i] = swiglu(gate_act[i]) * up_act[i];

        // Down projection
        for (int32_t d = 0; d < hidden_size; d++) {
            float sum = 0.0f;
            for (int32_t i = 0; i < ffn_dim; i++)
                sum += expert.down_proj.data[d * ffn_dim + i] * expert_out[i];
            out[d] += sum * score;
        }
    }

    delete[] gate_scores;
    delete[] expert_out;
    delete[] gate_act;
    delete[] up_act;
}

// ============================================================
// Forward pass (CPU)
// ============================================================

std::vector<float> forward_cpu(
    const Qwen36Model& model,
    const std::vector<int32_t>& input_ids,
    int32_t /*batch_size*/,
    bool debug) {

    int32_t hidden = model.hidden_size;
    int32_t seq_len = (int32_t)input_ids.size();
    int32_t num_layers = model.num_layers;
    int32_t kv_dim = model.num_kv_heads * model.head_dim;
    int32_t total_heads_dim = model.num_heads * model.head_dim;

    // KV cache: [layer][seq][kv_dim]
    float* k_cache = new float[num_layers * seq_len * kv_dim]();
    float* v_cache = new float[num_layers * seq_len * kv_dim]();

    // Hidden states
    float* hidden_states = new float[hidden]();

    // Add embeddings of all input tokens
    for (int32_t id : input_ids) {
        if (id >= 0 && id < model.vocab_size) {
            const float* emb = model.embeddings.data + id * hidden;
            for (int32_t d = 0; d < hidden; d++) hidden_states[d] += emb[d];
        }
    }

    // Temporary buffers
    float* residual = new float[hidden]();
    float* norm_out = new float[hidden]();
    float* attn_out = new float[hidden]();
    float* q = new float[total_heads_dim]();
    float* k = new float[kv_dim]();
    float* v = new float[kv_dim]();
    float* attn_result = new float[total_heads_dim]();
    float* mlp_out = new float[hidden]();
    float* gate_act = new float[model.ffn_intermediate]();
    float* up_act = new float[model.ffn_intermediate]();

    // Process each layer
    for (int32_t l = 0; l < num_layers; l++) {
        const auto& layer = model.layers[l];

        // Input normalization
        rms_norm(norm_out, hidden_states, layer.input_norm.weight.data,
                 hidden, layer.input_norm.eps);

        // Save residual
        std::copy(hidden_states, hidden_states + hidden, residual);

        // --- Attention ---
        // Q projection: norm @ q_proj^T
        matmul_transpose(layer.attn.q_proj.data, hidden, hidden,
                         norm_out, 1, q);

        // K projection
        matmul_transpose(layer.attn.k_proj.data, kv_dim, hidden,
                         norm_out, 1, k);

        // V projection
        matmul_transpose(layer.attn.v_proj.data, kv_dim, hidden,
                         norm_out, 1, v);

        // RoPE (apply to last position)
        apply_rope(q, k, total_heads_dim, model.head_dim,
                   seq_len - 1, model.rope_inv_freq, model.rope_inv_freq_size);

        // Store in KV cache
        std::copy(k, k + kv_dim,
                  k_cache + l * seq_len * kv_dim + (seq_len - 1) * kv_dim);
        std::copy(v, v + kv_dim,
                  v_cache + l * seq_len * kv_dim + (seq_len - 1) * kv_dim);

        // Attention
        compute_attention(attn_result, q,
                          k_cache + l * seq_len * kv_dim,
                          v_cache + l * seq_len * kv_dim,
                          seq_len, model.num_heads, model.head_dim, model.num_kv_heads);

        // Output projection
        matmul_transpose(layer.attn.o_proj.data, hidden, total_heads_dim,
                         attn_result, 1, attn_out);

        // Residual
        add_vectors(hidden_states, residual, attn_out, hidden);

        // --- MLP / MoE ---
        rms_norm(norm_out, hidden_states, layer.attn_norm.weight.data,
                 hidden, layer.attn_norm.eps);

        std::copy(hidden_states, hidden_states + hidden, residual);

        if (layer.moe.experts && layer.moe.gate.data) {
            // MoE path
            moe_forward(mlp_out, norm_out, hidden, layer.moe, model.top_k);
        } else if (layer.mlp.gate_proj.data) {
            // Dense MLP
            matmul_transpose(layer.mlp.gate_proj.data, model.ffn_intermediate, hidden,
                             norm_out, 1, gate_act);
            matmul_transpose(layer.mlp.up_proj.data, model.ffn_intermediate, hidden,
                             norm_out, 1, up_act);
            for (int32_t i = 0; i < model.ffn_intermediate; i++)
                mlp_out[i] = swiglu(gate_act[i]) * up_act[i];
            matmul_transpose(layer.mlp.down_proj.data, hidden, model.ffn_intermediate,
                             mlp_out, 1, mlp_out);
        }

        // Residual
        add_vectors(hidden_states, residual, mlp_out, hidden);
    }

    // Final norm
    rms_norm(norm_out, hidden_states, model.final_norm.data,
             hidden, model.norm_eps);

    // LM head
    int32_t vocab = model.vocab_size;
    std::vector<float> logits(vocab);
    matmul_transpose(model.lm_head.data, vocab, hidden,
                     norm_out, 1, logits.data());

    // Cleanup
    delete[] k_cache; delete[] v_cache;
    delete[] hidden_states; delete[] residual;
    delete[] norm_out; delete[] attn_out;
    delete[] q; delete[] k; delete[] v;
    delete[] attn_result; delete[] mlp_out;
    delete[] gate_act; delete[] up_act;

    if (debug) {
        std::vector<std::pair<float, int32_t>> top(logits.size());
        for (int32_t i = 0; i < (int32_t)logits.size(); i++) top[i] = {logits[i], (int32_t)i};
        std::sort(top.begin(), top.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });
        std::cout << "Top-5: ";
        for (int32_t i = 0; i < 5 && i < (int32_t)top.size(); i++)
            std::cout << "(" << top[i].second << ":" << top[i].first << ") ";
        std::cout << std::endl;
    }

    return logits;
}

// ============================================================
// Sampling
// ============================================================

int32_t sample_token(
    const std::vector<float>& logits,
    float temperature, float top_p, float repeat_penalty,
    const std::vector<int32_t>* history) {

    int32_t vocab_size = (int32_t)logits.size();
    std::vector<float> scores(logits.begin(), logits.end());

    // Repeat penalty
    if (history) {
        for (auto id : *history) {
            if (id >= 0 && id < vocab_size) {
                if (scores[id] > 0) scores[id] /= repeat_penalty;
                else if (scores[id] < 0) scores[id] *= repeat_penalty;
            }
        }
    }

    // Temperature scaling
    if (temperature != 1.0f) {
        for (auto& s : scores) s /= temperature;
    }

    // Rank by score
    std::vector<std::pair<float, int32_t>> ranked(scores.size());
    for (int32_t i = 0; i < vocab_size; i++) ranked[i] = {scores[i], i};
    std::sort(ranked.begin(), ranked.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    // Softmax
    float max_s = ranked[0].first;
    for (auto& r : ranked) r.first = std::exp(r.first - max_s);
    float sum = 0.0f;
    for (auto& r : ranked) sum += r.first;
    for (auto& r : ranked) r.first /= sum;

    // Top-p
    float cum = 0.0f;
    int32_t keep = 0;
    for (int32_t i = 0; i < vocab_size; i++) {
        cum += ranked[i].first;
        keep = i + 1;
        if (cum >= top_p) break;
    }

    // Rescale
    float kept_sum = 0.0f;
    for (int32_t i = 0; i < keep; i++) kept_sum += ranked[i].first;
    for (int32_t i = 0; i < keep; i++) ranked[i].first /= kept_sum;

    // Sample
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(gen);
    cum = 0.0f;
    for (int32_t i = 0; i < keep; i++) {
        cum += ranked[i].first;
        if (r <= cum) return ranked[i].second;
    }
    return ranked[0].second;
}

// ============================================================
// Generation loop
// ============================================================

std::vector<int32_t> generate_cpu(
    const Qwen36Model& model,
    const std::vector<int32_t>& prompt_tokens,
    int32_t max_new_tokens,
    float temperature, float top_p,
    std::function<void(int32_t)> on_token) {

    std::vector<int32_t> output = prompt_tokens;

    for (int32_t i = 0; i < max_new_tokens; i++) {
        auto logits = forward_cpu(model, output, 1, false);

        // Use tail of history for repeat penalty (last 64 tokens)
        std::vector<int32_t> hist;
        int32_t start = std::max(0, (int32_t)output.size() - 64);
        for (int32_t j = start; j < (int32_t)output.size(); j++) hist.push_back(output[j]);

        int32_t next = sample_token(logits, temperature, top_p, 1.1f, &hist);

        if (next == model.tokenizer.eos_token_id) break;

        output.push_back(next);
        if (on_token) on_token(next);
    }

    return output;
}

// ============================================================
// BlockPool
// ============================================================

BlockPool::BlockPool(int32_t num_blocks) : blocks(num_blocks) {
    for (auto& b : blocks) {
        b.block_id = (int32_t)(&b - blocks.data());
        b.free = true;
    }
    next_free = 0;
}

int32_t BlockPool::alloc() {
    for (int32_t i = next_free; i < (int32_t)blocks.size(); i++) {
        if (blocks[i].free) {
            blocks[i].free = false;
            blocks[i].ref_count = 1;
            next_free = i + 1;
            return blocks[i].block_id;
        }
    }
    for (int32_t i = 0; i < next_free; i++) {
        if (blocks[i].free) {
            blocks[i].free = false;
            blocks[i].ref_count = 1;
            next_free = i + 1;
            return blocks[i].block_id;
        }
    }
    return -1;
}

void BlockPool::free(int32_t block_id) {
    if (block_id < 0 || block_id >= (int32_t)blocks.size()) return;
    blocks[block_id].ref_count--;
    if (blocks[block_id].ref_count <= 0) {
        blocks[block_id].free = true;
        blocks[block_id].ref_count = 0;
        if (block_id < next_free) next_free = block_id;
    }
}

bool BlockPool::is_free(int32_t block_id) const {
    return block_id >= 0 && block_id < (int32_t)blocks.size() && blocks[block_id].free;
}
