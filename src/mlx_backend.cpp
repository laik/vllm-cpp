#include "mlx_backend.h"

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/random.h>
#include <mlx/device.h>
#include <mlx/io/load.h>

#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <algorithm>

namespace mx = mlx::core;

// ============================================================
// Helpers
// ============================================================

static mx::array* a(void* p) { return static_cast<mx::array*>(p); }
static const mx::array* a(const void* p) { return static_cast<const mx::array*>(p); }

static void* store(mx::array&& arr) {
    return new mx::array(std::move(arr));
}

// Convert BF16 raw bytes to float32
static float bf16_to_f32(uint16_t v) {
    uint32_t bits = ((uint32_t)v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Convert F16 raw bytes to float32
static float f16_to_f32(uint16_t v) {
    // IEEE 754 half → float
    uint32_t sign = (v & 0x8000u) << 16;
    uint32_t exp  = (v & 0x7C00u) >> 10;
    uint32_t mant = (v & 0x03FFu);
    uint32_t f32;
    if (exp == 0) {
        f32 = sign | (mant << 13);
    } else if (exp == 0x1F) {
        f32 = sign | 0x7F800000u | (mant << 13);
    } else {
        exp = exp + (127 - 15);
        f32 = sign | (exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &f32, sizeof(f));
    return f;
}

// Minimal JSON helpers for safetensors header
namespace json_ {
    static void skip_ws(const std::string& s, size_t& pos) {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' ||
               s[pos] == '\r' || s[pos] == '\t')) pos++;
    }
    static bool parse_str(const std::string& s, size_t& pos, std::string& out) {
        if (pos >= s.size() || s[pos] != '"') return false;
        pos++; out.clear();
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                pos++;
                switch (s[pos]) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    default: out += s[pos]; break;
                }
            } else { out += s[pos]; }
            pos++;
        }
        if (pos < s.size()) pos++; // skip closing quote
        return true;
    }
    static bool parse_num(const std::string& s, size_t& pos, double& out) {
        std::string num;
        if (pos < s.size() && s[pos] == '-') { num += s[pos++]; }
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') num += s[pos++];
        if (pos < s.size() && s[pos] == '.') {
            num += s[pos++];
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') num += s[pos++];
        }
        if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
            num += s[pos++];
            if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) num += s[pos++];
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') num += s[pos++];
        }
        if (num.empty()) return false;
        out = std::stod(num);
        return true;
    }
} // namespace json_

// ============================================================
// Safetensors loading into MLX arrays
// ============================================================

static std::map<std::string, void*> load_safetensors_mlx(const std::string& dir) {
    namespace fs = std::filesystem;
    std::string sf_path;
    for (auto& name : {"model.safetensors", "model-00001-of-00001.safetensors"}) {
        fs::path p = fs::path(dir) / name;
        if (fs::exists(p)) { sf_path = p.string(); break; }
    }
    if (sf_path.empty()) {
        std::cerr << "[ERROR] No safetensors file found in " << dir << std::endl;
        return {};
    }

    std::ifstream f(sf_path, std::ios::binary);
    if (!f) { std::cerr << "[ERROR] Cannot open " << sf_path << std::endl; return {}; }

    // Read header
    uint64_t header_size;
    f.read(reinterpret_cast<char*>(&header_size), 8);
    std::string header_json(header_size, '\0');
    f.read(&header_json[0], header_size);

    // Minimal JSON parse: extract name, dtype, shape, offsets
    struct TensorInfo {
        std::string name, dtype;
        std::vector<int> shape;
        size_t start, end;
    };
    std::vector<TensorInfo> tensors;

    size_t pos = 1; // skip opening brace
    while (pos < header_json.size()) {
        json_::skip_ws(header_json, pos);
        if (pos >= header_json.size() || header_json[pos] == '}') break;

        TensorInfo info;
        if (!json_::parse_str(header_json, pos, info.name)) break;

        // Skip ": {
        json_::skip_ws(header_json, pos);
        if (header_json[pos] == ':') pos++;
        json_::skip_ws(header_json, pos);
        if (header_json[pos] == '{') pos++;

        while (pos < header_json.size() && header_json[pos] != '}') {
            json_::skip_ws(header_json, pos);
            std::string key;
            if (!json_::parse_str(header_json, pos, key)) break;
            json_::skip_ws(header_json, pos);
            if (header_json[pos] == ':') pos++;
            json_::skip_ws(header_json, pos);

            if (key == "dtype") {
                json_::parse_str(header_json, pos, info.dtype);
            } else if (key == "shape") {
                if (header_json[pos] == '[') {
                    pos++; // skip [
                    json_::skip_ws(header_json, pos);
                    while (pos < header_json.size() && header_json[pos] != ']') {
                        double v;
                        json_::parse_num(header_json, pos, v);
                        info.shape.push_back((int)v);
                        json_::skip_ws(header_json, pos);
                        if (header_json[pos] == ',') pos++;
                        json_::skip_ws(header_json, pos);
                    }
                    if (header_json[pos] == ']') pos++;
                }
            } else if (key == "data_offsets") {
                if (header_json[pos] == '[') {
                    pos++;
                    double s, e;
                    json_::parse_num(header_json, pos, s);
                    info.start = (size_t)s;
                    json_::skip_ws(header_json, pos);
                    if (header_json[pos] == ',') pos++;
                    json_::parse_num(header_json, pos, e);
                    info.end = (size_t)e;
                    json_::skip_ws(header_json, pos);
                    if (header_json[pos] == ']') pos++;
                }
            }
            json_::skip_ws(header_json, pos);
            if (header_json[pos] == ',') pos++;
        }
        if (header_json[pos] == '}') pos++; // skip closing brace of tensor
        json_::skip_ws(header_json, pos);
        if (header_json[pos] == ',') pos++;

        if (info.name != "__metadata__" && !info.shape.empty()) {
            tensors.push_back(std::move(info));
        }
    }

    size_t data_start = 8 + header_size;

    // Default device (GPU on Mac)
    auto device = mx::Device(mx::Device::gpu);

    std::map<std::string, void*> weights;
    std::vector<uint8_t> raw;

    for (auto& t : tensors) {
        size_t byte_count = t.end - t.start;
        raw.resize(byte_count);
        f.seekg(data_start + t.start);
        f.read(reinterpret_cast<char*>(raw.data()), byte_count);

        int total_elems = 1;
        for (int s : t.shape) total_elems *= s;

        // Convert to float32
        std::vector<float> f32_data(total_elems);
        if (t.dtype == "BF16") {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
            for (int i = 0; i < total_elems; i++) f32_data[i] = bf16_to_f32(src[i]);
        } else if (t.dtype == "F16") {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
            for (int i = 0; i < total_elems; i++) f32_data[i] = f16_to_f32(src[i]);
        } else if (t.dtype == "F32") {
            std::memcpy(f32_data.data(), raw.data(), byte_count);
        } else {
            std::cerr << "[WARN] Unknown dtype " << t.dtype << " for " << t.name << std::endl;
            continue;
        }

        // Create MLX array
        auto arr = mx::array(f32_data.data(), mx::Shape(t.shape.begin(), t.shape.end()), mx::float32);
        mx::eval(arr); // Force evaluation on device

        weights[t.name] = new mx::array(std::move(arr));
    }

    return weights;
}

// ============================================================
// MlxModel destructor
// ============================================================

MlxModel::~MlxModel() {
    for (auto& [name, ptr] : weights) delete a(ptr);
    for (auto& [k, v] : kv_cache) {
        if (k) delete a(k);
        if (v) delete a(v);
    }
}

// ============================================================
// Model loading
// ============================================================

MlxModel* mlx_load_model(const std::string& model_path) {
    auto* m = new MlxModel();
    m->model_path = model_path;

    std::cerr << "[INFO] Loading weights into MLX..." << std::endl;
    m->weights = load_safetensors_mlx(model_path);

    // Infer architecture from weight shapes
    auto* embed = a(m->weights["model.embed_tokens.weight"]);
    m->vocab_size = embed->shape(0);
    m->hidden_size = embed->shape(1);

    // Count layers
    while (m->weights.count("model.layers." + std::to_string(m->num_layers) + ".input_layernorm.weight"))
        m->num_layers++;

    // Head dimensions from Q and K shapes
    auto* q0 = a(m->weights["model.layers.0.self_attn.q_proj.weight"]);
    auto* k0 = a(m->weights["model.layers.0.self_attn.k_proj.weight"]);

    // K has shape [num_kv_heads * head_dim, hidden]
    // Q has shape [num_heads * head_dim, hidden]
    // Qwen2.5-0.5B: num_kv_heads=2, head_dim=64
    // So K shape: [128, hidden], Q shape: [896, hidden]
    // We need to detect head_dim from K shape
    // Assume num_kv_heads=2 for 0.5B, but let's detect from Q/K ratio
    int k_dim0 = k0->shape(0); // num_kv_heads * head_dim
    int q_dim0 = q0->shape(0); // num_heads * head_dim

    // head_dim detection: try common values, prefer ones giving num_kv_heads > 1
    m->head_dim = 64;
    for (int d : {64, 80, 96, 128, 48, 32}) {
        if (k_dim0 % d == 0 && q_dim0 % d == 0 && k_dim0 / d > 1) {
            m->head_dim = d;
            break;
        }
    }
    m->num_kv_heads = k_dim0 / m->head_dim;
    m->num_heads = q_dim0 / m->head_dim;

    // FFN intermediate from gate_proj
    auto* gate0 = a(m->weights["model.layers.0.mlp.gate_proj.weight"]);
    m->ffn_intermediate = gate0->shape(0);

    // RoPE base
    m->rope_base = 1000000.0f; // Qwen2.5 default

    // RMSNorm epsilon
    m->norm_eps = 1e-6f;

    std::cerr << "[INFO] MLX model: " << m->num_layers << " layers, "
              << m->hidden_size << " hidden, "
              << m->num_heads << " heads (" << m->num_kv_heads << " KV), "
              << m->ffn_intermediate << " ffn, "
              << m->vocab_size << " vocab"
              << std::endl;

    // Load tokenizer
    std::string vocab_path = model_path + "/vocab.json";
    load_vocab(vocab_path, m->tokenizer.vocab, m->tokenizer.inv_vocab);
    m->tokenizer.vocab_path = vocab_path;

    // Load BPE merges
    std::string merges_path = model_path + "/merges.txt";
    if (std::filesystem::exists(merges_path)) {
        std::ifstream mf(merges_path);
        std::string line;
        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sp = line.find(' ');
            if (sp != std::string::npos) {
                m->tokenizer.merges.emplace_back(line.substr(0, sp), line.substr(sp + 1));
            }
        }
    }

    // Load added (special) tokens
    load_added_tokens(model_path, m->tokenizer.vocab, m->tokenizer.inv_vocab);
    // Update special token IDs from added tokens
    if (m->tokenizer.vocab.count("<|endoftext|>"))
        m->tokenizer.bos_token_id = m->tokenizer.vocab["<|endoftext|>"];
    if (m->tokenizer.vocab.count("<|im_end|>"))
        m->tokenizer.eos_token_id = m->tokenizer.vocab["<|im_end|>"];

    return m;
}

// ============================================================
// Forward pass (full sequence → logits for last position)
// ============================================================

static mx::array mlx_forward(MlxModel* model, const std::vector<int32_t>& input_ids) {
    int32_t seq_len = (int32_t)input_ids.size();
    int32_t hidden = model->hidden_size;
    int32_t num_layers = model->num_layers;
    int32_t num_heads = model->num_heads;
    int32_t num_kv_heads = model->num_kv_heads;
    int32_t head_dim = model->head_dim;
    int32_t ffn_dim = model->ffn_intermediate;
    float norm_eps = model->norm_eps;
    float rope_base = model->rope_base;

    auto& w = model->weights;

    // Embedding lookup using gather
    auto token_ids = mx::array(input_ids.data(), {seq_len}, mx::int32);
    auto h = mx::gather(
        *a(w["model.embed_tokens.weight"]),
        std::vector<mx::array>{token_ids},
        std::vector<int>{0},
        mx::Shape{1, hidden});  // gather returns [seq_len, 1, hidden]
    h = mx::reshape(h, {seq_len, hidden});  // squeeze to [seq_len, hidden]

    // Precompute SiLU
    auto silu = [](const mx::array& x) -> mx::array {
        return mx::multiply(x, mx::sigmoid(x));
    };

    for (int32_t l = 0; l < num_layers; l++) {
        std::string pfx = "model.layers." + std::to_string(l) + ".";

        // --- RMSNorm ---
        auto normed = mx::fast::rms_norm(
            h,
            *a(w[pfx + "input_layernorm.weight"]),
            norm_eps);
        auto residual = h;

        // --- Attention ---
        // Q projection
        auto q = mx::matmul(normed, mx::transpose(*a(w[pfx + "self_attn.q_proj.weight"])));
        if (w.count(pfx + "self_attn.q_proj.bias"))
            q = mx::add(q, *a(w[pfx + "self_attn.q_proj.bias"]));

        // K projection
        auto k = mx::matmul(normed, mx::transpose(*a(w[pfx + "self_attn.k_proj.weight"])));
        if (w.count(pfx + "self_attn.k_proj.bias"))
            k = mx::add(k, *a(w[pfx + "self_attn.k_proj.bias"]));

        // V projection
        auto v = mx::matmul(normed, mx::transpose(*a(w[pfx + "self_attn.v_proj.weight"])));
        if (w.count(pfx + "self_attn.v_proj.bias"))
            v = mx::add(v, *a(w[pfx + "self_attn.v_proj.bias"]));

        // Reshape for multi-head attention: [1, seq_len, heads, head_dim]
        q = mx::reshape(q, {1, seq_len, num_heads, head_dim});
        k = mx::reshape(k, {1, seq_len, num_kv_heads, head_dim});
        v = mx::reshape(v, {1, seq_len, num_kv_heads, head_dim});

        // RoPE — MLX rope uses axis -2 as position axis, so we need
        // [B, heads, seq, dim] layout. Transpose: [B, seq, heads, dim] → [B, heads, seq, dim]
        q = mx::transpose(q, {0, 2, 1, 3});
        k = mx::transpose(k, {0, 2, 1, 3});
        q = mx::fast::rope(q, head_dim, false, rope_base, 1.0f, 0);
        k = mx::fast::rope(k, head_dim, false, rope_base, 1.0f, 0);
        // Transpose back for SDPA: [B, heads, seq, dim] → [B, seq, heads, dim]
        q = mx::transpose(q, {0, 2, 1, 3});
        k = mx::transpose(k, {0, 2, 1, 3});

        // Manual attention — compute per KV head group
        // (MLX SDPA is broken with GQA; manual Q@K^T → mask → softmax → @V works)
        float scale = 1.0f / std::sqrt((float)head_dim);
        int32_t heads_per_group = num_heads / num_kv_heads;
        auto neg_inf = mx::array(-1e9f);
        auto causal_mask = mx::tril(mx::ones({seq_len, seq_len}, mx::float32));
        std::vector<mx::array> attn_parts;
        attn_parts.reserve(num_kv_heads);

        for (int32_t g = 0; g < num_kv_heads; g++) {
            // Slice Q group: [1, seq_len, heads_per_group, head_dim] → [seq_len, heads_per_group, head_dim]
            auto q_group = mx::reshape(
                mx::slice(q, {0, 0, g * heads_per_group, 0},
                              {1, seq_len, (g + 1) * heads_per_group, head_dim}),
                {seq_len, heads_per_group, head_dim});
            // Slice K, V: [1, seq_len, 1, head_dim] → [seq_len, 1, head_dim]
            auto k_g = mx::reshape(
                mx::slice(k, {0, 0, g, 0}, {1, seq_len, g + 1, head_dim}),
                {seq_len, 1, head_dim});
            auto v_g = mx::reshape(
                mx::slice(v, {0, 0, g, 0}, {1, seq_len, g + 1, head_dim}),
                {seq_len, 1, head_dim});

            // scores = Q @ K^T * scale → [heads_per_group, seq_len, seq_len]
            auto q_t = mx::transpose(q_group, {1, 0, 2});   // [heads_per_group, seq_len, head_dim]
            auto k_t = mx::transpose(k_g, {1, 2, 0});       // [1, head_dim, seq_len]
            auto scores = mx::multiply(mx::matmul(q_t, k_t), mx::array(scale));

            // Causal mask
            scores = mx::where(causal_mask, scores, mx::multiply(mx::ones_like(scores), neg_inf));

            // Softmax
            auto probs = mx::softmax(scores, -1);  // [heads_per_group, seq_len, seq_len]

            // probs @ V → [heads_per_group, seq_len, head_dim]
            auto out = mx::matmul(probs, mx::reshape(v_g, {1, seq_len, head_dim}));

            // [heads_per_group, seq_len, head_dim] → [seq_len, heads_per_group, head_dim]
            out = mx::transpose(out, {1, 0, 2});
            attn_parts.push_back(out);
        }
        auto attn_out = mx::concatenate(attn_parts, 1); // concat along heads axis
        attn_out = mx::reshape(attn_out, {seq_len, num_heads * head_dim});
        attn_out = mx::matmul(attn_out, mx::transpose(*a(w[pfx + "self_attn.o_proj.weight"])));

        h = mx::add(residual, attn_out);

        // --- MLP ---
        normed = mx::fast::rms_norm(
            h,
            *a(w[pfx + "post_attention_layernorm.weight"]),
            norm_eps);
        residual = h;

        auto gate = mx::matmul(normed, mx::transpose(*a(w[pfx + "mlp.gate_proj.weight"])));
        auto up   = mx::matmul(normed, mx::transpose(*a(w[pfx + "mlp.up_proj.weight"])));
        auto mlp_out = mx::matmul(
            mx::multiply(silu(gate), up),
            mx::transpose(*a(w[pfx + "mlp.down_proj.weight"])));

        h = mx::add(residual, mlp_out);
    }

    // Final norm
    h = mx::fast::rms_norm(h, *a(w["model.norm.weight"]), norm_eps);

    // Get last position
    h = mx::slice(h, {seq_len - 1, 0}, {seq_len, hidden});
    h = mx::reshape(h, {hidden});

    // LM head
    auto* lm_head_w = w.count("lm_head.weight")
        ? a(w["lm_head.weight"]) : a(w["model.embed_tokens.weight"]);
    auto logits = mx::matmul(h, mx::transpose(*lm_head_w));


    return logits;
}

// ============================================================
// Sampling
// ============================================================

static int32_t mlx_sample(const mx::array& logits, float temperature, float top_p, bool debug) {
    auto vocab_size = logits.shape(0);

    if (debug) {
        auto lmin = mx::min(logits, -1, false);
        auto lmax = mx::max(logits, -1, false);
        auto lmean = mx::mean(logits, -1, false);
        auto lstd = mx::sqrt(mx::mean(mx::square(mx::subtract(logits, lmean)), -1, false));
        mx::eval(lmin, lmax, lmean, lstd);
        std::cerr << "[DEBUG] logits[" << vocab_size << "]: min=" << lmin.item<float>()
                  << " max=" << lmax.item<float>()
                  << " mean=" << lmean.item<float>()
                  << " std=" << lstd.item<float>() << std::endl;
    }

    // Greedy: argmax
    if (temperature <= 0.0f) {
        auto idx = mx::argmax(logits, -1);
        int32_t token = static_cast<int32_t>(idx.item<int32_t>());
        if (debug) {
            float max_logit = mx::take(logits, idx, -1).item<float>();
            std::cerr << "[DEBUG] greedy token=" << token << " logit=" << max_logit << std::endl;
        }
        return token;
    }

    // Scale logits by temperature (categorical takes logits, not probs)
    mx::array scaled_logits = logits;
    if (temperature != 1.0f) {
        scaled_logits = mx::divide(logits, mx::array(temperature));
    }

    if (top_p < 1.0f) {
        // Compute probs for top-p filtering from scaled logits
        auto probs = mx::softmax(scaled_logits, -1);
        auto descending = mx::argsort(mx::negative(probs), -1);
        auto sorted_probs = mx::take_along_axis(probs, descending, -1);
        auto cumsum = mx::cumsum(sorted_probs, -1);

        auto mask = mx::greater(cumsum, mx::array(top_p));
        // Shift mask right: mask[i] = mask[i-1], mask[0] = false
        auto mask_no_last = mx::slice(mask, {0}, {vocab_size - 1});
        auto shifted = mx::concatenate(
            std::vector<mx::array>{mx::zeros({1}, mx::bool_), mask_no_last});

        sorted_probs = mx::where(shifted,
                                 mx::zeros({vocab_size}, mx::float32),
                                 sorted_probs);
        auto sum_p = mx::add(mx::sum(sorted_probs, false), mx::array(1e-9f));
        sorted_probs = mx::divide(sorted_probs, sum_p);

        // categorical expects logits, convert filtered probs back to log space
        auto sorted_logits = mx::log(mx::maximum(sorted_probs, mx::array(1e-9f)));
        auto idx = mx::random::categorical(sorted_logits, -1);
        int32_t sorted_token = static_cast<int32_t>(idx.item<int32_t>());

        // Map back to original token id
        auto orig_token = mx::take(descending, sorted_token, -1);
        return static_cast<int32_t>(orig_token.item<int32_t>());
    }

    auto idx = mx::random::categorical(scaled_logits, -1);
    return static_cast<int32_t>(idx.item<int32_t>());
}

// ============================================================
// Generation loop
// ============================================================

std::vector<int32_t> mlx_generate(
    MlxModel* model,
    const std::vector<int32_t>& prompt_tokens,
    int32_t max_new_tokens,
    float temperature, float top_p,
    std::function<void(int32_t)> on_token,
    bool debug)
{
    std::vector<int32_t> tokens = prompt_tokens;

    if (debug) {
        std::cerr << "[DEBUG] prompt has " << tokens.size() << " tokens:";
        for (size_t i = 0; i < tokens.size() && i < 20; i++)
            std::cerr << " " << tokens[i];
        if (tokens.size() > 20) std::cerr << " ...";
        std::cerr << std::endl;
    }

    for (int32_t step = 0; step < max_new_tokens; step++) {
        // Truncate to max context (512 tokens)
        if ((int32_t)tokens.size() > 512) {
            tokens.erase(tokens.begin(), tokens.begin() + (tokens.size() - 512));
        }

        auto logits = mlx_forward(model, tokens);
        mx::eval(logits); // Force GPU evaluation

        int32_t next = mlx_sample(logits, temperature, top_p, debug);

        if (next == model->tokenizer.eos_token_id) break;

        tokens.push_back(next);
        if (on_token) on_token(next);
    }

    return tokens;
}
