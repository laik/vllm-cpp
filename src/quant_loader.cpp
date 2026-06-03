#include "quant_loader.h"
#include "quantization.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <map>
#include <set>

// ============================================================
// Parse quantization_config from config.json
// ============================================================

// Minimal JSON value parser for config.json
static std::string json_get_string(const std::string& json, const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t' || json[pos] == '\r'))
        pos++;
    if (pos >= json.size() || json[pos] != '"') return "";
    pos++;
    std::string val;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) pos++;
        else val += json[pos];
        pos++;
    }
    return val;
}

static int64_t json_get_int(const std::string& json, const std::string& key, int64_t default_val = 0) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_val;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t' || json[pos] == '\r'))
        pos++;
    bool neg = false;
    if (pos < json.size() && json[pos] == '-') { neg = true; pos++; }
    int64_t val = 0;
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9')
        val = val * 10 + (json[pos++] - '0');
    return neg ? -val : val;
}

static bool json_get_bool(const std::string& json, const std::string& key, bool default_val = false) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_val;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t' || json[pos] == '\r'))
        pos++;
    if (json.substr(pos, 4) == "true") return true;
    if (json.substr(pos, 5) == "false") return false;
    return default_val;
}

QuantConfig parse_quant_config(const std::string& config_json) {
    QuantConfig cfg;

    // Try to find quantization_config section
    std::string quant_method = json_get_string(config_json, "quant_method");
    if (quant_method.empty()) {
        // Check inside quantization_config object
        auto pos = config_json.find("\"quantization_config\"");
        if (pos != std::string::npos) {
            // Extract the quantization_config sub-object
            auto start = config_json.find("{", pos);
            if (start != std::string::npos) {
                // Find matching closing brace (simple bracket counting)
                int depth = 0;
                auto end = start;
                for (size_t i = start; i < config_json.size(); i++) {
                    if (config_json[i] == '{') depth++;
                    else if (config_json[i] == '}') { depth--; if (depth == 0) { end = i + 1; break; } }
                }
                std::string sub = config_json.substr(start, end - start);
                quant_method = json_get_string(sub, "quant_method");
                if (!quant_method.empty()) {
                    cfg.bits = (int32_t)json_get_int(sub, "bits", 4);
                    cfg.group_size = (int32_t)json_get_int(sub, "group_size", 128);
                    cfg.desc_act = json_get_bool(sub, "desc_act", false);
                    cfg.sym = json_get_bool(sub, "sym", true);
                }
            }
        }
    }

    // Map method name to enum
    if (quant_method == "gptq") {
        cfg.method = QuantMethod::GPTQ;
    } else if (quant_method == "awq") {
        cfg.method = QuantMethod::AWQ;
    } else if (quant_method == "fp8") {
        cfg.method = QuantMethod::FP8;
    } else if (!quant_method.empty()) {
        std::cerr << "  Unknown quantization method: " << quant_method << std::endl;
    }

    if (cfg.enabled()) {
        std::cerr << "  Quantization: " << quant_method
                  << " bits=" << cfg.bits
                  << " group_size=" << cfg.group_size
                  << " desc_act=" << (cfg.desc_act ? "true" : "false")
                  << " sym=" << (cfg.sym ? "true" : "false") << std::endl;
    }

    return cfg;
}

// ============================================================
// Safetensors header parsing (reuse logic from model.cpp)
// ============================================================

// Forward declarations of JSON parse helpers (simplified for loader)
static bool parse_json_obj_for_tensors(
    const std::string& s, size_t& pos,
    std::map<std::string, std::string>& str_fields,
    std::map<std::string, std::vector<int64_t>>& arr_fields);

static std::string skip_ws_load(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t'))
        pos++;
    return s;
}

static bool parse_json_str_load(const std::string& s, size_t& pos, std::string& out) {
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

static bool parse_json_num_load(const std::string& s, size_t& pos, double& out) {
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

static bool parse_json_arr_int64_load(const std::string& s, size_t& pos, std::vector<int64_t>& out) {
    if (pos >= s.size() || s[pos] != '[') return false;
    pos++;
    out.clear();
    skip_ws_load(s, pos);
    if (pos < s.size() && s[pos] == ']') { pos++; return true; }
    while (pos < s.size()) {
        skip_ws_load(s, pos);
        double val;
        if (!parse_json_num_load(s, pos, val)) return false;
        out.push_back((int64_t)val);
        skip_ws_load(s, pos);
        if (pos < s.size() && s[pos] == ',') pos++;
        else if (pos < s.size() && s[pos] == ']') { pos++; return true; }
        else return false;
    }
    return true;
}

static bool parse_json_obj_for_tensors(
    const std::string& s, size_t& pos,
    std::map<std::string, std::string>& str_fields,
    std::map<std::string, std::vector<int64_t>>& arr_fields)
{
    if (pos >= s.size() || s[pos] != '{') return false;
    pos++;
    skip_ws_load(s, pos);
    if (pos < s.size() && s[pos] == '}') { pos++; return true; }
    while (pos < s.size()) {
        skip_ws_load(s, pos);
        std::string key;
        if (!parse_json_str_load(s, pos, key)) return false;
        skip_ws_load(s, pos);
        if (pos >= s.size() || s[pos] != ':') return false;
        pos++;
        skip_ws_load(s, pos);
        if (pos >= s.size()) return false;
        if (s[pos] == '"') {
            std::string val;
            if (!parse_json_str_load(s, pos, val)) return false;
            str_fields[key] = val;
        } else if (s[pos] == '[') {
            std::vector<int64_t> val;
            if (!parse_json_arr_int64_load(s, pos, val)) return false;
            arr_fields[key] = val;
        } else if (s[pos] == '{') {
            std::map<std::string, std::string> nested_str;
            std::map<std::string, std::vector<int64_t>> nested_arr;
            parse_json_obj_for_tensors(s, pos, nested_str, nested_arr);
            for (auto& [k, v] : nested_str) str_fields[key + "." + k] = v;
            for (auto& [k, v] : nested_arr) arr_fields[key + "." + k] = v;
        } else {
            double val;
            if (parse_json_num_load(s, pos, val)) {
                str_fields[key] = std::to_string((int64_t)val);
            } else if (s.substr(pos, 4) == "true") {
                str_fields[key] = "true"; pos += 4;
            } else if (s.substr(pos, 5) == "false") {
                str_fields[key] = "false"; pos += 5;
            } else if (s.substr(pos, 4) == "null") {
                str_fields[key] = "null"; pos += 4;
            } else return false;
        }
        skip_ws_load(s, pos);
        if (pos < s.size() && s[pos] == ',') pos++;
        else if (pos < s.size() && s[pos] == '}') { pos++; return true; }
        else return false;
    }
    return true;
}

// Parse a single safetensors file header, return tensor metadata
struct QuantTensorMeta {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t data_start = 0;   // byte offset of data in file
    int64_t data_end = 0;
};

static std::vector<QuantTensorMeta> parse_st_header_for_quant(const std::string& path) {
    std::vector<QuantTensorMeta> result;
    std::ifstream f(path, std::ios::binary);
    if (!f) return result;

    uint64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    if (!f || header_len > 100 * 1024 * 1024) return result;

    std::string header(header_len, '\0');
    f.read(&header[0], header_len);
    if (!f) return result;

    std::map<std::string, std::string> str_fields;
    std::map<std::string, std::vector<int64_t>> arr_fields;
    size_t pos = 0;
    skip_ws_load(header, pos);
    if (!parse_json_obj_for_tensors(header, pos, str_fields, arr_fields)) return result;

    // Find all tensor names (from shape entries)
    std::set<std::string> tensor_names;
    for (auto& [key, val] : arr_fields) {
        auto dot_pos = key.rfind('.');
        if (dot_pos != std::string::npos && key.substr(dot_pos + 1) == "shape") {
            tensor_names.insert(key.substr(0, dot_pos));
        }
    }

    int64_t header_bytes = 8 + header_len;

    for (auto& name : tensor_names) {
        QuantTensorMeta meta;
        meta.name = name;
        meta.dtype = str_fields[name + ".dtype"];
        meta.shape = arr_fields[name + ".shape"];
        auto& dof = arr_fields[name + ".data_offsets"];
        meta.data_start = ((dof.size() > 0) ? dof[0] : 0) + header_bytes;
        meta.data_end   = ((dof.size() > 1) ? dof[1] : 0) + header_bytes;
        result.push_back(meta);
    }

    return result;
}

// Read raw bytes from a safetensors file
static bool read_raw_data(const std::string& path, int64_t offset, int64_t size, void* dst) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(offset);
    f.read(reinterpret_cast<char*>(dst), size);
    return f.good();
}

// ============================================================
// Load quantized tensors from safetensors files
// ============================================================

// Find all tensor base names that have quantized weights.
// For GPTQ/AWQ, a quantized weight has: .qweight, .qzeros, .scales, and optionally .g_idx
// The base name is the weight name without these suffixes.
// E.g., "model.layers.0.self_attn.q_proj.qweight" → base = "model.layers.0.self_attn.q_proj"

static std::string strip_quant_suffix(const std::string& name) {
    const char* suffixes[] = {".qweight", ".qzeros", ".scales", ".g_idx"};
    for (const char* suffix : suffixes) {
        std::string suf(suffix);
        auto pos = name.rfind(suf);
        if (pos != std::string::npos && pos + suf.size() == name.size()) {
            return name.substr(0, pos);
        }
    }
    return "";
}

std::map<std::string, QuantizedTensor> load_quantized_tensors(
    const std::string& model_path,
    const QuantConfig& quant_config)
{
    std::map<std::string, QuantizedTensor> result;
    if (!quant_config.enabled()) return result;

    // Collect all safetensors files
    std::vector<std::string> st_files;
    for (auto& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.path().extension() == ".safetensors") {
            st_files.push_back(entry.path().string());
        }
    }
    std::sort(st_files.begin(), st_files.end());

    if (st_files.empty()) return result;

    // Phase 1: Collect all tensor metadata grouped by base name
    // base_name -> {suffix -> meta}
    struct QuantGroup {
        QuantTensorMeta qweight;
        QuantTensorMeta qzeros;
        QuantTensorMeta scales;
        QuantTensorMeta g_idx;
        std::string st_path;
    };
    std::map<std::string, QuantGroup> groups;

    for (auto& st_path : st_files) {
        auto metas = parse_st_header_for_quant(st_path);
        for (auto& meta : metas) {
            std::string base = strip_quant_suffix(meta.name);
            if (base.empty()) continue;

            auto& g = groups[base];
            g.st_path = st_path;

            if (meta.name.find(".qweight") != std::string::npos) {
                g.qweight = meta;
            } else if (meta.name.find(".qzeros") != std::string::npos) {
                g.qzeros = meta;
            } else if (meta.name.find(".scales") != std::string::npos) {
                g.scales = meta;
            } else if (meta.name.find(".g_idx") != std::string::npos) {
                g.g_idx = meta;
            }
        }
    }

    // Phase 2: Load each quant group
    for (auto& [base_name, group] : groups) {
        if (group.qweight.name.empty() || group.qzeros.name.empty() || group.scales.name.empty()) {
            std::cerr << "  Skipping incomplete quant group: " << base_name << std::endl;
            continue;
        }

        PackedInt4Tensor packed;
        packed.out_features = (int32_t)group.qweight.shape[0];
        packed.in_features = (int32_t)group.qweight.shape[1] * 2;  // each byte = 2 INT4
        packed.group_size = quant_config.group_size;
        packed.bits = quant_config.bits;
        packed.desc_act = quant_config.desc_act;
        packed.is_awq = quant_config.is_awq();

        // Load qweight
        int64_t qw_bytes = group.qweight.data_end - group.qweight.data_start;
        packed.qweight.resize(qw_bytes);
        if (!read_raw_data(group.st_path, group.qweight.data_start, qw_bytes, packed.qweight.data())) {
            std::cerr << "  Failed to read qweight for " << base_name << std::endl;
            continue;
        }

        // Load qzeros
        int64_t qz_bytes = group.qzeros.data_end - group.qzeros.data_start;
        packed.qzeros.resize(qz_bytes);
        if (!read_raw_data(group.st_path, group.qzeros.data_start, qz_bytes, packed.qzeros.data())) {
            std::cerr << "  Failed to read qzeros for " << base_name << std::endl;
            continue;
        }

        // Load scales (FP16 → float)
        int64_t sc_bytes = group.scales.data_end - group.scales.data_start;
        int64_t sc_count = sc_bytes / 2;  // FP16
        std::vector<uint16_t> scales_fp16(sc_count);
        if (!read_raw_data(group.st_path, group.scales.data_start, sc_bytes, scales_fp16.data())) {
            std::cerr << "  Failed to read scales for " << base_name << std::endl;
            continue;
        }
        packed.scales.resize(sc_count);
        for (int64_t i = 0; i < sc_count; i++) {
            // FP16 → float
            uint16_t h = scales_fp16[i];
            uint32_t sign = (h >> 15) & 0x1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t frac = h & 0x3FF;
            if (exp == 0) {
                if (frac == 0) { packed.scales[i] = sign ? -0.0f : 0.0f; }
                else {
                    frac <<= 1;
                    while ((frac & 0x400) == 0) { frac <<= 1; exp--; }
                    frac &= 0x3FF;
                    uint32_t e = (exp > 0) ? (uint32_t)exp - 9 : 0;
                    uint32_t bits = (sign << 31) | ((127 + e) << 23) | (frac << 13);
                    packed.scales[i] = *(float*)&bits;
                }
            } else if (exp == 31) {
                packed.scales[i] = sign ? -INFINITY : INFINITY;
            } else {
                uint32_t bits = (sign << 31) | ((127 - 15 + exp) << 23) | (frac << 13);
                packed.scales[i] = *(float*)&bits;
            }
        }

        // Load g_idx (optional, GPTQ act-order only)
        if (group.g_idx.name.size() > 0 && quant_config.desc_act) {
            int64_t gi_bytes = group.g_idx.data_end - group.g_idx.data_start;
            int64_t gi_count = gi_bytes / 4;  // INT32
            packed.g_idx.resize(gi_count);
            if (!read_raw_data(group.st_path, group.g_idx.data_start, gi_bytes, packed.g_idx.data())) {
                std::cerr << "  Failed to read g_idx for " << base_name << std::endl;
                packed.g_idx.clear();
            }
        }

        // Infer actual in_features from packed data size
        // qweight bytes = out_features * in_features / 2
        // in_features = qweight_bytes * 2 / out_features
        packed.in_features = (int32_t)(qw_bytes * 2 / packed.out_features);

        std::vector<int64_t> shape = { (int64_t)packed.out_features, (int64_t)packed.in_features };
        if (quant_config.is_awq()) {
            result[base_name] = QuantizedTensor::from_awq(std::move(packed), shape);
        } else {
            result[base_name] = QuantizedTensor::from_gptq(std::move(packed), shape);
        }

        std::cerr << "  Loaded quantized: " << base_name
                  << " [" << packed.out_features << " x " << packed.in_features << "]"
                  << " groups=" << packed.num_groups()
                  << " gsz=" << packed.group_size << std::endl;
    }

    std::cerr << "  Total quantized tensor groups: " << result.size() << std::endl;
    return result;
}

// ============================================================
// Dequantize all quantized tensors to FP32
// ============================================================

std::map<std::string, std::vector<float>> dequantize_all_tensors(
    const std::map<std::string, QuantizedTensor>& quant_tensors)
{
    std::map<std::string, std::vector<float>> result;
    for (auto& [name, qt] : quant_tensors) {
        result[name] = qt.to_float_vector();
        std::cerr << "  Dequantized: " << name
                  << " [" << result[name].size() << " floats]" << std::endl;
    }
    return result;
}

// ============================================================
// Quantized weight matmul: C = A @ W_quant^T
// ============================================================

void quantized_weight_matmul(
    const float* a, int32_t m, int32_t k,
    const QuantizedTensor& w_quant, int32_t n,
    float* c)
{
    // Delegate to the fused kernel
    quantized_matmul_transpose(a, m, k, w_quant, n, c);
}

// ============================================================
// FP8 activation quantization (online)
// ============================================================

FP8Activation quantize_activation_fp8(const float* src, int64_t n) {
    FP8Activation act;
    act.numel = n;
    act.data.resize(n);

    // Find max absolute value for scaling
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }

    // E4M3 max representable value: 240.0
    // Scale so that max_abs maps to 240.0
    if (max_abs > 0.0f) {
        act.scale = max_abs / 240.0f;
    } else {
        act.scale = 1.0f;
    }

    // Quantize: scale down, then convert to FP8
    float inv_scale = 1.0f / act.scale;
    for (int64_t i = 0; i < n; i++) {
        act.data[i] = float_to_fp8_e4m3(src[i] * inv_scale).bits;
    }

    return act;
}

void dequantize_activation_fp8(const FP8Activation& act, float* dst) {
    for (int64_t i = 0; i < act.numel; i++) {
        dst[i] = fp8_e4m3_to_float(fp8_e4m3(act.data[i])) * act.scale;
    }
}
