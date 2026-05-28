#include "tp_model.h"
#include "tp.h"
#include "utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <fstream>
#include <filesystem>
#include <map>
#include <algorithm>
#include <vector>
#include <cmath>

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
        if (s[pos] == '\\' && pos + 1 < s.size()) { pos++; }
        out += s[pos];
        pos++;
    }
    if (pos < s.size()) pos++;
    return true;
}

static bool parse_json_number(const std::string& s, size_t& pos, double& out) {
    if (pos >= s.size()) return false;
    std::string num;
    if (pos < s.size() && s[pos] == '-') { num += s[pos++]; }
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') num += s[pos++];
    if (pos < s.size() && s[pos] == '.') {
        num += s[pos++];
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') num += s[pos++];
    }
    if (num.empty()) return false;
    out = std::stod(num);
    return true;
}

static bool parse_json_array_i64(const std::string& s, size_t& pos, std::vector<int64_t>& out) {
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
            if (!parse_json_array_i64(s, pos, val)) return false;
            arr_fields[key] = val;
        } else if (s[pos] == '{') {
            std::map<std::string, std::string> nstr;
            std::map<std::string, std::vector<int64_t>> narr;
            parse_json_object(s, pos, nstr, narr);
            for (auto& [k, v] : nstr) str_fields[key + "." + k] = v;
            for (auto& [k, v] : narr) arr_fields[key + "." + k] = v;
        } else {
            double val;
            if (parse_json_number(s, pos, val))
                str_fields[key] = std::to_string((int64_t)val);
            else if (s.substr(pos, 4) == "true") { str_fields[key] = "true"; pos += 4; }
            else if (s.substr(pos, 5) == "false") { str_fields[key] = "false"; pos += 5; }
            else if (s.substr(pos, 4) == "null") { str_fields[key] = "null"; pos += 4; }
            else return false;
        }
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') pos++;
        else if (pos < s.size() && s[pos] == '}') { pos++; return true; }
        else return false;
    }
    return true;
}

// ============================================================
// Tensor metadata
// ============================================================

struct TensorMetaST {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t data_offsets[2];
};

static std::vector<TensorMetaST> parse_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    uint64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    if (!f || header_len > 100 * 1024 * 1024) return {};
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);
    if (!f) return {};

    std::map<std::string, std::string> sf;
    std::map<std::string, std::vector<int64_t>> af;
    size_t pos = 0;
    skip_ws(header, pos);
    if (!parse_json_object(header, pos, sf, af)) return {};

    std::vector<std::string> names;
    for (auto& [key, val] : af) {
        auto dot = key.rfind('.');
        if (dot != std::string::npos) {
            std::string suf = key.substr(dot + 1);
            std::string base = key.substr(0, dot);
            if ((suf == "shape" || suf == "data_offsets") &&
                std::find(names.begin(), names.end(), base) == names.end()) {
                names.push_back(base);
            }
        }
    }
    std::vector<TensorMetaST> tensors;
    for (auto& name : names) {
        TensorMetaST m;
        m.name = name;
        m.dtype = sf[name + ".dtype"];
        m.shape = af[name + ".shape"];
        auto dof = af[name + ".data_offsets"];
        m.data_offsets[0] = (dof.size() > 0 ? dof[0] : 0) + (int64_t)(8 + header_len);
        m.data_offsets[1] = (dof.size() > 1 ? dof[1] : 0) + (int64_t)(8 + header_len);
        tensors.push_back(m);
    }
    return tensors;
}

// ============================================================
// Safetensors tensor slice loader
// Reads a tensor from file, optionally extracts TP slice, uploads as half to GPU
// ============================================================

static bool load_slice(
    const std::string& file_path, const TensorMetaST& meta,
    int32_t dim_to_split,  // -1=no split, 0=split rows, 1=split cols
    half** gpu_ptr, int64_t* out_rows, int64_t* out_cols)
{
    int64_t numel = 1;
    for (auto d : meta.shape) numel *= d;
    int64_t rows = meta.shape.size() > 0 ? meta.shape[0] : 1;
    int64_t cols = meta.shape.size() > 1 ? meta.shape[1] : 1;

    TPConfig* tp = tp_get();
    int32_t tp_size = tp->tp_size;
    int32_t tp_rank = tp->tp_rank;

    int64_t lr = rows, lc = cols;
    int64_t ro = 0, co = 0;

    if (dim_to_split >= 0 && tp_size > 1) {
        int32_t fd = (int32_t)(dim_to_split == 0 ? rows : cols);
        int32_t lo, ls;
        tp_split_dim(fd, &lo, &ls);
        if (dim_to_split == 0) { lr = ls; ro = lo; }
        else { lc = ls; co = lo; }
    }

    int64_t lnum = lr * lc;

    std::ifstream f(file_path, std::ios::binary);
    if (!f) return false;
    f.seekg(meta.data_offsets[0]);

    half* dptr;
    CUDA_CHECK(cudaMalloc(&dptr, lnum * sizeof(half)));

    if (meta.dtype == "F16") {
        std::vector<uint16_t> raw(numel);
        f.read(reinterpret_cast<char*>(raw.data()), numel * sizeof(uint16_t));
        std::vector<half> slice(lnum);
        if (dim_to_split < 0 || tp_size <= 1) {
            for (int64_t i = 0; i < numel; i++) slice[i] = (half)(raw[i]);
        } else if (dim_to_split == 0) {
            for (int64_t r = 0; r < lr; r++)
                for (int64_t c = 0; c < cols; c++)
                    slice[r * cols + c] = (half)(raw[(ro + r) * cols + c]);
        } else {
            for (int64_t r = 0; r < rows; r++)
                for (int64_t c = 0; c < lc; c++)
                    slice[r * lc + c] = (half)(raw[r * cols + co + c]);
        }
        CUDA_CHECK(cudaMemcpy(dptr, slice.data(), lnum * sizeof(half), cudaMemcpyHostToDevice));
    } else if (meta.dtype == "F32") {
        std::vector<float> raw(numel);
        f.read(reinterpret_cast<char*>(raw.data()), numel * sizeof(float));
        std::vector<half> slice(lnum);
        if (dim_to_split < 0 || tp_size <= 1) {
            for (int64_t i = 0; i < numel; i++) slice[i] = __float2half(raw[i]);
        } else if (dim_to_split == 0) {
            for (int64_t r = 0; r < lr; r++)
                for (int64_t c = 0; c < cols; c++)
                    slice[r * cols + c] = __float2half(raw[(ro + r) * cols + c]);
        } else {
            for (int64_t r = 0; r < rows; r++)
                for (int64_t c = 0; c < lc; c++)
                    slice[r * lc + c] = __float2half(raw[r * cols + co + c]);
        }
        CUDA_CHECK(cudaMemcpy(dptr, slice.data(), lnum * sizeof(half), cudaMemcpyHostToDevice));
    } else {
        fprintf(stderr, "Unsupported dtype %s\n", meta.dtype.c_str());
        cudaFree(dptr);
        return false;
    }

    *gpu_ptr = dptr;
    *out_rows = lr;
    *out_cols = lc;
    return true;
}

// ============================================================
// cuBLAS GEMM helpers
// ============================================================

// C[m,n] = A[m,k] @ B[n,k]^T  (all half precision inputs, float output)
static void gemm_hh(cublasHandle_t h, const half* A, const half* B,
                     float* C, int32_t m, int32_t n, int32_t k) {
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
                 n, m, k, &alpha,
                 B, CUDA_R_16F, k,
                 A, CUDA_R_16F, k,
                 &beta,
                 C, CUDA_R_32F, n,
                 CUDA_R_32F, n);
}

// ============================================================
// CUDA kernels
// ============================================================

__global__ void float2half_kernel(half* out, const float* in, int32_t n) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

__global__ void rms_norm_multi_kernel(
    float* out, const float* input, const half* weight,
    int32_t dim, float eps, int32_t batch)
{
    int32_t b = blockIdx.x;
    if (b >= batch) return;
    const float* inp = input + b * dim;
    float* o = out + b * dim;

    __shared__ float sh_sum[256];
    float sum_sq = 0.0f;
    for (int32_t i = threadIdx.x; i < dim; i += blockDim.x)
        sum_sq += inp[i] * inp[i];
    sh_sum[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh_sum[threadIdx.x] += sh_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float inv_rms = rsqrtf(sh_sum[0] / dim + eps);
        for (int32_t i = 0; i < dim; i++)
            o[i] = inp[i] * __half2float(weight[i]) * inv_rms;
    }
}

__global__ void residual_kernel(float* out, const float* a, const float* b, int32_t n) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void swiglu_kernel(float* out, const float* gate, const float* up, int32_t n) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
}

__global__ void embed_lookup_kernel(float* out, const half* table, const int32_t* ids,
                                     int32_t hidden, int32_t n) {
    int32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int32_t id = ids[t];
    float* o = out + t * hidden;
    const half* row = table + id * hidden;
    for (int32_t d = threadIdx.y; d < hidden; d += blockDim.y)
        o[d] = __half2float(row[d]);
}

// ============================================================
// TP Model loading
// ============================================================

TPModel* tp_load_model(const std::string& model_path)
{
    TPConfig* tp = tp_get();
    TPModel* m = new TPModel();
    m->model_path = model_path;

    // Collect safetensors files
    std::vector<std::string> st_files;
    for (auto& entry : std::filesystem::directory_iterator(model_path))
        if (entry.path().extension() == ".safetensors")
            st_files.push_back(entry.path().string());
    if (st_files.empty()) {
        fprintf(stderr, "No .safetensors files in %s\n", model_path.c_str());
        delete m; return nullptr;
    }
    std::sort(st_files.begin(), st_files.end());

    // Build tensor index: name -> {file, meta}
    struct Entry { std::string file; TensorMetaST meta; };
    std::map<std::string, Entry> idx;
    for (auto& fp : st_files) {
        auto metas = parse_header(fp);
        for (auto& meta : metas) idx[meta.name] = {fp, meta};
    }

    // Lambda: find any of the named tensors in the index
    auto find_entry = [&](const std::vector<std::string>& names) -> Entry* {
        for (auto& name : names) {
            auto it = idx.find(name);
            if (it != idx.end()) return &it->second;
        }
        return nullptr;
    };

    // Lambda: load full (no TP split)
    auto load_full = [&](const std::vector<std::string>& names, half** ptr,
                         int64_t* r, int64_t* c) -> bool {
        auto* e = find_entry(names);
        if (!e) return false;
        return load_slice(e->file, e->meta, -1, ptr, r, c);
    };

    // Lambda: load with TP split
    auto load_split = [&](const std::string& name, int32_t dim, half** ptr,
                          int64_t* r, int64_t* c) -> bool {
        auto it = idx.find(name);
        if (it == idx.end()) return false;
        return load_slice(it->second.file, it->second.meta, dim, ptr, r, c);
    };

    // ---- Read config.json ----
    std::filesystem::path cfg_path = std::filesystem::path(model_path) / "config.json";
    if (std::filesystem::exists(cfg_path)) {
        std::ifstream f(cfg_path);
        std::string cfg((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        auto gi = [&](const std::string& k) -> int32_t {
            auto p = cfg.find(k);
            if (p == std::string::npos) return 0;
            p += k.size();
            while (p < cfg.size() && (cfg[p]=='"'||cfg[p]==':'||cfg[p]==' '||cfg[p]=='\n')) p++;
            int32_t v = 0;
            while (p < cfg.size() && cfg[p]>='0' && cfg[p]<='9') v = v*10 + (cfg[p++]-'0');
            return v;
        };
        auto gf = [&](const std::string& k) -> float {
            auto p = cfg.find(k);
            if (p == std::string::npos) return 0.0f;
            p += k.size();
            while (p < cfg.size() && (cfg[p]=='"'||cfg[p]==':'||cfg[p]==' '||cfg[p]=='\n')) p++;
            std::string num;
            while (p < cfg.size() && ((cfg[p]>='0'&&cfg[p]<='9')||cfg[p]=='.'||cfg[p]=='-')) num += cfg[p++];
            return num.empty() ? 0.0f : std::stof(num);
        };
        m->num_layers   = gi("num_hidden_layers");
        m->num_heads    = gi("num_attention_heads");
        m->num_kv_heads = gi("num_key_value_heads");
        m->hidden_size  = gi("hidden_size");
        m->vocab_size   = gi("vocab_size");
        m->norm_eps     = gf("rms_norm_eps");
        m->rope_theta   = gf("rope_theta");
        int32_t hd = gi("head_dim");
        if (hd > 0) m->head_dim = hd;
    }

    // ---- Load embeddings ----
    int64_t er, ec;
    if (load_full({"embed_tokens.weight", "model.embed_tokens.weight"}, &m->embeddings, &er, &ec)) {
        if (!m->vocab_size)  m->vocab_size  = (int32_t)er;
        if (!m->hidden_size) m->hidden_size = (int32_t)ec;
    }

    // ---- Infer dims ----
    if (!m->head_dim && m->num_heads > 0) m->head_dim = m->hidden_size / m->num_heads;
    if (!m->head_dim) m->head_dim = 128;

    // ---- Detect MoE ----
    for (auto& [name, _] : idx)
        if (name.find("mlp.experts.") != std::string::npos) { m->is_moe = true; break; }

    // ---- Count layers ----
    int32_t max_layer = -1;
    for (auto& [name, _] : idx) {
        auto pos = name.find("model.layers.");
        if (pos != std::string::npos) {
            auto end = name.find('.', pos + 13);
            if (end != std::string::npos)
                max_layer = std::max(max_layer, std::stoi(name.substr(pos+13, end-pos-13)));
        }
    }
    if (max_layer >= 0) m->num_layers = max_layer + 1;

    // ---- Compute TP-local dims ----
    if (tp->tp_size > 1) {
        int32_t lo, ls;
        tp_split_heads(m->num_heads, &lo, &ls);    m->local_num_heads = ls;
        if (m->num_kv_heads > 0) {
            tp_split_heads(m->num_kv_heads, &lo, &ls); m->local_num_kv_heads = ls;
        } else {
            m->local_num_kv_heads = m->local_num_heads;
            m->num_kv_heads = m->num_heads;
        }
    } else {
        m->local_num_heads = m->num_heads;
        m->local_num_kv_heads = m->num_kv_heads > 0 ? m->num_kv_heads : m->num_heads;
        if (m->num_kv_heads <= 0) m->num_kv_heads = m->num_heads;
    }
    m->local_q_out = m->local_num_heads * m->head_dim;
    m->local_k_out = m->local_num_kv_heads * m->head_dim;
    m->local_v_out = m->local_num_kv_heads * m->head_dim;
    m->local_o_in  = m->local_num_heads * m->head_dim;

    fprintf(stderr, "TP[%d/%d]: %dL hidden=%d heads=%d(l%d) kv=%d(l%d) hd=%d moe=%d\n",
            tp->tp_rank, tp->tp_size, m->num_layers, m->hidden_size,
            m->num_heads, m->local_num_heads,
            m->num_kv_heads, m->local_num_kv_heads,
            m->head_dim, m->is_moe);

    // ---- Load layers ----
    m->layers.resize(m->num_layers);
    for (int32_t l = 0; l < m->num_layers; l++) {
        auto pfx = "model.layers." + std::to_string(l) + ".";
        auto& L = m->layers[l];
        int64_t dr, dc;

        // Norms (shared)
        load_full({pfx + "input_layernorm.weight"}, &L.input_norm_w, &dr, &dc);
        load_full({pfx + "post_attention_layernorm.weight"}, &L.attn_norm_w, &dr, &dc);

        // Q/K/V (column-parallel dim 0)
        load_split(pfx + "self_attn.q_proj.weight", 0, &L.q_proj_w, &dr, &dc);
        load_full({pfx + "self_attn.q_proj.bias"}, &L.q_proj_bias, &dr, &dc);
        load_split(pfx + "self_attn.k_proj.weight", 0, &L.k_proj_w, &dr, &dc);
        load_full({pfx + "self_attn.k_proj.bias"}, &L.k_proj_bias, &dr, &dc);
        load_split(pfx + "self_attn.v_proj.weight", 0, &L.v_proj_w, &dr, &dc);
        load_full({pfx + "self_attn.v_proj.bias"}, &L.v_proj_bias, &dr, &dc);

        // O (row-parallel dim 1)
        load_split(pfx + "self_attn.o_proj.weight", 1, &L.o_proj_w, &dr, &dc);

        // Q/K norms (shared)
        load_full({pfx + "self_attn.q_norm.weight"}, &L.q_norm_w, &dr, &dc);
        load_full({pfx + "self_attn.k_norm.weight"}, &L.k_norm_w, &dr, &dc);

        // MLP
        if (m->is_moe) {
            load_full({pfx + "mlp.gate.weight"}, &L.moe_gate_w, &dr, &dc);
            // Count experts
            int32_t ne = 0;
            for (auto& [nm, _] : idx) {
                auto probe = pfx + "mlp.experts.";
                auto ep = nm.find(probe);
                if (ep != std::string::npos) {
                    auto ee = nm.find('.', ep + probe.size());
                    if (ep + probe.size() <= ee)
                        ne = std::max(ne, 1 + std::stoi(nm.substr(ep + probe.size(), ee - ep - probe.size())));
                }
            }
            L.num_experts = ne;
            if (ne > 0) {
                L.experts_gate_w = new half*[ne];
                L.experts_up_w   = new half*[ne];
                L.experts_down_w = new half*[ne];
                for (int32_t e = 0; e < ne; e++) {
                    auto ep = pfx + "mlp.experts." + std::to_string(e) + ".";
                    int64_t gr, gc;
                    load_split(ep + "gate_proj.weight", 0, &L.experts_gate_w[e], &gr, &gc);
                    if (gr > 0 && m->ffn_intermediate <= 0) {
                        m->ffn_intermediate = (int32_t)gr * tp->tp_size;
                        int32_t lo, ls;
                        tp_split_dim(m->ffn_intermediate, &lo, &ls);
                        m->local_ffn = ls;
                    }
                    load_split(ep + "up_proj.weight",   0, &L.experts_up_w[e], &dr, &dc);
                    load_split(ep + "down_proj.weight", 1, &L.experts_down_w[e], &dr, &dc);
                }
            }
        } else {
            int64_t gr;
            load_split(pfx + "mlp.gate_proj.weight", 0, &L.gate_proj_w, &gr, &dc);
            if (gr > 0) {
                m->ffn_intermediate = (int32_t)gr * tp->tp_size;
                int32_t lo, ls;
                tp_split_dim(m->ffn_intermediate, &lo, &ls);
                m->local_ffn = ls;
            }
            load_split(pfx + "mlp.up_proj.weight",   0, &L.up_proj_w, &dr, &dc);
            load_split(pfx + "mlp.down_proj.weight", 1, &L.down_proj_w, &dr, &dc);
        }

        if (l == 0) {
            fprintf(stderr, "TP[%d] L0: q=%p k=%p v=%p o=%p moe=%d ne=%d ffn=%d(l%d)\n",
                    tp->tp_rank, (void*)L.q_proj_w, (void*)L.k_proj_w,
                    (void*)L.v_proj_w, (void*)L.o_proj_w,
                    m->is_moe, L.num_experts,
                    m->ffn_intermediate, m->local_ffn);
        }
    }

    // ---- Final norm ----
    load_full({"norm.weight", "model.norm.weight"}, &m->final_norm_w, &er, &ec);

    // ---- LM head ----
    if (!load_full({"lm_head.weight"}, &m->lm_head, &er, &ec))
        m->lm_head = m->embeddings;

    // ---- RoPE ----
    int64_t rp_r, rp_c;
    // RoPE freq tensor is float, not half. Handle separately.
    {
        auto* e = find_entry({"rope_freqs.inv_freq", "model.rope_freqs_tensor"});
        if (e) {
            int64_t rn = 1;
            for (auto d : e->meta.shape) rn *= d;
            m->rope_inv_freq_size = (int32_t)rn;
            std::ifstream f(e->file, std::ios::binary);
            CUDA_CHECK(cudaMalloc(&m->rope_inv_freq, rn * sizeof(float)));
            if (f && e->meta.dtype == "F32") {
                std::vector<float> buf(rn);
                f.seekg(e->meta.data_offsets[0]);
                f.read(reinterpret_cast<char*>(buf.data()), rn * sizeof(float));
                CUDA_CHECK(cudaMemcpy(m->rope_inv_freq, buf.data(), rn * sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }
    if (!m->rope_inv_freq) {
        int32_t rd = m->head_dim / 2;
        m->rope_inv_freq_size = rd;
        std::vector<float> hf(rd);
        for (int32_t i = 0; i < rd; i++)
            hf[i] = 1.0f / std::pow(m->rope_theta, (2.0f * i) / m->head_dim);
        CUDA_CHECK(cudaMalloc(&m->rope_inv_freq, rd * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(m->rope_inv_freq, hf.data(), rd * sizeof(float), cudaMemcpyHostToDevice));
    }

    fprintf(stderr, "TP[%d]: Model loaded successfully.\n", tp->tp_rank);
    return m;
}

void tp_free_model(TPModel* m) {
    if (!m) return;
    auto fh = [](half* p) { if (p) cudaFree(p); };
    auto ff = [](float* p) { if (p) cudaFree(p); };

    fh(m->embeddings); fh(m->lm_head); fh(m->final_norm_w);
    ff(m->rope_inv_freq);

    for (auto& L : m->layers) {
        fh(L.input_norm_w); fh(L.attn_norm_w);
        fh(L.q_proj_w); fh(L.k_proj_w); fh(L.v_proj_w); fh(L.o_proj_w);
        fh(L.q_proj_bias); fh(L.k_proj_bias); fh(L.v_proj_bias);
        fh(L.q_norm_w); fh(L.k_norm_w);
        fh(L.gate_proj_w); fh(L.up_proj_w); fh(L.down_proj_w);
        fh(L.moe_gate_w); fh(L.moe_score_corr);
        if (L.experts_gate_w) {
            for (int32_t e = 0; e < L.num_experts; e++) {
                fh(L.experts_gate_w[e]); fh(L.experts_up_w[e]); fh(L.experts_down_w[e]);
            }
            delete[] L.experts_gate_w;
            delete[] L.experts_up_w;
            delete[] L.experts_down_w;
        }
    }

    if (m->k_cache) {
        for (int32_t l = 0; l < m->num_layers; l++) { fh(m->k_cache[l]); fh(m->v_cache[l]); }
        delete[] m->k_cache; delete[] m->v_cache;
    }
    delete[] m->seq_lens;
    delete m;
}

// ============================================================
// TP-aware forward pass
// ============================================================

void tp_forward_pass(
    TPModel* model,
    const int32_t* input_ids,
    float* logits,
    int32_t batch_size,
    int32_t seq_len,
    cudaStream_t stream)
{
    TPConfig* tp = tp_get();
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);

    int32_t H = model->hidden_size;
    int32_t L = model->num_layers;
    int32_t V = model->vocab_size;
    int32_t N = batch_size * seq_len;

    // Allocate float buffers
    float *hs, *normed, *attn_out, *mlp_out;
    float *q_buf, *k_buf, *v_buf;
    float *gate_buf, *up_buf, *swiglu_buf;
    float *ag_buf = nullptr;

    CUDA_CHECK(cudaMalloc(&hs,       N * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&normed,   N * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attn_out, N * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mlp_out,  N * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q_buf,    N * model->local_q_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&k_buf,    N * model->local_k_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v_buf,    N * model->local_v_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gate_buf,  N * model->local_ffn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&up_buf,    N * model->local_ffn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&swiglu_buf,N * model->local_ffn * sizeof(float)));

    if (tp->enabled())
        CUDA_CHECK(cudaMalloc(&ag_buf, N * model->num_heads * model->head_dim * sizeof(float)));

    // Half-precision GEMM input buffers (converted from float before each GEMM)
    half *normed_h, *attn_h, *swiglu_h;
    CUDA_CHECK(cudaMalloc(&normed_h,  N * H * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&attn_h,    N * H * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&swiglu_h,  N * model->local_ffn * sizeof(half)));

    // ---- Embedding lookup (GPU kernel) ----
    {
        dim3 block(256, 8);
        dim3 grid((N + block.x - 1) / block.x);
        embed_lookup_kernel<<<grid, block, 0, stream>>>(hs, model->embeddings, input_ids, H, N);
    }

    // ---- Process layers ----
    for (int32_t l = 0; l < L; l++) {
        const auto& ly = model->layers[l];

        // --- Input RMSNorm ---
        rms_norm_multi_kernel<<<N, 256, 0, stream>>>(normed, hs, ly.input_norm_w, H, model->norm_eps, N);

        // Convert normed float -> half for GEMM
        float2half_kernel<<<(N*H + 255)/256, 256, 0, stream>>>(normed_h, normed, N * H);

        // --- Q projection (column-parallel dim 0) ---
        gemm_hh(cublas_handle, ly.q_proj_w, normed_h, q_buf, model->local_q_out, N, H);
        // --- K projection (column-parallel dim 0) ---
        gemm_hh(cublas_handle, ly.k_proj_w, normed_h, k_buf, model->local_k_out, N, H);
        // --- V projection (column-parallel dim 0) ---
        gemm_hh(cublas_handle, ly.v_proj_w, normed_h, v_buf, model->local_v_out, N, H);

        // --- All-gather Q, K, V (when TP > 1) ---
        if (tp->enabled()) {
            int32_t full_q = N * model->num_heads * model->head_dim;
            int32_t full_kv = N * model->num_kv_heads * model->head_dim;
            tp_allgather(ag_buf, q_buf, N * model->local_q_out, stream);
            CUDA_CHECK(cudaMemcpyAsync(q_buf, ag_buf, full_q * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            tp_allgather(ag_buf, k_buf, N * model->local_k_out, stream);
            CUDA_CHECK(cudaMemcpyAsync(k_buf, ag_buf, full_kv * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            tp_allgather(ag_buf, v_buf, N * model->local_v_out, stream);
            CUDA_CHECK(cudaMemcpyAsync(v_buf, ag_buf, full_kv * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }

        // --- Attention: copy 0 to attn_out as placeholder ---
        CUDA_CHECK(cudaMemsetAsync(attn_out, 0, N * H * sizeof(float), stream));
        // TODO: real attention with RoPE + GQA + flash/paged attention

        // Convert attn_out -> half for O_proj GEMM
        float2half_kernel<<<(N*H + 255)/256, 256, 0, stream>>>(attn_h, attn_out, N * H);

        // --- O projection (row-parallel dim 1, needs all-reduce) ---
        gemm_hh(cublas_handle, ly.o_proj_w, attn_h, attn_out, H, N, model->local_o_in);
        if (tp->enabled()) tp_allreduce(attn_out, N * H, stream);

        // --- Residual ---
        residual_kernel<<<(N*H + 255)/256, 256, 0, stream>>>(hs, hs, attn_out, N * H);

        // --- Post-attention norm ---
        rms_norm_multi_kernel<<<N, 256, 0, stream>>>(normed, hs, ly.attn_norm_w, H, model->norm_eps, N);
        float2half_kernel<<<(N*H + 255)/256, 256, 0, stream>>>(normed_h, normed, N * H);

        // --- MLP ---
        if (model->is_moe && ly.experts_gate_w) {
            // MoE placeholder
            CUDA_CHECK(cudaMemsetAsync(mlp_out, 0, N * H * sizeof(float), stream));
            // TODO: router + expert dispatch + expert GEMMs + combine
        } else {
            // Gate + Up (column-parallel dim 0)
            gemm_hh(cublas_handle, ly.gate_proj_w, normed_h, gate_buf, model->local_ffn, N, H);
            gemm_hh(cublas_handle, ly.up_proj_w,   normed_h, up_buf,   model->local_ffn, N, H);

            // SwiGLU
            swiglu_kernel<<<(N * model->local_ffn + 255)/256, 256, 0, stream>>>(
                swiglu_buf, gate_buf, up_buf, N * model->local_ffn);

            // Convert swiglu -> half for down projection
            float2half_kernel<<<(N * model->local_ffn + 255)/256, 256, 0, stream>>>(
                swiglu_h, swiglu_buf, N * model->local_ffn);

            // Down projection (row-parallel dim 1, needs all-reduce)
            gemm_hh(cublas_handle, ly.down_proj_w, swiglu_h, mlp_out, H, N, model->local_ffn);
            if (tp->enabled()) tp_allreduce(mlp_out, N * H, stream);
        }

        // --- Residual ---
        residual_kernel<<<(N*H + 255)/256, 256, 0, stream>>>(hs, hs, mlp_out, N * H);
    }

    // ---- Final norm ----
    rms_norm_multi_kernel<<<N, 256, 0, stream>>>(normed, hs, model->final_norm_w, H, model->norm_eps, N);

    // ---- LM head: only last token per sequence ----
    {
        // normed[last_seq_pos, :] as half
        int32_t last = seq_len - 1;
        half* last_h;
        CUDA_CHECK(cudaMalloc(&last_h, batch_size * H * sizeof(half)));
        float2half_kernel<<<(batch_size * H + 255)/256, 256, 0, stream>>>(last_h, normed + last * H, batch_size * H);

        half* lm_w = model->lm_head ? model->lm_head : model->embeddings;
        gemm_hh(cublas_handle, lm_w, last_h, logits, V, batch_size, H);

        CUDA_CHECK(cudaFree(last_h));
    }

    // Cleanup
    cublasDestroy(cublas_handle);
    cudaFree(hs); cudaFree(normed); cudaFree(attn_out); cudaFree(mlp_out);
    cudaFree(q_buf); cudaFree(k_buf); cudaFree(v_buf);
    cudaFree(gate_buf); cudaFree(up_buf); cudaFree(swiglu_buf);
    cudaFree(normed_h); cudaFree(attn_h); cudaFree(swiglu_h);
    if (ag_buf) cudaFree(ag_buf);
}

// ============================================================
// Stub implementations (to be filled with real attention/MoE)
// ============================================================

void tp_attention_forward(
    const half*, const half*, const half*, const half*,
    const half*, const half*, const half*, const half*, const half*,
    const float*, int32_t, const half*, const half*,
    float*, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t, const float*, float*, float*, cudaStream_t) {}

void tp_dense_mlp_forward(
    const half*, const half*, const half*, float*, int32_t, int32_t,
    int32_t, const float*, float*, cudaStream_t) {}

void tp_moe_forward(
    const half*, const half**, const half**, const half**,
    float*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
    const float*, float*, cudaStream_t) {}
