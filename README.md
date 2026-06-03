# vllm-cpp

Minimal C++ inference engine for Qwen3.6 series models (MoE + Dense variants).

## Architecture

- **Qwen3.6-35B-A3B**: 35B total params, ~3.5B active per forward pass
  - 64 transformer layers, hidden_size=5120, 64 heads, head_dim=80
  - GQA: 8 KV head groups
  - 36 experts, top-8 routing, SwiGLU FFN (intermediate=12288)
  - RMSNorm + RoPE (base=1e6) + Yarn scaling

- **Qwen3.6-8B**: 8B dense, 36 layers, same head config

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

CUDA is auto-detected. Without CUDA, only CPU runner is built.

For explicit CUDA path:
```bash
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

## Usage

### CPU Runner  

```bash
./cpu_runner /path/to/Qwen3-8B "Hello world"
./cpu_runner /path/to/Qwen3-32B --tokens "151643,1234,5678" --max-new 50 --temp 0.7
```

### HTTP Server

```bash
./build/http_server /Users/dxp/claude/qwen-test/qwen/Qwen2___5-0___5B-Instruct --port 8080
```

Supports OpenAI-compatible endpoints:
- `POST /v1/chat/completions` - Chat completion (streaming supported)
- `POST /v1/completions` - Legacy completion
- `GET /v1/models` - List available models
- `GET /health` - Health check

### Example API Call

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2___5-0___5B-Instruct",
    "messages": [{"role": "user", "content": "What is C++? , explain func with 100 words"}],
    "temperature": 0.7,
    "max_tokens": 20
  }'
```

Streaming:
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2___5-0___5B-Instruct","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## Model Loading

### safetensors format

Place model files in a directory:
```
model/
├── model-00001-of-00008.safetensors
├── model-00002-of-00008.safetensors
├── ...
├── vocab.json
├── merges.txt
└── config.json (optional)
```

The loader auto-detects architecture from tensor shapes.

### Supported dtypes

- F32 (float32) - direct load
- F16 (float16) - converted to FP32 on CPU
- BF16 (bfloat16) - converted to FP32 on CPU

## Performance Notes

| Mode | Speed | Notes |
|------|-------|-------|
| CPU (naive) | ~5-10 ms/token | Baseline, single-threaded matmul |
| CPU (OpenBLAS) | ~2-5 ms/token | Link with -lopenblas |
| CUDA (cuBLAS) | ~1-2 ms/token | Requires GPU with enough VRAM |

The 35B-A3B MoE model requires ~70GB RAM for FP32 weights.
The 8B dense model requires ~32GB RAM for FP32 weights.

## Project Structure

```
vllm-cpp/
├── CMakeLists.txt          - Build configuration
├── src/
│   ├── config.h            - Qwen3.6 architecture constants
│   ├── model.h             - Model structures + forward API
│   ├── model.cpp           - safetensors loader + CPU forward pass
│   ├── cpu_runner.cpp      - CLI inference tool
│   └── http_server.cpp     - Raw socket HTTP server
├── cuda/
│   ├── attention.cu        - PagedAttention host launcher
│   ├── attention.h         - Attention kernel declarations
│   ├── attention_kernels.cuh - CUDA kernel implementations
│   ├── moe_dispatch.cu     - MoE dispatch/compute/combine
│   ├── moe_dispatch.h      - MoE kernel declarations
│   ├── moe_kernels.cuh     - MoE kernel implementations
│   └── utils.cuh           - FP16 ops, vectorized load/store
```

## Design Principles

- **No abstraction layers**: Direct Qwen3.6 implementation, no virtual functions
- **No external dependencies**: Raw sockets for HTTP, manual JSON parsing
- **No plugin system**: Hardcoded for Qwen3.6 architecture only
- **CPU-first**: Validate on CPU before CUDA optimization
- **Simple over clever**: Functions < 50 lines, straightforward code

## License

MIT
