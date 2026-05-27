"""Debug MLX forward pass to find C++ bugs."""
import mlx.core as mx
import numpy as np
import json, os
from pathlib import Path
from safetensors import safe_open

MODEL_PATH = "/Users/dxp/claude/qwen-test/qwen/Qwen2___5-0___5B-Instruct"

with open(f"{MODEL_PATH}/config.json") as f:
    config = json.load(f)

hidden = config['hidden_size']
num_heads = config['num_attention_heads']
num_kv_heads = config['num_key_value_heads']
head_dim = hidden // num_heads
num_layers = config['num_hidden_layers']
ffn_dim = config['intermediate_size']
vocab_size = config['vocab_size']
rope_base = config.get('rope_theta', 1000000.0)

print(f"hidden={hidden}, heads={num_heads}/{num_kv_heads}, head_dim={head_dim}, layers={num_layers}")

# Load weights
weights = {}
for fname in sorted(Path(MODEL_PATH).glob("*.safetensors")):
    with safe_open(str(fname), framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            weights[key] = mx.array(tensor.float().numpy())
print(f"Loaded {len(weights)} tensors")

# Tokenizer
with open(f"{MODEL_PATH}/vocab.json") as f:
    vocab = json.load(f)
inv_vocab = {v: k for k, v in vocab.items()}
with open(f"{MODEL_PATH}/tokenizer_config.json") as f:
    tok_cfg = json.load(f)
added = tok_cfg.get('added_tokens_decoder', {})
for key, info in added.items():
    vocab[info['content']] = int(key)
    inv_vocab[int(key)] = info['content']

# BPE tokenizer
b2u = {}
bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
b2u = dict(zip(bs, [chr(c) for c in cs]))

with open(f"{MODEL_PATH}/merges.txt") as f:
    merges_raw = f.read().splitlines()
merges = []
for line in merges_raw:
    if line and not line.startswith('#'):
        parts = line.split()
        if len(parts) == 2:
            merges.append(tuple(parts))
merge_lookup = {m: i for i, m in enumerate(merges)}

def bpe_tokenize(text):
    chars = [b2u[b] for b in text.encode('utf-8')]
    while True:
        pairs = {}
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i+1])
            pairs[pair] = pairs.get(pair, 0) + 1
        if not pairs:
            break
        best = None
        for pair in pairs:
            if pair in merge_lookup:
                rank = merge_lookup[pair]
                if best is None or rank < merge_lookup[best]:
                    best = pair
        if best is None:
            break
        new_chars = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and chars[i] == best[0] and chars[i+1] == best[1]:
                new_chars.append(best[0] + best[1])
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        chars = new_chars
    return chars

def tokenize(text):
    return [vocab.get(p, 0) for p in bpe_tokenize(text)]

# Build prompt
system_prompt = "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
tokens = [vocab['<|endoftext|>']]
tokens.append(vocab['<|im_start|>'])
tokens.extend(tokenize(system_prompt))
tokens.append(vocab['<|im_end|>'])
tokens.extend(tokenize("\n"))
tokens.append(vocab['<|im_start|>'])
tokens.extend(tokenize("user\nHello"))
tokens.append(vocab['<|im_end|>'])
tokens.extend(tokenize("\n"))
tokens.append(vocab['<|im_start|>'])
tokens.extend(tokenize("assistant\n"))
print(f"Prompt: {len(tokens)} tokens")

# ============================================================
# Forward pass with per-group GQA (matching C++ code)
# ============================================================
def forward_per_group(input_ids):
    seq_len = len(input_ids)
    token_ids = mx.array(input_ids, dtype=mx.int32)
    h = mx.take(weights['model.embed_tokens.weight'], token_ids, axis=0)

    for l in range(num_layers):
        pfx = f"model.layers.{l}."
        normed = mx.fast.rms_norm(h, weights[pfx + "input_layernorm.weight"], 1e-6)
        residual = h

        q = normed @ weights[pfx + "self_attn.q_proj.weight"].T
        k = normed @ weights[pfx + "self_attn.k_proj.weight"].T
        v = normed @ weights[pfx + "self_attn.v_proj.weight"].T
        if pfx + "self_attn.q_proj.bias" in weights:
            q = q + weights[pfx + "self_attn.q_proj.bias"]
            k = k + weights[pfx + "self_attn.k_proj.bias"]
            v = v + weights[pfx + "self_attn.v_proj.bias"]

        q = q.reshape(1, seq_len, num_heads, head_dim)
        k = k.reshape(1, seq_len, num_kv_heads, head_dim)
        v = v.reshape(1, seq_len, num_kv_heads, head_dim)

        q = mx.fast.rope(q, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(k, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)

        scale = 1.0 / (head_dim ** 0.5)
        heads_per_group = num_heads // num_kv_heads
        attn_parts = []
        for g in range(num_kv_heads):
            qg = mx.slice(q, mx.array([0, 0, g * heads_per_group, 0]),
                         [0, 1, 2, 3], [1, seq_len, heads_per_group, head_dim])
            kg = mx.slice(k, mx.array([0, 0, g, 0]),
                         [0, 1, 2, 3], [1, seq_len, 1, head_dim])
            vg = mx.slice(v, mx.array([0, 0, g, 0]),
                         [0, 1, 2, 3], [1, seq_len, 1, head_dim])
            attn_parts.append(mx.fast.scaled_dot_product_attention(
                qg, kg, vg, scale=scale, mask="causal"))

        attn_out = mx.concatenate(attn_parts, axis=2)
        attn_out = attn_out.reshape(seq_len, num_heads * head_dim)
        attn_out = attn_out @ weights[pfx + "self_attn.o_proj.weight"].T
        h = residual + attn_out

        normed = mx.fast.rms_norm(h, weights[pfx + "post_attention_layernorm.weight"], 1e-6)
        residual = h
        gate = normed @ weights[pfx + "mlp.gate_proj.weight"].T
        up = normed @ weights[pfx + "mlp.up_proj.weight"].T
        h = residual + ((gate * mx.sigmoid(gate)) * up) @ weights[pfx + "mlp.down_proj.weight"].T

    h = mx.fast.rms_norm(h, weights["model.norm.weight"], 1e-6)
    h_last = h[-1]
    lm_head_w = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])
    return h_last @ lm_head_w.T

# ============================================================
# Forward pass with native GQA (MLX handles head broadcasting)
# ============================================================
def forward_native(input_ids):
    seq_len = len(input_ids)
    token_ids = mx.array(input_ids, dtype=mx.int32)
    h = mx.take(weights['model.embed_tokens.weight'], token_ids, axis=0)

    for l in range(num_layers):
        pfx = f"model.layers.{l}."
        normed = mx.fast.rms_norm(h, weights[pfx + "input_layernorm.weight"], 1e-6)
        residual = h

        q = normed @ weights[pfx + "self_attn.q_proj.weight"].T
        k = normed @ weights[pfx + "self_attn.k_proj.weight"].T
        v = normed @ weights[pfx + "self_attn.v_proj.weight"].T
        if pfx + "self_attn.q_proj.bias" in weights:
            q = q + weights[pfx + "self_attn.q_proj.bias"]
            k = k + weights[pfx + "self_attn.k_proj.bias"]
            v = v + weights[pfx + "self_attn.v_proj.bias"]

        q = q.reshape(1, seq_len, num_heads, head_dim)
        k = k.reshape(1, seq_len, num_kv_heads, head_dim)
        v = v.reshape(1, seq_len, num_kv_heads, head_dim)

        q = mx.fast.rope(q, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(k, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)

        scale = 1.0 / (head_dim ** 0.5)
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        attn_out = attn_out.reshape(seq_len, num_heads * head_dim)
        attn_out = attn_out @ weights[pfx + "self_attn.o_proj.weight"].T
        h = residual + attn_out

        normed = mx.fast.rms_norm(h, weights[pfx + "post_attention_layernorm.weight"], 1e-6)
        residual = h
        gate = normed @ weights[pfx + "mlp.gate_proj.weight"].T
        up = normed @ weights[pfx + "mlp.up_proj.weight"].T
        h = residual + ((gate * mx.sigmoid(gate)) * up) @ weights[pfx + "mlp.down_proj.weight"].T

    h = mx.fast.rms_norm(h, weights["model.norm.weight"], 1e-6)
    h_last = h[-1]
    lm_head_w = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])
    return h_last @ lm_head_w.T

def show_top5(logits, label):
    mx.eval(logits)
    # Get top-5 using argsort of negative values
    sorted_idx = mx.argsort(-logits)[:5]
    sorted_vals = mx.take(logits, sorted_idx)
    mx.eval(sorted_idx, sorted_vals)
    print(f"  {label}:")
    for i in range(5):
        tid = sorted_idx[i].item()
        val = sorted_vals[i].item()
        tok = inv_vocab.get(tid, '?')
        print(f"    {tid} ({tok!r}): {val:.4f}")
    return int(sorted_idx[0].item())

# ============================================================
# Run tests
# ============================================================
print("\n========== Per-Group GQA (matching C++) ==========")
test_tokens = list(tokens)  # start with original prompt

logits = forward_per_group(test_tokens)
n0 = show_top5(logits, f"Step 0 (seq={len(test_tokens)})")

test_tokens.append(n0)
logits = forward_per_group(test_tokens)
n1 = show_top5(logits, f"Step 1 (seq={len(test_tokens)})")

test_tokens.append(n1)
logits = forward_per_group(test_tokens)
n2 = show_top5(logits, f"Step 2 (seq={len(test_tokens)})")

test_tokens.append(n2)
logits = forward_per_group(test_tokens)
n3 = show_top5(logits, f"Step 3 (seq={len(test_tokens)})")

print("\n========== Native GQA ==========")
test_tokens2 = list(tokens)
logits = forward_native(test_tokens2)
m0 = show_top5(logits, f"Step 0 (seq={len(test_tokens2)})")

test_tokens2.append(m0)
logits = forward_native(test_tokens2)
m1 = show_top5(logits, f"Step 1 (seq={len(test_tokens2)})")

test_tokens2.append(m1)
logits = forward_native(test_tokens2)
m2 = show_top5(logits, f"Step 2 (seq={len(test_tokens2)})")

# Also test traditional=true
print("\n========== Per-Group GQA (traditional=True) ==========")
def forward_traditional(input_ids):
    seq_len = len(input_ids)
    token_ids = mx.array(input_ids, dtype=mx.int32)
    h = mx.take(weights['model.embed_tokens.weight'], token_ids, axis=0)

    for l in range(num_layers):
        pfx = f"model.layers.{l}."
        normed = mx.fast.rms_norm(h, weights[pfx + "input_layernorm.weight"], 1e-6)
        residual = h
        q = normed @ weights[pfx + "self_attn.q_proj.weight"].T
        k = normed @ weights[pfx + "self_attn.k_proj.weight"].T
        v = normed @ weights[pfx + "self_attn.v_proj.weight"].T
        if pfx + "self_attn.q_proj.bias" in weights:
            q = q + weights[pfx + "self_attn.q_proj.bias"]
            k = k + weights[pfx + "self_attn.k_proj.bias"]
            v = v + weights[pfx + "self_attn.v_proj.bias"]
        q = q.reshape(1, seq_len, num_heads, head_dim)
        k = k.reshape(1, seq_len, num_kv_heads, head_dim)
        v = v.reshape(1, seq_len, num_kv_heads, head_dim)
        q = mx.fast.rope(q, head_dim, traditional=True, base=rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(k, head_dim, traditional=True, base=rope_base, scale=1.0, offset=0)
        scale = 1.0 / (head_dim ** 0.5)
        heads_per_group = num_heads // num_kv_heads
        attn_parts = []
        for g in range(num_kv_heads):
            qg = mx.slice(q, mx.array([0, 0, g * heads_per_group, 0]),
                         [0, 1, 2, 3], [1, seq_len, heads_per_group, head_dim])
            kg = mx.slice(k, mx.array([0, 0, g, 0]),
                         [0, 1, 2, 3], [1, seq_len, 1, head_dim])
            vg = mx.slice(v, mx.array([0, 0, g, 0]),
                         [0, 1, 2, 3], [1, seq_len, 1, head_dim])
            attn_parts.append(mx.fast.scaled_dot_product_attention(
                qg, kg, vg, scale=scale, mask="causal"))
        attn_out = mx.concatenate(attn_parts, axis=2)
        attn_out = attn_out.reshape(seq_len, num_heads * head_dim)
        attn_out = attn_out @ weights[pfx + "self_attn.o_proj.weight"].T
        h = residual + attn_out
        normed = mx.fast.rms_norm(h, weights[pfx + "post_attention_layernorm.weight"], 1e-6)
        residual = h
        gate = normed @ weights[pfx + "mlp.gate_proj.weight"].T
        up = normed @ weights[pfx + "mlp.up_proj.weight"].T
        h = residual + ((gate * mx.sigmoid(gate)) * up) @ weights[pfx + "mlp.down_proj.weight"].T

    h = mx.fast.rms_norm(h, weights["model.norm.weight"], 1e-6)
    h_last = h[-1]
    lm_head_w = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])
    return h_last @ lm_head_w.T

test_tokens3 = list(tokens)
logits = forward_traditional(test_tokens3)
t0 = show_top5(logits, f"Step 0 (seq={len(test_tokens3)})")
test_tokens3.append(t0)
logits = forward_traditional(test_tokens3)
t1 = show_top5(logits, f"Step 1 (seq={len(test_tokens3)})")
test_tokens3.append(t1)
logits = forward_traditional(test_tokens3)
t2 = show_top5(logits, f"Step 2 (seq={len(test_tokens3)})")
