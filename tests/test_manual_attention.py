"""Full forward pass comparing manual attention vs SDPA per-group GQA."""
import mlx.core as mx
import numpy as np
import json
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
rope_base = config.get('rope_theta', 1000000.0)

weights = {}
for fname in sorted(Path(MODEL_PATH).glob("*.safetensors")):
    with safe_open(str(fname), framework="pt") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key).float().numpy())

with open(f"{MODEL_PATH}/vocab.json") as f:
    vocab = json.load(f)
inv_vocab = {v: k for k, v in vocab.items()}
with open(f"{MODEL_PATH}/tokenizer_config.json") as f:
    tok_cfg = json.load(f)
for key, info in tok_cfg.get('added_tokens_decoder', {}).items():
    vocab[info['content']] = int(key)
    inv_vocab[int(key)] = info['content']

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

def tokenize(text):
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
    return [vocab.get(p, 0) for p in chars]

# Build prompt
system_prompt = "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
tokens = [vocab['Ġ']]
tokens.append(vocab['Ġ'])
tokens.extend(tokenize(system_prompt))
tokens.append(vocab['Ġ'])
tokens.extend(tokenize("\n"))
tokens.append(vocab['Ġ'])
tokens.extend(tokenize("user\nHello"))
tokens.append(vocab['Ġ'])
tokens.extend(tokenize("\n"))
tokens.append(vocab['Ġ'])
tokens.extend(tokenize("assistant\n"))

def manual_attention(q, k, v, seq_len, scale, num_heads, num_kv_heads, head_dim):
    """Compute attention manually: Q@K^T, causal mask, softmax, @V. No SDPA."""
    attn_parts = []
    heads_per_group = num_heads // num_kv_heads
    for g in range(num_kv_heads):
        qg = mx.reshape(q[0, :, g*heads_per_group:(g+1)*heads_per_group, :],
                        [seq_len, heads_per_group, head_dim])
        kg = mx.reshape(k[0, :, g:g+1, :], [seq_len, 1, head_dim])
        vg = mx.reshape(v[0, :, g:g+1, :], [seq_len, 1, head_dim])
        # [heads_per_group, seq_len, head_dim] @ [seq_len, 1, head_dim]^T -> [heads_per_group, seq_len, seq_len]
        scores = mx.matmul(
            mx.transpose(qg, [1, 0, 2]),  # [seq_len, heads_per_group, head_dim]
            mx.transpose(kg, [1, 2, 0])   # [1, head_dim, seq_len]
        ) * scale  # [seq_len, heads_per_group, seq_len]
        # Apply causal mask
        causal = mx.tril(mx.ones((seq_len, seq_len)))
        scores = scores + (1 - causal) * -1e9
        # Softmax over last axis
        probs = mx.softmax(scores, axis=-1)
        # [seq_len, heads_per_group, seq_len] @ [seq_len, 1, head_dim]
        out = mx.matmul(
            probs,  # [seq_len, heads_per_group, seq_len]
            mx.reshape(vg, [1, seq_len, head_dim])  # broadcast to [heads_per_group, seq_len, head_dim]
        )  # [seq_len, heads_per_group, head_dim]
        attn_parts.append(mx.transpose(out, [1, 0, 2]))  # [heads_per_group, seq_len, head_dim]
    # Concat heads: [num_heads, seq_len, head_dim]
    all_heads = mx.concatenate(attn_parts, axis=0)
    # [num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim] -> [seq_len, hidden]
    result = all_heads.reshape(seq_len, num_heads * head_dim)
    return result

def sdpa_attention(q, k, v, seq_len, scale, num_heads, num_kv_heads, head_dim):
    """Compute attention using per-group SDPA."""
    attn_parts = []
    heads_per_group = num_heads // num_kv_heads
    for g in range(num_kv_heads):
        qg = mx.slice(q, mx.array([0, 0, g*heads_per_group, 0]),
                      mx.array([1, 2, 3]), [1, seq_len, heads_per_group, head_dim])
        kg = mx.slice(k, mx.array([0, 0, g, 0]),
                      mx.array([1, 2, 3]), [1, seq_len, 1, head_dim])
        vg = mx.slice(v, mx.array([0, 0, g, 0]),
                      mx.array([1, 2, 3]), [1, seq_len, 1, head_dim])
        attn_parts.append(mx.fast.scaled_dot_product_attention(
            qg, kg, vg, scale=scale, mask="causal"))
    attn_out = mx.concatenate(attn_parts, axis=2)
    return attn_out.reshape(seq_len, num_heads * head_dim)

def forward_attn(attn_fn, input_ids):
    """Full forward pass with specified attention function."""
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

        # RoPE
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        q = mx.fast.rope(q, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(k, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))

        # Attention
        scale = 1.0 / (head_dim ** 0.5)
        attn_out = attn_fn(q, k, v, seq_len, scale, num_heads, num_kv_heads, head_dim)
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

# Test both approaches
print("=== MANUAL ATTENTION ===")
test_tokens = list(tokens)
logits = forward_attn(manual_attention, test_tokens)
n0 = show_top5(logits, f"Step 0 (seq={len(test_tokens)})")
test_tokens.append(n0)
logits = forward_attn(manual_attention, test_tokens)
n1 = show_top5(logits, f"Step 1 (seq={len(test_tokens)})")
test_tokens.append(n1)
logits = forward_attn(manual_attention, test_tokens)
n2 = show_top5(logits, f"Step 2 (seq={len(test_tokens)})")
test_tokens.append(n2)
logits = forward_attn(manual_attention, test_tokens)
n3 = show_top5(logits, f"Step 3 (seq={len(test_tokens)})")
print(f"Tokens: {inv_vocab.get(n0,'?')}, {inv_vocab.get(n1,'?')}, {inv_vocab.get(n2,'?')}, {inv_vocab.get(n3,'?')}")

print("\n=== SDPA PER-GROUP ATTENTION ===")
test_tokens2 = list(tokens)
logits = forward_attn(sdpa_attention, test_tokens2)
n0b = show_top5(logits, f"Step 0 (seq={len(test_tokens2)})")
test_tokens2.append(n0b)
logits = forward_attn(sdpa_attention, test_tokens2)
n1b = show_top5(logits, f"Step 1 (seq={len(test_tokens2)})")
test_tokens2.append(n1b)
logits = forward_attn(sdpa_attention, test_tokens2)
n2b = show_top5(logits, f"Step 2 (seq={len(test_tokens2)})")
test_tokens2.append(n2b)
logits = forward_attn(sdpa_attention, test_tokens2)
n3b = show_top5(logits, f"Step 3 (seq={len(test_tokens2)})")
print(f"Tokens: {inv_vocab.get(n0b,'?')}, {inv_vocab.get(n1b,'?')}, {inv_vocab.get(n2b,'?')}, {inv_vocab.get(n3b,'?')}")
