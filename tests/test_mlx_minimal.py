"""Minimal test to isolate the stuck-logits bug."""
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

# Load weights
weights = {}
for fname in sorted(Path(MODEL_PATH).glob("*.safetensors")):
    with safe_open(str(fname), framework="pt") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key).float().numpy())

def forward_debug(input_ids, label=""):
    """Forward with debug output for last position hidden state."""
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

    # DEBUG: Compare last two positions
    last = h[-1]
    prev = h[-2]
    diff = last - prev
    mx.eval(last, prev, diff)
    print(f"  {label}: last_norm={mx.sum(last*last).item():.2f}, "
          f"prev_norm={mx.sum(prev*prev).item():.2f}, "
          f"diff_norm={mx.sum(diff*diff).item():.6f}")

    lm_head_w = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])
    return last @ lm_head_w.T

# Build a MINIMAL test: just 2 identical tokens
t1 = [100, 200]
t2 = [100, 200, 200]
t3 = [100, 200, 200, 200]

print("=== Minimal test: identical tokens ===")
l1 = forward_debug(t1, "seq=[100,200]")
l2 = forward_debug(t2, "seq=[100,200,200]")
l3 = forward_debug(t3, "seq=[100,200,200,200]")

mx.eval(l1, l2, l3)
print(f"logits1 last: {l1[-1].item():.4f}" if False else "")
print(f"l1 argmax: {mx.argmax(l1).item()}, l2 argmax: {mx.argmax(l2).item()}, l3 argmax: {mx.argmax(l3).item()}")
# Check if logits for last position are identical between l2 and l3
l2_last = l2  # logits are already just for last position
l3_last = l3
diff_logits = l2_last - l3_last
mx.eval(diff_logits)
print(f"logits diff norm (l2 vs l3): {mx.sum(diff_logits * diff_logits).item():.6f}")

# Now test with diverse tokens
print("\n=== Diverse tokens test ===")
d1 = [100, 200, 300]
d2 = [100, 200, 300, 400]
d3 = [100, 200, 300, 400, 500]

ld1 = forward_debug(d1, "seq=[100,200,300]")
ld2 = forward_debug(d2, "seq=[100,200,300,400]")
ld3 = forward_debug(d3, "seq=[100,200,300,400,500]")

print(f"d1 argmax: {mx.argmax(ld1).item()}, d2 argmax: {mx.argmax(ld2).item()}, d3 argmax: {mx.argmax(ld3).item()}")
diff_d = ld2 - ld3
mx.eval(diff_d)
print(f"logits diff norm (ld2 vs ld3): {mx.sum(diff_d * diff_d).item():.6f}")
