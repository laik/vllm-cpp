"""Verify that transpose+rope+untranspose changes outputs in a full layer."""
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

# Test with random input, single layer
seq_len = 4
np.random.seed(42)
h = mx.array(np.random.randn(seq_len, hidden).astype(np.float32))

pfx = "model.layers.0."

# Approach A: WRONG layout (no transpose — RoPE sees heads as positions)
print("=== Approach A: WRONG layout (no transpose) ===")
normed_A = mx.fast.rms_norm(h, weights[pfx + "input_layernorm.weight"], 1e-6)
residual_A = h

q_A = normed_A @ weights[pfx + "self_attn.q_proj.weight"].T
k_A = normed_A @ weights[pfx + "self_attn.k_proj.weight"].T
v_A = normed_A @ weights[pfx + "self_attn.v_proj.weight"].T

q_A = q_A.reshape(1, seq_len, num_heads, head_dim)
k_A = k_A.reshape(1, seq_len, num_kv_heads, head_dim)
v_A = v_A.reshape(1, seq_len, num_kv_heads, head_dim)

# Wrong: RoPE applied directly to [B, seq, heads, dim]
q_A_rope = mx.fast.rope(q_A, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
k_A_rope = mx.fast.rope(k_A, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)

# Approach B: CORRECT layout (transpose → RoPE → un-transpose)
print("=== Approach B: CORRECT layout (transpose) ===")
normed_B = mx.fast.rms_norm(h, weights[pfx + "input_layernorm.weight"], 1e-6)

q_B = normed_B @ weights[pfx + "self_attn.q_proj.weight"].T
k_B = normed_B @ weights[pfx + "self_attn.k_proj.weight"].T
v_B = normed_B @ weights[pfx + "self_attn.v_proj.weight"].T

q_B = q_B.reshape(1, seq_len, num_heads, head_dim)
k_B = k_B.reshape(1, seq_len, num_kv_heads, head_dim)
v_B = v_B.reshape(1, seq_len, num_kv_heads, head_dim)

# Correct: transpose before/after RoPE
q_B = mx.transpose(q_B, (0, 2, 1, 3))
k_B = mx.transpose(k_B, (0, 2, 1, 3))
q_B_rope = mx.fast.rope(q_B, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
k_B_rope = mx.fast.rope(k_B, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
q_B_rope = mx.transpose(q_B_rope, (0, 2, 1, 3))
k_B_rope = mx.transpose(k_B_rope, (0, 2, 1, 3))

# Compare
mx.eval(q_A_rope, q_B_rope, k_A_rope, k_B_rope)
print(f"Q_A_rope shape: {q_A_rope.shape}")
print(f"Q_B_rope shape: {q_B_rope.shape}")
print(f"Q_A == Q_B? {mx.allclose(q_A_rope, q_B_rope).item()}")

q_diff = q_A_rope - q_B_rope
mx.eval(q_diff)
print(f"Q diff norm: {mx.sum(q_diff * q_diff).item():.4f}")

# Check position discrimination
print("\n=== Position discrimination check ===")
# A: wrong layout
a0 = q_A_rope[0, 0, 0]  # seq_pos=0, head=0
a1 = q_A_rope[0, 1, 0]  # seq_pos=1, head=0
a2 = q_A_rope[0, 2, 0]  # seq_pos=2, head=0
mx.eval(a0, a1, a2)
print(f"A: pos0 norm={mx.sum(a0*a0).item():.2f}, pos1 norm={mx.sum(a1*a1).item():.2f}, pos2 norm={mx.sum(a2*a2).item():.2f}")
da01 = a0 - a1
da02 = a0 - a2
mx.eval(da01, da02)
print(f"A: pos0-pos1 diff={mx.sum(da01*da01).item():.4f}, pos0-pos2 diff={mx.sum(da02*da02).item():.4f}")

# B: correct layout
b0 = q_B_rope[0, 0, 0]
b1 = q_B_rope[0, 1, 0]
b2 = q_B_rope[0, 2, 0]
mx.eval(b0, b1, b2)
print(f"B: pos0 norm={mx.sum(b0*b0).item():.2f}, pos1 norm={mx.sum(b1*b1).item():.2f}, pos2 norm={mx.sum(b2*b2).item():.2f}")
db01 = b0 - b1
db02 = b0 - b2
mx.eval(db01, db02)
print(f"B: pos0-pos1 diff={mx.sum(db01*db01).item():.4f}, pos0-pos2 diff={mx.sum(db02*db02).item():.4f}")

# Check if within the SAME approach, all positions are identical
# (which would confirm the RoPE layout bug)
print(f"\nA: all positions same = {mx.allclose(a0, a1).item() and mx.allclose(a1, a2).item()}")
print(f"B: all positions same = {mx.allclose(b0, b1).item() and mx.allclose(b1, b2).item()}")

# Test: compare full attention output for both approaches
scale = 1.0 / (head_dim ** 0.5)
heads_per_group = num_heads // num_kv_heads

def compute_attn(q_rope, k_rope, v):
    attn_parts = []
    for g in range(num_kv_heads):
        qg = mx.slice(q_rope, mx.array([0, 0, g * heads_per_group, 0]),
                     [0, 1, 2, 3], [1, seq_len, heads_per_group, head_dim])
        kg = mx.slice(k_rope, mx.array([0, 0, g, 0]),
                     [0, 1, 2, 3], [1, seq_len, 1, head_dim])
        vg = mx.slice(v, mx.array([0, 0, g, 0]),
                     [0, 1, 2, 3], [1, seq_len, 1, head_dim])
        attn_parts.append(mx.fast.scaled_dot_product_attention(
            qg, kg, vg, scale=scale, mask="causal"))
    attn_out = mx.concatenate(attn_parts, axis=2)
    return attn_out.reshape(seq_len, num_heads * head_dim)

attn_A = compute_attn(q_A_rope, k_A_rope, v_A)
attn_B = compute_attn(q_B_rope, k_B_rope, v_B)
mx.eval(attn_A, attn_B)
attn_diff = attn_A - attn_B
mx.eval(attn_diff)
print(f"\nAttention output diff norm: {mx.sum(attn_diff * attn_diff).item():.4f}")
print(f"Attn A == Attn B? {mx.allclose(attn_A, attn_B).item()}")
# Check per-position attn output for last position
last_A = attn_A[-1]
last_B = attn_B[-1]
mx.eval(last_A, last_B)
la_diff = last_A - last_B
mx.eval(la_diff)
print(f"Last position attn diff norm: {mx.sum(la_diff * la_diff).item():.4f}")
