"""Deeper debug: trace why attention output is identical despite different Q/K."""
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
rope_base = config.get('rope_theta', 1000000.0)

weights = {}
for fname in sorted(Path(MODEL_PATH).glob("*.safetensors")):
    with safe_open(str(fname), framework="pt") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key).float().numpy())

seq_len = 4
np.random.seed(42)
h = mx.array(np.random.randn(seq_len, hidden).astype(np.float32))

pfx = "model.layers.0."
scale = 1.0 / (head_dim ** 0.5)
heads_per_group = num_heads // num_kv_heads

# Compute Q, K, V (same for both approaches)
normed = mx.fast.rms_norm(h, weights[pfx + "input_layernorm.weight"], 1e-6)

q_raw = normed @ weights[pfx + "self_attn.q_proj.weight"].T
k_raw = normed @ weights[pfx + "self_attn.k_proj.weight"].T
v_raw = normed @ weights[pfx + "self_attn.v_proj.weight"].T

q = q_raw.reshape(1, seq_len, num_heads, head_dim)
k = k_raw.reshape(1, seq_len, num_kv_heads, head_dim)
v = v_raw.reshape(1, seq_len, num_kv_heads, head_dim)

# Approach A: no transpose
qA = mx.fast.rope(q, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
kA = mx.fast.rope(k, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)

# Approach B: with transpose
qBt = mx.transpose(q, (0, 2, 1, 3))
kBt = mx.transpose(k, (0, 2, 1, 3))
qBt = mx.fast.rope(qBt, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
kBt = mx.fast.rope(kBt, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
qB = mx.transpose(qBt, (0, 2, 1, 3))
kB = mx.transpose(kBt, (0, 2, 1, 3))

mx.eval(qA, kA, qB, kB)

# Check if K differs
k_diff = kA - kB
mx.eval(k_diff)
print(f"K diff norm: {mx.sum(k_diff * k_diff).item():.4f}")
print(f"Q diff norm: {mx.sum((qA-qB)*(qA-qB)).item():.4f}")

# Now compute attention SCORES (Q * K^T) for each approach
# Focus on last position's Q vs all K
qA_last = qA[0, -1]      # [num_heads, head_dim]
qB_last = qB[0, -1]      # [num_heads, head_dim]
kA_all = kA[0]            # [seq_len, num_kv_heads, head_dim]
kB_all = kB[0]            # [seq_len, num_kv_heads, head_dim]

mx.eval(qA_last, qB_last, kA_all, kB_all)

# For group 0: heads 0-6 use KV head 0
qA_g0 = qA_last[:7]       # [7, 64]
qB_g0 = qB_last[:7]       # [7, 64]
kA_g0 = kA_all[:, 0, :]   # [seq_len, 64]
kB_g0 = kB_all[:, 0, :]   # [seq_len, 64]

# Scores: Q * K^T
scores_A = qA_g0 @ kA_g0.T  # [7, seq_len]
scores_B = qB_g0 @ kB_g0.T  # [7, seq_len]
mx.eval(scores_A, scores_B)
scores_diff = scores_A - scores_B
mx.eval(scores_diff)
print(f"\nGroup 0 scores diff norm: {mx.sum(scores_diff * scores_diff).item():.6f}")
print(f"Scores A mean: {mx.mean(scores_A).item():.4f}, Scores B mean: {mx.mean(scores_B).item():.4f}")

# Compare scores for the LAST Q head (head 6) with all K positions
sA_h6 = scores_A[6]  # attention scores for head 6
sB_h6 = scores_B[6]
mx.eval(sA_h6, sB_h6)
print(f"Scores A head6: {np.array(sA_h6).round(4)}")
print(f"Scores B head6: {np.array(sB_h6).round(4)}")

# Apply causal mask and softmax
def causal_softmax(scores):
    # Apply causal mask by setting upper triangle to -inf
    L = scores.shape[-1]
    mask = mx.tril(mx.ones((L, L)))
    return mx.softmax(scores * scale, axis=-1)  # simplified, no causal mask test

# Actually let's test the full SDPA output per group
print("\n=== Full SDPA per group ===")
for g in range(num_kv_heads):
    qAg = mx.slice(qA, mx.array([0, 0, g * heads_per_group, 0]),
                  [0, 1, 2, 3], [1, seq_len, heads_per_group, head_dim])
    qBg = mx.slice(qB, mx.array([0, 0, g * heads_per_group, 0]),
                  [0, 1, 2, 3], [1, seq_len, heads_per_group, head_dim])
    kAg = mx.slice(kA, mx.array([0, 0, g, 0]),
                  [0, 1, 2, 3], [1, seq_len, 1, head_dim])
    kBg = mx.slice(kB, mx.array([0, 0, g, 0]),
                  [0, 1, 2, 3], [1, seq_len, 1, head_dim])
    vAg = mx.slice(v, mx.array([0, 0, g, 0]),
                  [0, 1, 2, 3], [1, seq_len, 1, head_dim])

    attnA = mx.fast.scaled_dot_product_attention(qAg, kAg, vAg, scale=scale, mask="causal")
    attnB = mx.fast.scaled_dot_product_attention(qBg, kBg, vAg, scale=scale, mask="causal")
    mx.eval(attnA, attnB)
    d = attnA - attnB
    mx.eval(d)
    print(f"  Group {g}: attn diff norm = {mx.sum(d*d).item():.6f}")

# What if the issue is that RoPE produces same Q/K for same input positions
# regardless of layout? When the raw Q/K are different per position (random data),
# the layouts differ. But when Q/K are the SAME per position (identical tokens),
# they produce same results regardless of layout?

# Let's test: same input at positions 2 and 3
print("\n=== Test with identical input at positions 2,3 ===")
q_same = mx.array(q)
k_same = mx.array(k)
q_same_slice = q_same[0, 2:3]  # position 2's Q
q_same_slice2 = mx.concatenate([q_same_slice, q_same_slice], axis=0)  # duplicate

# Make positions 2 and 3 identical in Q
# (We'll create a smaller test with positions 2 and 3 being the same)

# Actually, let me test a simpler thing: does MLX rope produce different
# outputs for the same input at different positions when using the correct layout?
print("=== Simple RoPE test: same input, different positions ===")
test_input = mx.array(np.random.randn(1, 2, 1, head_dim).astype(np.float32))
test_input[0, 1] = test_input[0, 0]  # copy position 0 to position 1

# WRONG: [B, seq=2, heads=1, dim]
r1 = mx.fast.rope(test_input, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
mx.eval(r1)
d1 = r1[0, 0] - r1[0, 1]
mx.eval(d1)
print(f"Layout [B, seq, heads, dim]: pos0-pos1 diff = {mx.sum(d1*d1).item():.6f}")

# CORRECT: transpose to [B, heads, seq, dim]
test_t = mx.transpose(test_input, (0, 2, 1, 3))  # [1, 1, 2, 64]
r2 = mx.fast.rope(test_t, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
r2 = mx.transpose(r2, (0, 2, 1, 3))  # back to [1, 2, 1, 64]
mx.eval(r2)
d2 = r2[0, 0] - r2[0, 1]
mx.eval(d2)
print(f"Layout [B, heads, seq, dim]: pos0-pos1 diff = {mx.sum(d2*d2).item():.6f}")
