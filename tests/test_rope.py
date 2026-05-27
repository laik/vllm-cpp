"""Test if RoPE makes position-dependent encodings."""
import mlx.core as mx
import numpy as np

head_dim = 64
hidden = 896
num_heads = 14
num_kv_heads = 2
rope_base = 1000000.0

# Create random Q, K vectors for two adjacent positions
np.random.seed(42)
q_data = np.random.randn(1, 2, num_heads, head_dim).astype(np.float32)
k_data = np.random.randn(1, 2, num_kv_heads, head_dim).astype(np.float32)

q = mx.array(q_data)
k = mx.array(k_data)

print(f"Q shape: {q.shape}")
print(f"K shape: {k.shape}")

# Apply RoPE
q_roped = mx.fast.rope(q, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
k_roped = mx.fast.rope(k, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)

mx.eval(q_roped, k_roped)

# Check if position 0 and 1 have different RoPE encodings
q0 = q_roped[0, 0]  # position 0
q1 = q_roped[0, 1]  # position 1
k0 = k_roped[0, 0]
k1 = k_roped[0, 1]

mx.eval(q0, q1, k0, k1)

# Difference
q_diff = q0 - q1
k_diff = k0 - k1
mx.eval(q_diff, k_diff)

print(f"\nRope Q position 0 vs 1 diff norm: {mx.sum(q_diff * q_diff).item():.4f}")
print(f"Rope K position 0 vs 1 diff norm: {mx.sum(k_diff * k_diff).item():.4f}")

# Check if q0 == q1 (without RoPE the original would differ since random data)
# But with RoPE, positions 0 and 1 should differ MORE
print(f"Q pos0 norm: {mx.sum(q0*q0).item():.2f}, Q pos1 norm: {mx.sum(q1*q1).item():.2f}")

# Now test: what if Q input for positions 0 and 1 are IDENTICAL?
same_q = mx.array(np.random.randn(1, 2, 1, head_dim).astype(np.float32))
same_q[0, 1] = same_q[0, 0]  # make identical
same_k = mx.array(np.random.randn(1, 2, 1, head_dim).astype(np.float32))
same_k[0, 1] = same_k[0, 0]

print(f"\n=== Same input, different positions ===")
same_q_rope = mx.fast.rope(same_q, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
same_k_rope = mx.fast.rope(same_k, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
mx.eval(same_q_rope, same_k_rope)

sq0 = same_q_rope[0, 0]
sq1 = same_q_rope[0, 1]
sk0 = same_k_rope[0, 0]
sk1 = same_k_rope[0, 1]
mx.eval(sq0, sq1, sk0, sk1)

sq_diff = sq0 - sq1
sk_diff = sk0 - sk1
mx.eval(sq_diff, sk_diff)

print(f"Same-input Q diff norm: {mx.sum(sq_diff*sq_diff).item():.6f}")
print(f"Same-input K diff norm: {mx.sum(sk_diff*sk_diff).item():.6f}")
print(f"Input Q norm: {mx.sum(same_q[0,0]*same_q[0,0]).item():.2f}")
print(f"Rope Q pos0 norm: {mx.sum(sq0*sq0).item():.2f}, pos1 norm: {mx.sum(sq1*sq1).item():.2f}")

# Test: what happens with the "causal" mask and identical Q,K at consecutive positions?
# Does SDPA produce same output for both positions?
scale = 1.0 / (head_dim ** 0.5)
print(f"\n=== SDPA with identical Q,K at adjacent positions ===")
attn = mx.fast.scaled_dot_product_attention(same_q_rope, same_k_rope, same_k_rope, scale=scale, mask="causal")
mx.eval(attn)
a0 = attn[0, 0]
a1 = attn[0, 1]
mx.eval(a0, a1)
a_diff = a0 - a1
mx.eval(a_diff)
print(f"Attn output diff norm (pos 0 vs pos 1): {mx.sum(a_diff*a_diff).item():.6f}")
print(f"Attn pos0 norm: {mx.sum(a0*a0).item():.6f}, pos1 norm: {mx.sum(a1*a1).item():.6f}")
print(f"Attn pos0[:8]: {np.array(a0).flatten()[:8]}")
print(f"Attn pos1[:8]: {np.array(a1).flatten()[:8]}")
