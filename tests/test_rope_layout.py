"""Test MLX rope tensor layout requirements."""
import mlx.core as mx
import numpy as np

head_dim = 64
rope_base = 1000000.0

# Create identical inputs at two positions
data = np.random.randn(1, 2, 1, head_dim).astype(np.float32)
data[:, 1, :, :] = data[:, 0, :, :]  # make position 1 identical to position 0

# Test shape: [B, seq, heads, dim] — what we currently use
x_seq = mx.array(data)

# Test shape: [B, heads, seq, dim] — what PyTorch uses
x_heads = mx.array(data.transpose(0, 2, 1, 3))

print("=== Layout [B=1, seq=2, heads=1, dim=64] (current) ===")
rope_seq = mx.fast.rope(x_seq, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
mx.eval(rope_seq)
p0 = rope_seq[0, 0, 0]
p1 = rope_seq[0, 1, 0]
mx.eval(p0, p1)
diff = p0 - p1
mx.eval(diff)
print(f"pos0==pos1: {mx.allclose(p0, p1).item()}, diff_norm: {mx.sum(diff*diff).item():.6f}")

print("\n=== Layout [B=1, heads=1, seq=2, dim=64] ===")
rope_heads = mx.fast.rope(x_heads, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
mx.eval(rope_heads)
q0 = rope_heads[0, 0, 0]
q1 = rope_heads[0, 0, 1]
mx.eval(q0, q1)
diff2 = q0 - q1
mx.eval(diff2)
print(f"pos0==pos1: {mx.allclose(q0, q1).item()}, diff_norm: {mx.sum(diff2*diff2).item():.6f}")

# Test with different offset values
print("\n=== Testing offset parameter [B, seq, heads, dim] ===")
for offs in [0, 1, 100]:
    r = mx.fast.rope(x_seq, head_dim, traditional=False, base=rope_base, scale=1.0, offset=offs)
    mx.eval(r)
    a0 = r[0, 0, 0]
    a1 = r[0, 1, 0]
    mx.eval(a0, a1)
    d = a0 - a1
    mx.eval(d)
    print(f"  offset={offs}: pos0==pos1={mx.allclose(a0, a1).item()}, diff_norm={mx.sum(d*d).item():.6f}")

# What about traditional=True?
print("\n=== traditional=True [B, seq, heads, dim] ===")
r_trad = mx.fast.rope(x_seq, head_dim, traditional=True, base=rope_base, scale=1.0, offset=0)
mx.eval(r_trad)
t0 = r_trad[0, 0, 0]
t1 = r_trad[0, 1, 0]
mx.eval(t0, t1)
d_t = t0 - t1
mx.eval(d_t)
print(f"pos0==pos1: {mx.allclose(t0, t1).item()}, diff_norm: {mx.sum(d_t*d_t).item():.6f}")

# Let's try with big position differences
print("\n=== Bigger sequence: 5 positions, all same input [B, 5, heads, dim] ===")
data5 = np.tile(data[:, :1, :, :], (1, 5, 1, 1))  # [1, 5, 1, 64] all identical
x5 = mx.array(data5)
r5 = mx.fast.rope(x5, head_dim, traditional=False, base=rope_base, scale=1.0, offset=0)
mx.eval(r5)
for p in range(5):
    vec = r5[0, p, 0]
    mx.eval(vec)
    print(f"  pos {p}: norm={mx.sum(vec*vec).item():.4f}, first4={np.array(vec)[:4]}")

print("\nCheck if all positions are identical:")
all_same = True
for p in range(1, 5):
    d = r5[0, 0, 0] - r5[0, p, 0]
    mx.eval(d)
    same = mx.allclose(r5[0, 0, 0], r5[0, p, 0]).item()
    print(f"  pos0 vs pos{p}: same={same}, diff_norm={mx.sum(d*d).item():.6f}")
    if not same:
        all_same = False
if all_same:
    print("  ALL IDENTICAL! RoPE is NOT position-dependent.")
