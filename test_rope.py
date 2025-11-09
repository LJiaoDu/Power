import torch
import numpy as np

def _rope_cos_sin(seq_len: int, dim: int, device, dtype, base: float = 10000.0):
    """
    计算RoPE位置编码的cos和sin值
    标准公式: theta_i = base^(-2i/d), 其中 i = 0, 1, ..., d/2-1
    """
    assert dim % 2 == 0, "RoPE 维度必须为偶数"
    half = dim // 2
    # 修复：除以完整维度 dim，而不是 half
    inv_freq = 1.0 / (base ** (2 * torch.arange(0, half, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum('l,d->ld', t, inv_freq)  # [seq_len, half]
    # 修复：不需要 cat，直接返回
    emb = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half]
    return emb.cos(), emb.sin()

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    应用RoPE旋转变换
    旋转矩阵: [[cos, -sin], [sin, cos]]
    """
    x_even = x[..., ::2]   # 偶数位置
    x_odd = x[..., 1::2]   # 奇数位置
    # 修复：标准旋转公式，不需要 [..., ::2] 索引
    x_rope_even = x_even * cos - x_odd * sin
    x_rope_odd = x_even * sin + x_odd * cos  # 修复：x_even 在前
    out = torch.empty_like(x)
    out[..., ::2] = x_rope_even
    out[..., 1::2] = x_rope_odd
    return out

# 测试
print("=== RoPE Implementation Test ===\n")

# 测试1: 形状验证
print("Test 1: Shape validation")
B, nhead, L, dim = 2, 8, 10, 64
x = torch.randn(B, nhead, L, dim)
cos, sin = _rope_cos_sin(L, dim, device='cpu', dtype=torch.float32)

print(f"Input x shape: {x.shape}")
print(f"cos shape: {cos.shape}")
print(f"sin shape: {sin.shape}")

x_rotated = _apply_rope(x, cos, sin)
print(f"Output shape: {x_rotated.shape}")
print(f"Shape match: {x.shape == x_rotated.shape}")
print()

# 测试2: 频率公式验证
print("Test 2: Frequency formula validation")
dim = 8
half = dim // 2
base = 10000.0
inv_freq = 1.0 / (base ** (2 * torch.arange(0, half) / dim))

print(f"dim = {dim}, half = {half}")
print(f"Expected theta values (base^(-2i/d)):")
for i in range(half):
    expected = base ** (-2 * i / dim)
    actual = inv_freq[i].item()
    print(f"  i={i}: expected={expected:.6f}, actual={actual:.6f}, match={abs(expected-actual)<1e-6}")
print()

# 测试3: 旋转不变性验证（范数应该保持）
print("Test 3: Rotation invariance (norm preservation)")
x = torch.randn(2, 4, 10, 16)
cos, sin = _rope_cos_sin(10, 16, device='cpu', dtype=torch.float32)
x_rotated = _apply_rope(x, cos, sin)

norm_before = torch.norm(x, dim=-1)
norm_after = torch.norm(x_rotated, dim=-1)
norm_diff = torch.abs(norm_before - norm_after).max().item()

print(f"Max norm difference: {norm_diff:.6e}")
print(f"Norm preserved: {norm_diff < 1e-5}")
print()

# 测试4: 位置0应该保持theta=0
print("Test 4: Position 0 should have theta=0")
seq_len = 5
dim = 4
cos, sin = _rope_cos_sin(seq_len, dim, device='cpu', dtype=torch.float32)

print(f"cos at position 0: {cos[0, 0, 0].numpy()}")
print(f"sin at position 0: {sin[0, 0, 0].numpy()}")
print(f"cos[0] ≈ 1: {torch.allclose(cos[0, 0, 0], torch.ones(dim//2))}")
print(f"sin[0] ≈ 0: {torch.allclose(sin[0, 0, 0], torch.zeros(dim//2))}")
print()

print("=== All Tests Completed ===")
