# 最小复现：self_attn 分块（调制下放+K/V buffer）vs 全量
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

import comfy.ldm.wan.model as m

torch.manual_seed(0)
B, S, DIM, HEADS = 1, 100, 512, 8
attn = m.WanSelfAttention(DIM, HEADS, operation_settings={
    "operations": torch.nn, "device": "cuda", "dtype": torch.float16}).cuda()
x = torch.randn(B, S, DIM, device="cuda", dtype=torch.float16)
e = torch.randn(B, 6, DIM, device="cuda", dtype=torch.float16)
e = e.chunk(6, dim=1)  # 模拟 block 里的 chunk
freqs = torch.randn(B, S, 1, DIM // HEADS // 2, 2, 2, device="cuda", dtype=torch.float32)

with torch.inference_mode():
    full = attn(m.modulate(x, torch.nn.Identity(), e[0], e[1]), freqs)
    # 全量路径也走 modulation 入口（和 block 现在的调用一致）
    m.BLOCK_CHUNK_MIN_SEQ = 10**9
    full2 = attn(x, freqs, modulation=(torch.nn.Identity(), e[0], e[1]))
    m.BLOCK_CHUNK_MIN_SEQ = 64
    m.SELF_ATTN_CHUNK = 33
    chunked = attn(x, freqs, modulation=(torch.nn.Identity(), e[0], e[1]))

for name, out in (("full vs full2", full2), ("full vs chunked", chunked)):
    d = (full - out).abs().max().item()
    print(f"{name}: max|diff|={d:.6f} 相对={d / full.abs().max().item():.2e}")

# 逐环节对比：调制块 / k 块(rope 后) / v 块 / q 块 / attention 输出块
with torch.inference_mode():
    from comfy.ldm.flux.math import apply_rope1
    from comfy.ldm.modules.attention import optimized_attention
    xm_full = m.modulate(x, torch.nn.Identity(), e[0], e[1])
    k_full = apply_rope1(attn.norm_k(attn.k(xm_full)).view(B, S, HEADS, DIM // HEADS), freqs).view(B, S, DIM)
    v_full = attn.v(xm_full)
    q_full = apply_rope1(attn.norm_q(attn.q(xm_full)).view(B, S, HEADS, DIM // HEADS), freqs).view(B, S, DIM)
    out_full = optimized_attention(q_full, k_full, v_full, heads=HEADS)
    for s0 in range(0, S, 33):
        s1 = min(s0 + 33, S)
        xm = m.modulate(x[:, s0:s1], torch.nn.Identity(), e[0], e[1])
        kb = apply_rope1(attn.norm_k(attn.k(xm)).view(B, s1 - s0, HEADS, DIM // HEADS), freqs[:, s0:s1]).view(B, s1 - s0, DIM)
        vb = attn.v(xm)
        qb = apply_rope1(attn.norm_q(attn.q(xm)).view(B, s1 - s0, HEADS, DIM // HEADS), freqs[:, s0:s1]).view(B, s1 - s0, DIM)
        ob = optimized_attention(qb, k_full, v_full, heads=HEADS)  # 用全量 k/v 隔离 q 因素
        ob2 = optimized_attention(qb, kb, vb, heads=HEADS)  # 分块 k/v
        print(f"chunk {s0}:{s1} q diff={((q_full[:, s0:s1] - qb).abs().max()).item():.6f} "
              f"attn(full kv) diff={((out_full[:, s0:s1] - ob).abs().max()).item():.6f} "
              f"attn(chunk kv) diff={((out_full[:, s0:s1] - ob2).abs().max()).item():.6f}")
