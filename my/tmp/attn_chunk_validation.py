# WanSelfAttention Q 分块路径数值校验：同一模块同一输入，分块 vs 全量输出必须一致
# 用法: uv run python my/tmp/attn_chunk_validation.py
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
freqs = torch.randn(B, S, 1, DIM // HEADS // 2, 2, 2, device="cuda", dtype=torch.float32)  # rope_encode 同款: [B,S,1,D/2,2,2]

with torch.inference_mode():
    m.SELF_ATTN_CHUNK_MIN_SEQ = 10**9  # 全量路径
    full = attn(x, freqs)
    m.SELF_ATTN_CHUNK_MIN_SEQ = 64     # 分块路径（chunk=4096 > 100，实际单块，再验 33 的小块）
    m.SELF_ATTN_CHUNK = 33
    chunked = attn(x, freqs)

diff = (full - chunked).abs().max().item()
scale = full.abs().max().item()
print(f"max|diff|={diff:.6f}  max|out|={scale:.4f}  相对误差={diff/scale:.2e}")
assert diff / scale < 1e-2, "分块路径数值偏差过大"
print("OK: 分块与全量一致")
