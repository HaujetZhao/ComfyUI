# Wan block 分块路径数值校验：self-attention Q 分块 + FFN 分块 vs 全量，输出必须一致
# 用法: uv run python my/tmp/attn_chunk_validation.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

import comfy.ldm.wan.model as m

torch.manual_seed(0)
B, S, DIM, FFN_DIM, HEADS = 1, 100, 512, 13824, 8
block = m.WanAttentionBlock("t2v_cross_attn", DIM, FFN_DIM, HEADS, operation_settings={
    "operations": torch.nn, "device": "cuda", "dtype": torch.float16}).cuda()
x = torch.randn(B, S, DIM, device="cuda", dtype=torch.float16)
e = torch.randn(B, 6, DIM, device="cuda", dtype=torch.float16)
context = torch.randn(B, 77, DIM, device="cuda", dtype=torch.float16)
freqs = torch.randn(B, S, 1, DIM // HEADS // 2, 2, 2, device="cuda", dtype=torch.float32)  # rope_encode 同款

with torch.inference_mode():
    m.BLOCK_CHUNK_MIN_SEQ = 10**9  # 全量路径
    full = block(x, e, freqs, context)
    m.BLOCK_CHUNK_MIN_SEQ = 64     # 分块路径
    m.SELF_ATTN_CHUNK = 33
    m.FFN_CHUNK = 33
    chunked = block(x, e, freqs, context)

diff = (full - chunked).abs().max().item()
scale = full.abs().max().item()
print(f"max|diff|={diff:.6f}  max|out|={scale:.4f}  相对误差={diff/scale:.2e}")
assert diff / scale < 1e-2, "分块路径数值偏差过大"
print("OK: 分块与全量一致")
