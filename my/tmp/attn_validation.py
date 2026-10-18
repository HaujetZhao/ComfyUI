import os
import sys

# Add ComfyUI to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import math
from comfy.ldm.wan.model import WanAttentionBlock
from comfy.ldm.flux.layers import EmbedND
import comfy.ops
import comfy.model_management as mm

def test_original_block():
    # Use CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Testing on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU (Memory stats won't be accurate)")
        device = torch.device("cpu")
        
    dtype = torch.float16
    
    # Init params for Wan 2.1 T2V 1.3B (1.3B model has dim=1536, 14B likely larger)
    dim = 1536
    ffn_dim = 8192 
    num_heads = 12
    head_dim = dim // num_heads # 128
    
    # Using standard Comfy operations
    ops = comfy.ops.disable_weight_init
    operation_settings = {
        "operations": ops,
        "device": device,
        "dtype": dtype,
    }
    
    print(f"Initializing WanAttentionBlock on {device}...")
    # NOTE: Set seeds for reproducibility where possible
    torch.manual_seed(42)
    
    block = WanAttentionBlock(
        cross_attn_type='t2v_cross_attn',
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        operation_settings=operation_settings
    ).to(device=device, dtype=dtype)
    
    # --- 用户可调参数 ---
    video_width = 720    # 原始宽度
    video_height = 1280   # 原始高度
    video_frames = 150   # 原始帧数
    # -------------------

    # VAE 压缩 (空间 8x, 时间 4x)
    lat_h = video_height // 8
    lat_w = video_width // 8
    lat_t = ((video_frames - 1) // 4) + 1
    
    # DIT Patch 压缩 (空间 2x, 时间 2x) 为 AnimateWanModel 默认配置
    # 注意：Wan 2.1 T2V 官网通常是 (1, 2, 2)，但 Animate 为 (2, 2, 2)
    patch_t, patch_h, patch_w = 2, 2, 2
    
    L = (lat_t // patch_t) * (lat_h // patch_h) * (lat_w // patch_w)
    B = 1
    C = dim
    
    print(f"Video Configuration: {video_width}x{video_height}, {video_frames} frames")
    print(f"VAE Latent: {lat_w}x{lat_h}, {lat_t} frames")
    print(f"Transformer Sequence Length (L): {L}")
    
    print(f"Preparing input tensors (L={L}, B={B}, C={C})...")
    x = torch.randn(B, L, C, device=device, dtype=dtype)

    e = torch.randn(B, 6, C, device=device, dtype=dtype)
    context = torch.randn(B, 257, C, device=device, dtype=dtype)
    
    # ROPE Frequencies
    d = head_dim
    # axes_dim logic from WanModel
    axes_dim = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=axes_dim).to(device=device)
    
    # Fake grid for L tokens
    # Wan uses 3D grid [T, H, W]
    grid = torch.stack(torch.meshgrid(
        torch.arange(lat_t // patch_t, device=device),
        torch.arange(lat_h // patch_h, device=device),
        torch.arange(lat_w // patch_w, device=device),
        indexing='ij'
    ), dim=-1).float().reshape(1, -1, 3)

    
    print("Generating ROPE frequencies...")
    with torch.no_grad():
        freqs = rope_embedder(grid).transpose(1, 2) # [B, L, 1, 64, 2, 2]
    
    # Transformer options
    transformer_options = {
        "device": device,
        "dtype": dtype,
    }
    
    # Warmup pass
    print("Warming up (Pass 1)...")
    try:
        with torch.inference_mode():
            _ = block(x, e=e, freqs=freqs, context=context, transformer_options=transformer_options)
    except Exception as err:
        print(f"Warmup failed: {err}")
        return

    # Reset stats
    mm.soft_empty_cache()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    initial_mem = torch.cuda.memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0
    
    print(f"Running benchmark forward pass...")
    start_time = time.time()
    try:
        with torch.inference_mode():
            out = block(x, e=e, freqs=freqs, context=context, transformer_options=transformer_options)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        end_time = time.time()
        
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0
        print("\n" + "="*40)
        print(f"BASELINE RESULTS (Official Block)")
        print("="*40)
        print(f"Sequence Length: {L}")
        print(f"Inference Time:  {end_time - start_time:.4f}s")
        print(f"Initial VRAM:    {initial_mem:.2f} MB")
        print(f"Peak VRAM:       {peak_vram:.2f} MB")
        print(f"Active Overhead: {peak_vram - initial_mem:.2f} MB")
        print(f"Output Signal:   {out.abs().mean().item():.6f}")
        print("="*40 + "\n")
        
    except Exception as err:
        print(f"Benchmark failed: {err}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_original_block()
