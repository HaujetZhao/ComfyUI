# 720p 对比：tiled 32×32 vs 64×64 latent 切片（重叠均为 8 latent）的速度/峰值/相互差
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import comfy.model_management as mm
import comfy.utils
from comfy.sd import VAE

WIDTH, HEIGHT = 1280, 720
LATENT_FRAMES = 4
TILES = (32, 64)  # latent 切片边长，重叠固定 8 latent
VAE_PATH = os.path.abspath("models/vae/wan_2.1_vae.safetensors")

device = mm.vae_device()
sd = comfy.utils.load_torch_file(VAE_PATH)
vae = VAE(sd=sd, device=device)
vae.first_stage_model.to(device)
torch.cuda.empty_cache()

z_cpu = torch.rand((1, 16, LATENT_FRAMES, HEIGHT // 8, WIDTH // 8))
dec_blocks = 1 + LATENT_FRAMES // 2

results = {}
with torch.inference_mode():
    for ts in TILES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = vae.decode_tiled_3d(z_cpu, tile_t=999, tile_x=ts, tile_y=ts, overlap=(1, 8, 8))
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        results[f"tile{ts}"] = (dt, peak, out.float())
        print(f"[tile{ts:2d}] {dt:.1f}s ({dt*1000/dec_blocks:.0f}ms/块) 峰值={peak:.0f}MB", flush=True)

ref = results[f"tile{TILES[0]}"][2]
for ts in TILES[1:]:
    dt, peak, px = results[f"tile{ts}"]
    d = (px - ref).abs()
    print(f"\n[tile{ts} vs tile{TILES[0]}] 最大逐元素差={d.max().item():.2e} "
          f"平均差={d.mean().item():.2e}", flush=True)
