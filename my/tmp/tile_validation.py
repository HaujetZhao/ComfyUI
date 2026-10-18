import os
import sys
import torch
import numpy as np

# Add ComfyUI to path
sys.path.append(r"e:\ComfyUI")

import comfy.model_management as model_management
import comfy.utils
from comfy.sd import VAE
from comfy_extras.wan_vae_stream import WanVAEStreamEncoder

# Dummy wrapper for validation script if needed
def vae_stream_encode(vae, image_cpu, chunk_size=32, tile_size=512, overlap=64, offload_to_cpu=True):
    encoder = WanVAEStreamEncoder(vae, tile_size=tile_size, overlap=overlap, offload_to_cpu=offload_to_cpu)
    return encoder.encode(image_cpu, chunk_size=chunk_size)

def compare_latents(l1, l2):
    # Flatten and calculate cosine similarity
    v1 = l1.flatten().float()
    v2 = l2.flatten().float()
    
    cos_sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0)
    mae = torch.mean(torch.abs(v1 - v2))
    max_err = torch.max(torch.abs(v1 - v2))
    
    return cos_sim.item(), mae.item(), max_err.item()

def run_tiled_stream_validation(width=1920, height=1080, length=16, chunk_size=8, tile_size=512, overlap=64, 
                               test_official=True, test_safe=True):
    device = model_management.vae_device()
    vae_path = r"e:\ComfyUI\models\vae\wan_2.1_vae.safetensors"
    
    print(f"Loading VAE from {vae_path}...")
    sd = comfy.utils.load_torch_file(vae_path)
    vae = VAE(sd=sd)
    print("VAE Loaded.")
    
    # Generate random data [T, H, W, C]
    print(f"Generating random {length} frames of {width}x{height} pixels...")
    torch.manual_seed(42)
    pixels = torch.randn((length, height, width, 3))
    
    import time

    # 0. Official VAE One-shot (Baseline)
    latent_off = None
    time_off = 0
    peak_off = 0
    if test_official:
        print(f"\n[Test 0] Performing Official VAE One-shot Inference...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            start_time = time.perf_counter()
            with torch.inference_mode():
                # Official encode method
                latent_off = vae.encode(pixels)
            time_off = time.perf_counter() - start_time
            peak_off = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Official VAE Peak VRAM: {peak_off:.2f} MB (Time: {time_off:.2f}s)")
        except RuntimeError as e:
            if "Out of memory" in str(e):
                print("Official VAE FAILED: Out of Memory")
                torch.cuda.empty_cache()
            else:
                raise e
    else:
        print("\n[Test 0] Skipped (Official VAE One-shot)")

    # 1. Tiled Streaming (Safe Mode: offload_to_cpu=True)
    latent_tile_safe = None
    time_safe = 0
    peak_tile_safe = 0
    if test_safe:
        print(f"\n[Test 1] Performing Tiled Streaming (Safe Mode, Tile={tile_size})...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.perf_counter()
        with torch.inference_mode():
            latent_tile_safe = vae_stream_encode(vae, pixels, chunk_size=chunk_size, tile_size=tile_size, overlap=overlap, offload_to_cpu=True)
        time_safe = time.perf_counter() - start_time
        
        peak_tile_safe = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Safe Mode Peak VRAM: {peak_tile_safe:.2f} MB (Time: {time_safe:.2f}s)")
    else:
        print("\n[Test 1] Skipped (Tiled Safe Mode)")

    print("\n--- Summary Table ---")
    print(f"{'Mode':<15} | {'VRAM (MB)':<12} | {'Time (s)':<10} | {'Cosine Sim':<12}")
    print("-" * 55)
    
    # Official
    if test_official and latent_off is not None:
        print(f"{'Official':<15} | {peak_off:12.2f} | {time_off:10.2f} | {'Ref'}")
    else:
        print(f"{'Official':<15} | {'SKIPPED':^12} | {'-':^10} | {'-':^12}")

    # Safe
    if test_safe and latent_tile_safe is not None:
        sim_str = "Ref" if not (test_official and latent_off is not None) else "-"
        if test_official and latent_off is not None:
            sim, _, _ = compare_latents(latent_off, latent_tile_safe)
            sim_str = f"{sim:12.6f}"
        print(f"{'Tiled Safe':<15} | {peak_tile_safe:12.2f} | {time_safe:10.2f} | {sim_str}")
    else:
        print(f"{'Tiled Safe':<15} | {'SKIPPED':^12} | {'-':^10} | {'-':^12}")

if __name__ == "__main__":

    run_tiled_stream_validation(width=720, height=1280, length=32, chunk_size=16, tile_size=512, 
                               test_official=False, test_safe=True)
