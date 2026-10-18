Low CUDA efficiency with WanVideoSampler and solution.



I noticed that when my workflow runs to WanVideoSampler, even though my GPU memory is not fully occupied, the CUDA section in Task Manager always shows a low usage rate, indicating significant memory swapping. This might be due to the optimization design in the project.

<img width="800"  alt="Image" src="https://github.com/user-attachments/assets/05020e6e-b6fe-4b31-a209-93278442428f" />

Additionally, when I tested the generation of 720p 30frames content, the GPU memory usage reached 6GB.

I think this might mean that the optimization design in the Attention section cannot fully utilize the CUDA computing power, nor can it meet the requirements for generating 720P 100frames content on 8GB of GPU memory. More effective optimization of Attention calculations to reduce memory usage is needed.

I only came across this node this afternoon, and it works great. While this morning, I just made improvements to the attention part in the official `comfy\ldm\wan\model.py` of ComfyUI. I kept only the complete K and V in memory and performed chunked attention computation for Q, and the results were excellent. When sampling at 720P 120 frames, it only occupied 4GB of GPU memory, and CUDA also maintained high efficiency. With 6 steps of sampling on my RTX5050, the total sampling time was 90 minutes. Due to the quadratic increase in computational cost for global attention, it can be calculated that if used for an 81-frame context, each step would only take 7 minutes. (81^2 / 120^2 * 90 / 6)

However, this project is a bit too complex for me, and I couldn’t find the right place to make improvements in a short time. So, I’ve uploaded the code I implemented for `comfy\ldm\wan\model.py` for your reference, hoping that such memory-saving and CUDA-efficient methods can be applied.

[StreamingVAE-and-StreamlinedAttention.zip](https://github.com/user-attachments/files/26039896/StreamingVAE-and-StreamlinedAttention.zip)


The attention optimization inside was completed with the help of Gemini, so the code might not be very formal, but the results are excellent. I’ll first have the AI summarize the logic within it, so you can review it before downloading to determine if it’s valuable:

> ### Technical Report: VRAM Optimization for Wan 2.1 Long-Sequence Generation
>
> #### 1. The Challenge: The 8GB "Physical Ceiling"
>
> Wan 2.1 generates sequences of up to **547k tokens** for a 720p 150-frame video. In a standard implementation, the intermediate tensors ($X \approx 1.6\text{GB}$, $\text{KV Cache} \approx 3.2\text{GB}$) combined with model weights immediately exceed the 8GB VRAM limit. This triggers Windows Shared Memory swapping, causing a 90% performance drop and rendering the generation practically useless.
>
> #### 2. Strategy I: Offloading the "Latent Bedrock" to CPU
>
> - **Concept**: The main latent tensor ($X$) is used as a residual base. It is frequently accessed but only small slices participate in active computation at any given moment.
> - **Implementation**: We force the residency of $X$ in **System RAM (CPU)** rather than VRAM when the sequence length exceeds a specific threshold (e.g., 65,536 tokens).
> - **Result**: Permanently reclaimed **1.6GB** of VRAM. Only 4096-token "Compute Chunks" are uploaded to the GPU temporarily during the forward pass.
>
> #### 3. Strategy II: Sequential/Frame-based Patch Embedding
>
> - **Concept**: Standard 3D convolutions for Patch Embedding create massive FP32 intermediate buffers (peaking at ~3.2GB for 720p).
> - **Implementation**: Refactored the embedding layer to process the video **frame-by-frame** (iteratively) instead of one large 3D convolution.
> - **Result**: Smashed the embedding peak from **3.2GB down to <100MB**, ensuring the "ignition phase" of the model stays within safe memory margins.
>
> #### 4. Strategy III: Two-Pass Streamlined Attention (In-place)
>
> To eliminate the memory-doubling effect of `torch.cat`, we implemented a specialized two-pass flow:
>
> - **Phase A (KV Buffer Filling)**: Pre-allocate the full K/V buffers on GPU. Project K and V in chunks and write directly into the pre-allocated slice.
> - **Phase B (Chunked Q-Attention)**: Fetch the Query from the CPU-resident latent $X$ in chunks. Run the Attention op against the full KV buffer.
> - **Direct Writeback**: The output of the Attention projection is added **directly back** to the CPU-resident latent $X$ via in-place operations.
> - **Result**: The only major occupancy is the **KV Buffer (3.2GB)**. No massive intermediate "Part" tensors are created or concatenated.
>
> #### 5. Strategy IV: Hard Memory Trimming (Cache Eradication)
>
> - **Implementation**: Leveraging ComfyUI's dynamic loading, we injected explicit `gc.collect()` and `torch.cuda.empty_cache()` calls at critical bottleneck junctions (specifically right after CPU offloading and before the heavy KV allocation).
> - **Benefit**: Prevented PyTorch's `CachingAllocator` from fragmenting or holding onto "stale" memory that belongs to the previous layer's hidden states, ensuring a contiguous block for the 3.2GB KV buffer.
>
> #### 6. Strategy V: Dynamic Spatial Alignment (Robustness)
>
> - **Issue**: Video sequences often have inconsistent spatial padding (e.g., 1280x720 might result in 136px vs 135px latents due to different VAE/Pose padding).
> - **Implementation**: Introduced **Min-Slicing** in the chunked addition logic. It automatically aligns the main latent and pose latent by taking the minimum intersection of their spatial dimensions.
> - **Result**: Eliminated `Size Mismatch` RuntimeErrors during chunked operations for non-standard aspect ratios.
>
> ------
>
> ### Summary of Impact
>
> By shifting from "Continuous Allocation" to **"Streamed Chunking,"** we compressed the peak VRAM for 720p 150f from **8.9GB to 7.3GB**.
>
> Crucially, this allows the process to stay within the **Hardware VRAM Dedicated Limit**, avoiding the "Cliff Edge" of Windows system swapping and moving the computational bottleneck back to the CUDA cores where it belongs.

---

Using this method, the sampler would only use 4.5G VRAM when K-sampleing for 720P 120 frames, with quit good CUDA efficiency: 

<img width="800"  alt="Image" src="https://github.com/user-attachments/assets/cfbafbc5-9d58-4bcd-a4e1-8487966cc187" />

---

I thought `ComfyUI-WanAnimatePreprocess`'s `VideoVAE_` already handled VAE chunk encoding, so there shouldn't be any VRAM issues. However, last night when generating a 30-second 480P video, I encountered insufficient VRAM during the VAE Encode step. That means I'll have to bring in the streaming and chunked VAE implementation I developed for `WanHuMoImageToVideo` as well.

[StreamingVAE-and-StreamlinedAttention.zip](https://github.com/user-attachments/files/26039896/StreamingVAE-and-StreamlinedAttention.zip)

These two implementation would allow 8G VRAM to run 720P and 1080P video, no longer limited by VRAM space, making cuda computation most busy and efficient. 

@kijai  
