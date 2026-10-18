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

---

# 机理中文版（技术报告翻译）

## 挑战：8GB 的物理天花板

Wan 2.1 生成 720p 150 帧视频时序列长达 **54.7 万 token**。标准实现下，中间张量
（X ≈ 1.6GB，KV 缓存 ≈ 3.2GB）加上权重立即超过 8GB 显存上限，触发 Windows 共享内存
交换，性能掉 90%，生成基本不可用。

## 策略 I：把"latent 基座"卸载到 CPU

- **思路**：主 latent 张量 X 作为残差基座，访问频繁但任一时刻只有小切片参与计算。
- **做法**：序列长度超过阈值（如 65536 token）时，强制 X 驻留**系统内存（CPU）**而非显存。
- **收益**：永久省回 1.6GB 显存；前向时只有 4096 token 的"计算块"临时上传 GPU。

## 策略 II：逐帧/顺序 Patch Embedding

- **思路**：标准 3D 卷积的 patch embedding 产生巨大的 FP32 中间缓冲（720p 峰值约 3.2GB）。
- **做法**：把 embedding 改成**逐帧迭代**处理，而不是一整个大 3D 卷积。
- **收益**：embedding 峰值从 3.2GB 砍到 <100MB，"点火阶段"不再爆。

## 策略 III：两遍流式 attention（原地写回）

消除 `torch.cat` 的显存翻倍效应，专门的两遍流程：

- **Phase A（KV buffer 填充）**：在 GPU 上预分配完整 K/V buffer，**分块投影 K/V、
  直接写入预分配切片**。
- **Phase B（分块 Q attention）**：逐块取 Q（对 CPU 驻留的 X 而言是从 CPU 拉），
  对完整 KV buffer 跑 attention。
- **直接写回**：attention 投影的输出**原地直接加回** latent X。
- **收益**：唯一的大占用就是 KV buffer（3.2GB），不再产生巨大的中间"Part"张量去做拼接。

## 策略 IV：硬性显存修剪（清缓存）

- **做法**：借助 ComfyUI 的动态加载，在关键瓶颈节点显式注入 `gc.collect()` 和
  `torch.cuda.empty_cache()`（具体位置：CPU offload 之后、大块 KV 分配之前）。
- **收益**：防止 PyTorch 缓存分配器碎片化或攥着上一层的"陈旧"隐藏状态不放，
  给 3.2GB 的 KV buffer 腾出连续大块。

## 策略 V：动态空间对齐（鲁棒性）

- **问题**：视频序列空间 padding 不一致（如 1280x720 经不同 VAE/Pose padding 得到
  136px vs 135px 的 latent）。
- **做法**：分块加法逻辑里引入 **min-slicing**，自动取主 latent 与 pose latent 的
  空间维度最小交集对齐。
- **收益**：非标准长宽比下分块运算不再报 `Size Mismatch`。

## 效果总结

从"连续分配"转为"流式分块"，720p 150f 峰值显存从 8.9GB 压到 7.3GB，全程待在
显存专用上限内不触碰 Windows 交换的悬崖，瓶颈回到 CUDA 算力本身。

---

# 实现对照（wan-attention-stream 分支，2026-10-18）

| 文档策略 | 状态 | 说明 |
|---|---|---|
| III Phase B：Q 分块 attention，KV 全量驻留 | ✅ 已实现 | `5432e4e`，self_attn；`63805e9` 举一反三到 cross_attn（文档未明说 cross_attn，但同一思路） |
| III Phase A：K/V 分块投影写预分配 buffer | 🔨 进行中 | 第 4 刀（调制下放 + K/V 分块投影），数值校验未通过，调试中 |
| III 直接写回：attention 输出原地加回 | ⬜ 未做 | block 里仍是 `x = x + cross_attn(...)`，可改 `x.add_(y)` 省 210MB |
| FFN 按 token 分块 + 原地 addcmul | ✅ 已实现（文档外延） | `6b8b314`，文档没提 FFN，是同思路延伸 |
| I：主 latent X CPU 驻留（>65536 token） | ⬜ 未做 | 512×512（21504 token）不触发；高分辨率再说 |
| II：逐帧 patch embedding | ⬜ 未做 | 同上，点火阶段一次性开销，高分辨率才显著 |
| IV：注入 gc.collect/empty_cache | ⬜ 不做 | dirty hack，与 aimdo 动态加载叠加后行为难推理；先看不做能压到多少 |
| V：min-slicing 空间对齐 | ⬜ 不适用 | 补丁处理 pose latent 对齐的鲁棒性问题，我们未遇到 |


