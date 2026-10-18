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

# 实现对照（wan-attention-stream 分支，2026-10-18 完结）

## 最终落地清单（7 刀全部完成）

| # | 改进 | 提交 | 720p 实测收益 |
|---|---|---|---|
| 1 | 基线 + 逐层探针（double_block patch + 阶段探针） | `0601e08` | — |
| 2 | Q 分块 attention（III Phase B），K/V 全量驻留 | `5432e4e` | 省全量 q/输出 |
| 3 | cross-attention q 分块（含 img 分支） | `63805e9` | 1588→1384MB |
| 4 | FFN 按 token 分块 + 原地 addcmul | `6b8b314` | 2364→1104MB |
| 5 | 调制下放 + K/V 分块投影写 buffer（III Phase A，两轮循环） | `4585a83` | 1498→1328MB |
| 6 | 残差原地写回（addcmul_/add_） | `8f80dc0` | 各段再省 210MB |
| 7 | 策略 III 完全体：消灭 y buffer，attention 输出逐块直接写回 x | `751bd66` | self -680 / cross -699MB |

## 分辨率档位最终数据（块级峰值，torch allocated 口径）

| 段 | 512×512 (21504 tok) | 1024×512 (43008 tok) | 1280×720 (75600 tok) |
|---|---|---|---|
| self_attn | 1328→**1118** | 2326→~1900 | 3831→**3151** |
| cross_attn | 1174→**964** | 1963→~1540 | 3149→**2450** |
| ffn | 894 | 1262 | 1879 |
| 耗时/层 | 1.12s | 2.89s | 7.3s（无损） |

（→ 后为第 6/7 刀的推算值；720p 为实测。）

## Server 实测（2026-10-18，512×512×81，6 步）

- CUDA 全程 90%~100% 平稳，专用+共享显存合计 <7GB，画质逐帧对比优化前正常。
- 设备剩余日志曾探底 0MB 但 torch 只占 ~1GB——WDDM 把权重 mmap 页计入设备占用的
  记账口径，无性能影响，不用管。

## 逐刀速度基准（2026-10-18，512×512×151，worktree 逐提交实测）

方法：独立 worktree 按提交历史逐个检出，跑 6 层取第 4/5/6 层平均耗时
（脚本自退，见 `e:/ComfyUI-bench/bench_run.py`）。单轮无 warmup，噪声 ±0.03s，
单刀之间的差异在噪声内，可靠的结论是**基线 vs 终态**：

| 提交 | 累计状态 | 耗时/层 | vs 基线 |
|---|---|---|---|
| `0601e08` | 基线（全量路径） | **1.08s** | — |
| `5432e4e` | +Q 分块 | 1.12s | +3.7% |
| `6b8b314` | +FFN 分块 | 1.13s | +4.6% |
| `63805e9` | +cross 分块 | 1.10s | +1.9% |
| `4585a83` | +调制下放/KV buffer | 1.13s | +4.6% |
| `8f80dc0` | +残差原地写回 | 1.17s | +8.3%（重跑 1.16-1.22，偏移最大但仍在噪声边缘） |
| `751bd66` | +消灭 y buffer（终态） | **1.12s** | **+3.7%** |

结论（正确框架）：**全部收益都是 O(S) 线性瞬时激活项，O(S²) 的 attention 主导项
一分未动**。每刀省的都是若干份 [B,S,C] 瞬时张量（@21504 token 每份 210MB，
随 token 数线性放大）；attention 始终对全量 K/V 做（softmax 分母完整），
2·S²·C 的计算量与 O(S·C) 的 KV 驻留没有减。所以 512 上"+4% 时间换 -54% 峰值"
的比例不能外推——高分辨率下省的绝对量按 S 线性增长（720p 每份 [B,S,C]=738MB），
而 512 基线本就是"刚好能跑"的临界点，高分辨率基线直接不可行，没有对照可言。

## 数学等价性与计算量审计（逐提交）

| 提交 | 等价性 | FLOPs 变化 | 省的显存 | 类型 |
|---|---|---|---|---|
| `5432e4e` self Q 分块 | 严格等价（fp 舍入序可能不同） | 0 | q/attn输出/o 三份 [B,S,C] 瞬时 | O(S) |
| `63805e9` cross q 分块 | 严格等价 | 0 | q/两次 attn 输出等 ~5 份瞬时 | O(S) |
| `6b8b314` FFN 分块 | 严格等价（实测 7.4e-4 = fp16 舍入） | 0 | 调制副本 + [B,S,13824] 隐层 + 输出 | O(S) |
| `4585a83` 调制下放两轮 | 严格等价（modulate 确定性重算，逐比特相同） | **+~1e-4**（modulate 多算一遍 ≈10·S·C，占单层 1e13 的万分之一） | 省全量调制副本；代价 +~420MB/层 LayerNorm 读写 | O(S)↔O(S) 交换 |
| `8f80dc0` 原地写回 | 逐比特等价 | 0 | 两处各一份 [B,S,C] 相和 | O(S) |
| `751bd66` 直接写回 | 严格等价（残差逐块独立，不跨块） | 0 | self/cross 各一份全量 y buffer（写+读 ~420MB 流量） | O(S) |

要点：

1. **没有任何一刀引入真近似**。全部是"数学等价、浮点舍入序可能不同"；7.4e-4
   实测误差是 fp16 GEMM 分块后走不同 kernel 的舍入序差异，不是丢项。
2. **唯一增加的计算**是 `4585a83` 两轮循环里 modulate 重算一遍（正确性必需：
   单轮时第 i 块 attention 读到的 k_buf 后半还是垃圾），量级 ~1e-4 占比。
3. 单层 FLOPs 底数（S=21504, C=5120, F=13824）：qkvo 4SC²≈2.3e12 + attention
   2S²C≈4.7e12 + FFN 2SCF≈3.1e12 ≈ 1e13 MAC，attention 占一半——这就是为什么
   砍线性激活不伤速度。

显存贡献（各刀当时实测，512 档）：FFN 分块 -1260（2364→1104）> 调制下放/KV
buffer -170（1498→1328）≈ 消灭 y buffer -210（推算）≈ 残差原地写回 -210×2
（cross/ffn 段）> cross 分块 -204（1588→1384）。

注意：PR 里不要拿本文数值当"性能提升"卖点——基线（未优化）只在 512×512 能全量
跑通，720p/1024 宽没有可对比的基线数据；本文记录的是**显存可行性**优化，非提速。

## 未做（对照文档）

- **策略 I**（x 驻留 CPU，>65536 token）：720p 已触发阈值但实测够用；做的话所有逐块
  读 x 变 H2D、写回变 D2H，速度与 RAM 都是代价（aimdo 的 RAM 流失还没查清，不往 RAM 上堆）。
- **策略 II**（逐帧 patch embedding）：点火阶段一次性开销，需要时再做。
- **策略 IV**（gc/empty_cache 注入）：dirty hack，不做。
- **策略 V**（min-slicing）：未遇到对齐问题，不适用。

## 踩坑记录

- **K/V 分块投影必须两轮循环**：self-attention 每个 q 块要对全量 K/V 做 attention，
  单轮合并时第 i 块 attention 读到的 k_buf 后半还是 `torch.empty` 垃圾（校验误差 0.48
  当场抓出）。3 月补丁分两轮正是为此。
- `AttentionTensorContainer` 是一次性消费壳（`take()` 后标记 consumed），k/v 循环多轮、
  i2v 的 q 同块喂两次，都要在循环内重建零拷贝薄壳。
- 残差原地写回后，block 会原地改输入 x——数值校验脚本同一批张量跑两遍会互相污染，
  每次前向必须 clone 输入。
- WanAnimateToVideo 的 latent 总帧数 = 1（参考图 latent）+ 内容 latent，context window
  报的帧数比 `(length-1)/4+1` 多 1 是正常的，ref 帧由 retain_first_frame 保留、
  trim_latent 在采样后裁掉。


