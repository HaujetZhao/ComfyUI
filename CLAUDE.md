# ComfyUI 环境说明

## Python 环境

- 用 `uv` 管理，`.venv` 在仓库根目录，跑脚本用 `uv run`。
- torch 全家桶版本（2026-10-16 定版，勿随意升级）：
  - `torch==2.12.0+cu130`
  - `torchvision==0.27.0+cu130`
  - `torchaudio==2.11.0+cu130`
  - 安装索引：`--index-url https://download.pytorch.org/whl/cu130`
- 原因：torch 2.13/2.14 没有 Windows 预编译 flash-attn wheel，2.12 是社区 wheel 能覆盖的最高版本。

## GPU

- NVIDIA GeForce RTX 5050 Laptop GPU，8GB，Blackwell 架构（sm_120 / cc 12.0）。
- 选 wheel/编译目标时要带 sm_120（CUDA 12.8+ 工具链用 `compute_120f`）。

## flash-attn

- 已装 `flash_attn 2.8.4`，来源：wildminder/AI-windows-whl 的 Windows 预编译 wheel（cu130 + torch 2.12.0 + cp313，含 sm_120，已验证数值正确）。
- wheel 仓库：https://huggingface.co/Wildminder/AI-windows-whl （含 xformers、triton、NATTEN 等 Windows wheel 索引）
- WanVideoWrapper 里对应 `attention_mode` 选 `flash_attn_2`。

## 不用的东西

- **sageattention：不装**。省显存有限（只靠低精度减半），不省 attention 中间显存。
- flash-attn 不要升级到 torch 2.13/2.14 对应版本再折腾——等社区有 wheel 再说。
- 根目录 `constraints.txt`（pin 旧版 torch 2.14+cu132）已过时，忽略它。

## 显存调试

- **模型前向必须包 `torch.inference_mode()`**：server 在 execution.py 把整个工作流包在里面，裸测试脚本不包的话 autograd 会逐块保留全部激活（实测每 2 帧块 +7GB），直接爆显存蓝屏。`torch.no_grad()` 不等价（inference tensor 语义）。
- 裸脚本复刻 server 显存行为的完整清单（aimdo 初始化顺序、sys.argv 注入、WDDM 溢出护栏等）：[doc/裸脚本复刻ComfyUI显存行为.md](doc/裸脚本复刻ComfyUI显存行为.md)，现成脚本 `my/vae_encode_vram_test.py`。
- 仓库内临时加了 `[显存日志]` 埋点（cli_args.py 启动参数、execution.py 节点级、comfy/sd.py VAE encode/decode、comfy/ldm/wan/vae.py 时间块级），查完显存用 `git diff` 定位还原。

## 第三方节点中文翻译

- 给 custom_nodes 插件做界面翻译的规范（文件格式、下划线键、校验）：[doc/自定义节点界面翻译规范.md](doc/自定义节点界面翻译规范.md)。官方前端源码在 `E:\repos\ComfyUI_frontend`。

## StreamingVAE + StreamlinedAttention 补丁（2026-10 移植）

- 分支 `streaming-vae-attention-0310`，基线 `535c16c`（2026-03-09/10），提交 `68992df`。源自 issue #1969 的 zip 补丁（`my/` 下有原件）。
- 内容：分块 patch embedding、CPU latent 驻留（>65536 token）、Q 分块 attention（KV 全量驻留显存）、`WanVAEStreamEncoder` 流式 VAE 编码。
- **必须用 `--disable-pinned-memory` 启动**：新版 comfy 的 pinned 权重缓存上限是 RAM 的 45%（32GB 机器 = 14.5GB），配合 aimdo 动态加载会把系统 RAM 压到 99% 后 native abort（`Fatal Python error: Aborted`，死在 `cast_to` 的 H2D 拷贝）。
- 踩坑记录：
  - `wan_vae_stream.py` 的 `count_conv3d` 必须从 `comfy.ldm.wan.vae` 导入（老版布局）；`035414e`（3-17）重写了 vae.py 流式语义并删了这个函数，所以基线必须早于它。
  - 显存/RAM 泄漏排查套路：在补丁的 `print_vram` 里加 `Pinned / GPUCommit / RAMAvail` 三个量（`wan_attention_stream.py`，未提交），谁单调涨谁就是泄漏池；torch `Allocated` 平的不代表没事，问题都在 torch 之外的驱动层池子。
  - 2026-10-19 定论：pinned 关掉后 RAM 缓慢增长不是泄漏，是 aimdo prefetch 的 vbar mmap 页，系统内存不够时会自动缩小，不用管（更不用试 `--disable-dynamic-vram`）。
- 旧版 ComfyUI 配新版数据库会报 alembic `ResolutionError: 0003_add_metadata_job_id`（新版把 user 库升过级了），只影响资产库功能，不影响生成，忽略。
