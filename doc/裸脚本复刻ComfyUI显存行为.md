# 裸脚本复刻 ComfyUI 显存行为

写独立 Python 脚本直接调 `comfy.sd.VAE` 等类测显存时，行为会和 server 跑工作流不一致
（脚本里 OOM 退 tiled、fast_disk/动态加载不生效等）。本文记录 2026-10 排查
VAE encode 显存时踩过的全部差异点和修复方法。现成脚本：`my/vae_encode_vram_test.py`。

## 差异点清单（按初始化顺序）

### 1. `my/` 下的 comfy 副本会遮住仓库的 comfy 包

Python 把脚本所在目录加进 `sys.path`，`my/comfy/`（streaming patch 副本）会先被找到。
脚本开头强制插入仓库根目录：

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### 2. 启动参数必须在 `import comfy` 之前注入

model_management 按 `sys.argv` 初始化（`--disable-pinned-memory` 等）。把 server 的
启动参数照抄进脚本、在 import comfy 前设好：

```python
SERVER_ARGS = ["--disable-pinned-memory"]
sys.argv = [sys.argv[0]] + SERVER_ARGS
```

server 的实际启动参数看控制台最开头 `[显存日志] 启动参数` 那行（cli_args.py 加的）。

### 3. aimdo 原生库初始化（顺序敏感！）

server 在 `main.py` 启动时执行 `comfy_aimdo.control.init()`，裸脚本不做这步的话：
`fast_disk=False`、权重动态加载不生效，`load_models_gpu` 行为整个不同。

```python
import comfy.cli_args as cli_args          # 轻量，只依赖 comfy.options
os.environ.setdefault('MIMALLOC_PURGE_DELAY', '0')   # Windows 下 main.py 也设这个
if cli_args.enables_dynamic_vram():        # 默认 True
    import comfy_aimdo.control
    comfy_aimdo.control.init(
        simple_vram_headroom=None,         # server: args.reserve_vram 为 None 时同
        nvml_pressure=True)                # server: not args.disable_nvml_pressure
```

**关键坑**：`comfy_aimdo/storage.py` 在 **import 时**就把 `control.lib` 拷贝进自己的
模块变量。`comfy.utils` / `comfy.sd` 会连带 import `comfy.storage → comfy_aimdo.storage`，
所以 `control.init()` 必须发生在这些 import **之前**，否则 lib 永远是 None。
（`comfy` 是命名空间包、无 `__init__.py`，先 import `comfy.cli_args` 不会拖入重模块。）

### 4. VAE 文件用绝对路径

`fast_storage()` 的 NVMe 探测对相对路径返回 None → `fast_disk=False`。
用 `os.path.abspath(...)`。验证方法：脚本打印 `patcher.fast_disk`，应为 True。

已知残留差异：脚本里 `patcher.is_dynamic()` 仍为 False（server 会走动态加载 staged
路径，`memory_management.aimdo_enabled` 只在 server 里置 True）。对显存测量的影响：
模型权重远小于空闲显存时两条路径都全量进显存，峰值一致；测大模型（UNet）时要注意
这条不完全等价。

### 5. 设备用 `mm.vae_device()`

与 `VAELoader` 的 `device=None` 路径一致。

### 6. 裸脚本里 `logging.info` 不可见

没有 handler，INFO 级别全被丢弃（server 里那些 "Model storage policy" 等日志脚本里
都看不到）。脚本自己打印用 `print(..., flush=True)`。

### 7. 梯度模式：server 全程 `torch.inference_mode()`（最重要的一条）

`PromptExecutor.execute`（execution.py:766）把整个工作流包在 `torch.inference_mode()`
里。裸脚本如果直接调 `vae.encode(x)` 而不包 inference_mode/no_grad，autograd 会为
每个时间块保留全部中间激活（实测 512×1024 每个 2 帧块 +7GB），显存线性爆炸 →
WDDM 静默溢出 → 蓝屏。**2026-10-18 的"601 帧蓝屏事件"根因就是它**，当天用护栏
线程（设备剩余归零即杀进程）当场复现并钉死。所有 VAE/模型前向都必须包：

```python
with torch.inference_mode():
    latent = vae.encode(x)
```

注意与 `torch.no_grad()` 的差别：inference_mode 下产生的张量是 inference tensor，
后续不能参与 autograd——server 语义即如此，保持一致。

## 显存观测手段

| 指标 | 用法 | 对应任务管理器 |
|---|---|---|
| torch 已分配 | `torch.cuda.memory_allocated()` | — |
| torch 保留（分配器缓存块） | `torch.cuda.memory_reserved()` | ≈ 进程专用显存 + CUDA context(~0.4GB) |
| 设备级剩余 | `torch.cuda.mem_get_info()` | 任务管理器"专用 GPU 内存"全卡视角 |

- pynvml 在 Windows 上拿不到进程级占用（`GetComputeRunningProcesses` 返回空），别用。
- 缓存分配器不主动归还，任务管理器读数是峰值驻留，不是实时占用。

## 安全护栏（显存测试脚本必备）

Windows 驱动默认开 WDDM sysmem fallback：显存不够时静默溢出到系统内存，**torch 收不到
OOM**，分配无限制增长，会把机器直接拖蓝屏。裸脚本必须带：

1. `torch.cuda.set_per_process_memory_fraction(f)`：超限抛 OOM（注意：这本身是行为差异
   —— server 没有此上限，VOE encode 峰值需要 ~5GB 保留时，6GB 的 cap 会造成假的
   OOM→tiled 回退。想完全复刻行为就别设，靠护栏线程兜底）。
2. 护栏线程：每 2 秒 `torch.cuda.mem_get_info()`，设备剩余低于地板值（1200MB）就
   `os._exit(1)`。这是唯一能拦住 cuDNN workspace 等绕过 torch 分配器的开销的手段。

## 已知数据（Wan 2.1 VAE encode, 151帧@512×1024, bf16, 8GB 卡）

非分块 encode（tiled 之前的原始路径）在 inference_mode 下实测成功，账本能完全闭合：

| 分项 | 大小 | 验证方式 |
|---|---|---|
| 权重 | 241MB | server 日志 "241MB Staged" |
| 输入张量 | 453MB | 697−241 实测差值 ✓ |
| feat_cache（28槽24填） | 1350MB | 逐槽枚举实测 = 结构公式 ✓ |
| 输出 latent | 19MB | shape×dtype |
| **合计** | **2063MB** | vs 实测稳态占用 2059MB ✓ |

feat_cache 公式（可按分辨率预算）：`1350MB × (H·W)/(512×1024)`（720p≈2.4GB）。
保留值 4890MB 是分配器给"计算中峰值激活"留的余量；逐块日志在块结束后打，
看不到瞬时激活。

- 帧数线性项只有输入张量：`T×3×H×W×2` 字节，tiled 与否都一样（tiled 只分空间）。
- 显存分解工具：脚本 `dump_cuda_tensors()` 用 gc 枚举存活 CUDA 张量按形状分组；
  注意 view 共享存储会重复计数（cache_x 是切片 view），真实账以 allocated 为准。
