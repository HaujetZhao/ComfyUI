# Wan 2.1 VAE encode 显存测试：构造随机视频，走 comfy 真实 encode 路径
# 用法: uv run python my/vae_encode_vram_test.py   （在仓库根目录跑）
# 安全护栏:
#   1. MEM_FRACTION（默认关）: 设了则 torch 分配超限抛 OOM → encode 自动退 tiled
#   2. 后台线程每 2 秒查设备剩余显存，低于地板值立即杀进程（防驱动静默溢出蓝屏）
# 流程: 先做一次"非分块试探"抓原始 OOM 报错（里面有待确认的巨额分配请求），再走完整 vae.encode / vae.decode
# 跑起来后盯任务管理器，随时 Ctrl+C

import os
import sys
import threading
import time

# my/ 下有 comfy/ 副本会遮住仓库的 comfy 包，强制用仓库根目录的
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== 可改参数 =====
WIDTH = 1280
HEIGHT = 720
FRAMES = 9            # 像素帧数，会按 4N+1 取整（encode 显存与帧数无关，小帧数即可）
VAE_PATH = os.path.abspath("models/vae/wan_2.1_vae.safetensors")  # 绝对路径，fast_disk/动态加载探测需要
# 服务端启动参数快照（ComfyUI 控制台 [显存日志] 启动参数 那行），照抄到这里还原行为
SERVER_ARGS: list[str] = [
    # "--disable-pinned-memory",
]
MEM_FRACTION = None   # torch 显存上限比例；None=不设（服务端也没有）。上次蓝屏靠下面的护栏兜底
FREE_FLOOR_MB = 100  # 设备剩余显存低于此值就杀进程
LOG_INTERVAL = 2.0    # 显存日志间隔（秒）
# ====================

if SERVER_ARGS:
    sys.argv = [sys.argv[0]] + SERVER_ARGS  # 必须在 import comfy 之前生效

# 复刻 main.py:79-84 的初始化，否则 aimdo 原生库不加载 → fast_disk/动态加载全部不生效。
# 必须在 comfy.utils/sd（会连带 import comfy.storage → comfy_aimdo.storage）之前执行
import comfy.cli_args as cli_args

os.environ.setdefault('MIMALLOC_PURGE_DELAY', '0')
if cli_args.enables_dynamic_vram():
    import comfy_aimdo.control
    comfy_aimdo.control.init(
        simple_vram_headroom=None if cli_args.args.reserve_vram is None else int(cli_args.args.reserve_vram * 1024**3),
        nvml_pressure=not cli_args.args.disable_nvml_pressure)

import torch

import comfy.model_management as mm
import comfy.utils
from comfy.sd import VAE

stop = threading.Event()


def log(tag: str):
    alloc = torch.cuda.memory_allocated() / 2**20
    reserved = torch.cuda.memory_reserved() / 2**20
    free, _total = torch.cuda.mem_get_info()
    print(f"[{tag}] torch占用={alloc:.0f}MB torch保留={reserved:.0f}MB 设备剩余={free/2**20:.0f}MB", flush=True)


def guard():
    """设备级显存护栏：低于地板值立即退出，防蓝屏。"""
    while not stop.is_set():
        free, _total = torch.cuda.mem_get_info()
        if free / 2**20 < FREE_FLOOR_MB:
            print(f"\n!!! 设备剩余显存 {free/2**20:.0f}MB < {FREE_FLOOR_MB}MB，杀进程防溢出", flush=True)
            os._exit(1)
        log("监测")
        stop.wait(LOG_INTERVAL)


def probe_untiled(vae, x, device):
    """复现 encode 的非分块路径，抓原始 OOM 报错文本（含 'Tried to allocate XX GiB'）。"""
    vae.first_stage_model.to(device)  # ≈ load_models_gpu 全载
    # 与 VAE.encode 相同的 reshape: (T,H,W,C) → (1,3,T,H,W)
    px = x.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
    px = px.to(vae.vae_dtype).to(device)
    log("非分块试探:输入上卡")
    try:
        with torch.inference_mode():  # 服务端在 execution.py:766 全程 inference_mode，必须对齐
            out = vae.first_stage_model.encode(px)
        print(f"[非分块试探] 成功, out={tuple(out.shape)}", flush=True)
        del out
    except torch.cuda.OutOfMemoryError as e:
        print(f"!!! 非分块试探 OOM 原文:\n{e}", flush=True)
    del px


def patch_chunk_logs(vae):
    """给 encoder/decoder 的每次前向（=每个时间块）打显存日志；顺手捕获 feat_cache"""
    captured_feat_cache = [None]
    for name in ("encoder", "decoder"):
        module = getattr(vae.first_stage_model, name, None)
        if module is None:
            continue
        orig = module.forward
        counter = [0]

        def wrapped(*a, _orig=orig, _name=name, _c=counter, _cap=captured_feat_cache, **k):
            _c[0] += 1
            if _name == "encoder" and k.get("feat_cache") is not None:
                _cap[0] = k["feat_cache"]
            out = _orig(*a, **k)
            log(f"{_name} 块{_c[0]}")
            return out

        module.forward = wrapped
    return captured_feat_cache


def predict_feat_cache_mb(width, height):
    """按实测槽位布局枚举（dim=96, 512×1024 实测=1350MB 验证通过）。
    P=像素面积; 每槽 (通道×面积×帧数)：2帧槽 [3P + 4×96P + 96P/4 + 3×192P/4 + 192P/16 + 3×384P/16]，
    1帧槽 [192P/16 + 10×384P/64]"""
    P = height * width
    two_frame = 3 * P + 4 * 96 * P + 96 * P // 4 + 3 * 192 * P // 4 + 192 * P // 16 + 3 * 384 * P // 16
    one_frame = 192 * P // 16 + 10 * 384 * P // 64
    return (2 * two_frame + one_frame) * 2 / 2**20  # ×2B dtype


def dump_cuda_tensors(tag):
    """枚举进程内所有存活 CUDA 张量，按形状分组——显存分解的实测手段"""
    import gc
    groups = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                key = (tuple(obj.shape), str(obj.dtype).replace("torch.", ""))
                g = groups.setdefault(key, [0, 0])
                g[0] += 1
                g[1] += obj.numel() * obj.element_size()
        except Exception:
            pass  # 已析构的张量
    items = sorted(groups.items(), key=lambda kv: -kv[1][1])
    print(f"\n[{tag}] 存活 CUDA 张量分组 (共{len(items)}种形状):", flush=True)
    for (shape, dtype), (cnt, nbytes) in items:
        print(f"  {shape} ×{cnt} = {nbytes/2**20:.0f}MB {dtype}", flush=True)


def report_feat_cache(fm):
    if not fm:
        print("feat_cache: 未捕获", flush=True)
        return
    tensors = [(i, t) for i, t in enumerate(fm) if torch.is_tensor(t)]
    total = sum(t.numel() * t.element_size() for _, t in tensors)
    print(f"\nfeat_cache 槽位={len(fm)} 已填={len(tensors)} 实测总计={total/2**20:.0f}MB "
          f"(预测={predict_feat_cache_mb(WIDTH, HEIGHT):.0f}MB)", flush=True)
    for i, t in tensors:
        mb = t.numel() * t.element_size() / 2**20
        if mb >= 1:
            print(f"  槽{i:2d}: {tuple(t.shape)} {mb:.0f}MB", flush=True)


RESULTS = []  # (模式, 耗时s, 峰值占用MB, 每块ms)


def run_phase(name, fn, blocks=None):
    """跑一个阶段并记录耗时/峰值占用（单轮、不预热），最后统一打印对比表"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    per_chunk = f" {dt * 1000 / blocks:.0f}ms/块" if blocks else ""
    print(f"[计时] {name}: {dt:.1f}s{per_chunk}, 峰值占用={peak:.0f}MB", flush=True)
    RESULTS.append((name, dt, peak, dt * 1000 / blocks if blocks else None))
    return out


def print_summary():
    print("\n===== 各模式对比（单轮）=====", flush=True)
    for name, dt, peak, per_chunk in RESULTS:
        per = f" {per_chunk:.0f}ms/块" if per_chunk else ""
        print(f"  {name}: {dt:.1f}s{per}, 峰值占用={peak:.0f}MB", flush=True)


def main():
    device = mm.vae_device()  # 与 VAELoader(device=None) 一致
    total = torch.cuda.get_device_properties(device).total_memory / 2**20
    if MEM_FRACTION is not None:
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, device=device)
    print(f"device={device}, 总显存={total:.0f}MB"
          + (f", torch 分配上限={total*MEM_FRACTION:.0f}MB" if MEM_FRACTION else ", 无 torch 上限(靠护栏线程)")
          + f", 启动参数={SERVER_ARGS or '无'}")

    sd = comfy.utils.load_torch_file(VAE_PATH)
    vae = VAE(sd=sd, device=device)
    print(f"vae dtype = {vae.vae_dtype}, patcher dynamic={vae.patcher.is_dynamic()}, "
          f"fast_disk={getattr(vae.patcher, 'fast_disk', '?')}")
    captured_feat_cache = patch_chunk_logs(vae)

    # comfy 的 image 格式: (T, H, W, C) float32, CPU —— 和工作流里 pose_video 一致
    x = torch.rand((FRAMES, HEIGHT, WIDTH, 3), dtype=torch.float32)
    print(f"输入像素: {tuple(x.shape)}, {x.numel()*4/2**30:.2f}GiB (CPU RAM, 不占显存)\n", flush=True)

    g = threading.Thread(target=guard, daemon=True)
    g.start()

    # 块数：像素帧按 4N+1 截断后 1+2+2...；decode 按 1+2 latent 一块
    t_px = 1 + ((FRAMES - 1) // 4) * 4
    enc_blocks = 1 + (t_px - 1) // 2
    dec_blocks = 1 + ((t_px - 1) // 4 + 1) // 2

    # run_phase("非分块 encode（整段输入上卡）", lambda: probe_untiled(vae, x, device), enc_blocks)
    # log("非分块试探结束")
    # report_feat_cache(captured_feat_cache[0])
    # dump_cuda_tensors("显存分解")

    print("\n--- 完整 vae.encode（comfy_has_chunked_io：像素驻留 CPU，逐块上卡）---")
    with torch.inference_mode():
        latent = run_phase("分块 encode", lambda: vae.encode(x), enc_blocks)
    print(f"latent shape={tuple(latent.shape)}", flush=True)
    log("encode 稳态")
    report_feat_cache(captured_feat_cache[0])

    # print("\n--- 完整 vae.decode（comfy_has_chunked_io：逐块写 output_buffer）---")
    # with torch.inference_mode():
    #     pixels = run_phase("分块 decode", lambda: vae.decode(latent), dec_blocks)
    # print(f"pixels shape={tuple(pixels.shape)} {pixels.dtype} @ {pixels.device}", flush=True)
    stop.set()
    g.join(timeout=3)
    print_summary()


if __name__ == "__main__":
    main()
