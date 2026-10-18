# Wan 2.1 VAE decode 显存专测：随机 latent 直接进 decoder，专盯 decode 的显存构成
# 用法: uv run python my/vae_decode_vram_test.py   （在仓库根目录跑）
# 安全护栏: 后台线程每 2 秒查设备剩余显存，低于地板值立即杀进程（防驱动静默溢出蓝屏）
# 流程: 载入 VAE → 权重上卡记录基线 → 随机 latent 分块 decode（逐块日志）→
#       稳态时枚举全部存活 CUDA 张量（分解 feat_cache / 被钉住的父张量 / 瞬态）
# 跑起来后盯任务管理器，随时 Ctrl+C

import os
import sys
import threading
import time

# my/ 下有 comfy/ 副本会遮住仓库的 comfy 包，强制用仓库根目录的
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== 可改参数 =====
WIDTH = 512
HEIGHT = 512
LATENT_FRAMES = 6    # latent 帧数 → 解出 (38-1)*4+1 = 149 像素帧
VAE_PATH = os.path.abspath("models/vae/wan_2.1_vae.safetensors")
FREE_FLOOR_MB = 300   # 设备剩余显存低于此值就杀进程
LOG_INTERVAL = 2.0    # 显存日志间隔（秒）
DUMP_BLOCKS = ()      # 在第 N 个 decode 块结束后枚举存活张量（A/B 计时跑不需要）
RUN_BLOCKS = 99       # 跑满 N 块后跳出（99=跑完整段，输出完整才能做 A/B 等价对比）
HONOR_ESTIMATE = False  # True=估算超空闲时拒绝实测（防 WDDM 共享显存爬行）；False=强闯 untiled
MODES = ("gpu", "ring2", "ringasync2")  # gpu=显存；ring2=RAM+同步写回；ringasync2=RAM+SeedVR 单在途异步写回
# ====================

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
    """设备级显存护栏：低于地板值立即退出，防蓝屏。静默值守，不刷日志。"""
    while not stop.is_set():
        free, _total = torch.cuda.mem_get_info()
        if free / 2**20 < FREE_FLOOR_MB:
            print(f"\n!!! 设备剩余显存 {free/2**20:.0f}MB < {FREE_FLOOR_MB}MB，杀进程防溢出\n"
                  f"    现场: torch占用={torch.cuda.memory_allocated()/2**20:.0f}MB "
                  f"torch保留={torch.cuda.memory_reserved()/2**20:.0f}MB", flush=True)
            os._exit(1)
        stop.wait(LOG_INTERVAL)


class StopDecode(Exception):
    """跑到预设块数后从 decoder 内部跳出，省时间（稳态与总块数无关）"""


def feat_cache_mb(fm):
    if not fm:
        return 0.0
    return sum(t.numel() * t.element_size() for t in fm if torch.is_tensor(t)) / 2**20


def dump_cuda_tensors(tag):
    """枚举进程内所有存活 CUDA 张量，按形状分组——显存分解的实测手段。
    注意：视图不重复计存储，numel 统计的是视图自身，被钉住的父张量会以原始大形状出现在列表里。"""
    import gc
    groups = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.untyped_storage().nbytes() > 0:
                # 按底层存储分组才能看到"被视图钉住的父张量"：同一 storage 多个视图只算一次
                st = obj.untyped_storage().data_ptr()
                g = groups.setdefault(st, {"shapes": set(), "bytes": obj.untyped_storage().nbytes(),
                                           "views": 0})
                g["views"] += 1
                g["shapes"].add(tuple(obj.shape))
        except Exception:
            pass  # 已析构的张量
    items = sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])
    total = sum(g["bytes"] for g in groups.values())
    print(f"\n[{tag}] 存活 CUDA 存储 (共{len(items)}个, 合计≈{total/2**20:.0f}MB):", flush=True)
    for st, g in items[:40]:
        if g["bytes"] / 2**20 < 0.5:
            break
        shapes = " | ".join(str(s) for s in list(g["shapes"])[:3])
        print(f"  {g['bytes']/2**20:7.0f}MB  视图×{g['views']:3d}  {shapes}", flush=True)


INSTR = {"block": 0, "call": 0, "rows": None}


def patch_instruments(vae):
    """块级：显存 + feat_cache + 块内瞬态；逐层：每个 CausalConv3d 的进出显存轨迹
    （只在最后一块打印全表）。轨迹呈锯齿=逐层回收正常；阶梯式上涨=有东西没释放。"""
    from comfy.ldm.wan.vae import CausalConv3d

    for m in vae.first_stage_model.decoder.modules():
        if isinstance(m, CausalConv3d):
            orig = m.forward

            def conv_wrapped(x, *a, _orig=orig, **k):
                INSTR["call"] += 1
                entry = torch.cuda.memory_allocated() / 2**20
                out = _orig(x, *a, **k)
                if INSTR["rows"] is not None:
                    INSTR["rows"].append((INSTR["call"], tuple(x.shape),
                                          entry, torch.cuda.memory_allocated() / 2**20))
                return out

            m.forward = conv_wrapped

    module = vae.first_stage_model.decoder
    orig = module.forward
    counter = [0]
    captured = [None]
    global_peak = [0.0]  # 手动累计全局峰值（块内 reset 会清掉 max_memory_allocated 的历史）

    def block_wrapped(*a, _orig=orig, _c=counter, _cap=captured, _peak=global_peak, **k):
        _c[0] += 1
        _cap[0] = k.get("feat_cache")
        INSTR["block"] = _c[0]
        INSTR["call"] = 0
        INSTR["rows"] = [] if _c[0] == RUN_BLOCKS else None
        entry = torch.cuda.memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
        out = _orig(*a, **k)
        inblock_peak = torch.cuda.max_memory_allocated() / 2**20
        _peak[0] = max(_peak[0], inblock_peak)
        # 稳态与块号无关：只报前 2 块（建缓存期），块4 后分解，最后一块跳出+打逐层轨迹
        if _c[0] <= 2 or _c[0] == RUN_BLOCKS:
            reserved = torch.cuda.memory_reserved() / 2**20
            fc = feat_cache_mb(_cap[0])
            print(f"[decoder 块{_c[0]:02d}] torch占用={entry:.0f}MB torch保留={reserved:.0f}MB "
                  f"feat_cache={fc:.0f}MB 块内瞬态={inblock_peak - entry:.0f}MB", flush=True)
        if INSTR["rows"]:
            print(f"\n[块{_c[0]} 逐层显存轨迹] #序号 输入形状 进出MB(净变化):", flush=True)
            for call, shape, e, x_ in INSTR["rows"]:
                print(f"  #{call:02d} {str(shape):28s} {e:6.0f} -> {x_:6.0f} ({x_ - e:+5.0f})", flush=True)
        if _c[0] in DUMP_BLOCKS:
            dump_cuda_tensors(f"块{_c[0]}后分解")
        if _c[0] >= RUN_BLOCKS:
            raise StopDecode
        return out

    module.forward = block_wrapped
    return captured, global_peak


def main():
    device = mm.vae_device()
    total = torch.cuda.get_device_properties(device).total_memory / 2**20
    print(f"device={device}, 总显存={total:.0f}MB, 分辨率={WIDTH}x{HEIGHT}, latent帧={LATENT_FRAMES}")

    sd = comfy.utils.load_torch_file(VAE_PATH)
    vae = VAE(sd=sd, device=device)
    print(f"vae dtype = {vae.vae_dtype}")
    vae.first_stage_model.to(device)
    torch.cuda.empty_cache()
    log("权重上卡后基线")

    captured, global_peak = patch_instruments(vae)

    # 方向1验证：wrapper 选路依据 = memory_used_decode 估算 vs 空闲显存（comfy_decode_estimate_is_reliable）
    est = vae.memory_used_decode((1, 16, LATENT_FRAMES, HEIGHT // 8, WIDTH // 8), vae.vae_dtype)
    free = vae.patcher.get_free_memory(device)
    will_tile = est > free
    print(f"wrapper 估算 decode 需 {est/2**30:.2f}GiB, 当前空闲 {free/2**30:.2f}GiB → 选路: "
          f"{'tiled' if will_tile else 'untiled'}"
          + ("" if will_tile else f"（{WIDTH}x{HEIGHT} untiled 预期峰值 ≈ {est*4265/5500/2**20:.0f}MB）"),
          flush=True)
    if will_tile and HONOR_ESTIMATE:
        # 本脚本只测 untiled 路径。估算超空闲时 wrapper 会走 tiled；硬闯 untiled 会掉进
        # WDDM 共享显存（无 OOM、速度掉两个数量级）。要强闯（如测 feat_cache 上卡方案）设 HONOR_ESTIMATE=False
        print("估算超过空闲显存，wrapper 实际会走 tiled；untiled 实测已无意义，退出", flush=True)
        return

    # 随机 latent：显存分配只由形状决定，值无所谓
    latent = torch.rand((1, 16, LATENT_FRAMES, HEIGHT // 8, WIDTH // 8), dtype=torch.float32)
    px_frames = (LATENT_FRAMES - 1) * 4 + 1
    dec_blocks = 1 + LATENT_FRAMES // 2
    print(f"latent={tuple(latent.shape)} (cpu) → 预期 pixels 帧数={px_frames}, decode 块数={dec_blocks}\n", flush=True)

    g = threading.Thread(target=guard, daemon=True)
    g.start()

    # 直接调 first_stage_model.decode，绕开 VAE wrapper（本分支 wrapper 还会整段视频驻留显存，
    # 污染对 decoder 本身的测量）。output_buffer 放 CPU，等价 chunked_io 路径的直写行为
    import comfy.ldm.wan.vae as wan_vae
    z = latent.to(device=device, dtype=vae.vae_dtype)
    buffer = torch.empty((1, vae.output_channels, px_frames, HEIGHT, WIDTH),
                         device="cpu", dtype=vae.vae_dtype)

    results = {}
    for mode in MODES:
        wan_vae.RAM_CACHE = mode.startswith("ring")
        wan_vae.RING_ASYNC = mode.startswith("ringasync")
        wan_vae.RING_R = 2
        global_peak[0] = 0.0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():  # 服务端在 execution.py 全程 inference_mode，必须对齐
            pixels = vae.first_stage_model.decode(z, output_buffer=buffer)
        dt = time.perf_counter() - t0
        peak = global_peak[0]  # 埋点累计的全程峰值（max_memory_allocated 被逐块 reset，只反映末块）
        results[mode] = (dt, peak, pixels.clone())
        print(f"[计时] {mode}: {dt:.1f}s {dt*1000/dec_blocks:.0f}ms/块, 峰值占用={peak:.0f}MB", flush=True)

    base, (dt0, peak0, px0) = MODES[0], results[MODES[0]]
    print(f"\n[A/B 对比] 基准={base}")
    for mode in MODES[1:]:
        dt, peak, px = results[mode]
        d = (px.float() - px0.float()).abs().max().item()
        print(f"  {mode:10s}: {dt:.2f}s ({(dt/dt0-1)*100:+5.1f}%) 峰值{peak:.0f}MB "
              f"与基准最大逐元素差={d:.2e} {'(位精确)' if d == 0 else ''}", flush=True)
    print(f"  {base:10s}: {dt0:.2f}s 峰值{peak0:.0f}MB (基准)", flush=True)
    log("A/B 结束后")
    print(f"feat_cache 终态: {feat_cache_mb(captured[0]):.0f}MB", flush=True)

    stop.set()
    g.join(timeout=3)


if __name__ == "__main__":
    main()
