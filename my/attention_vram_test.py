# Wan2.2-Animate-14B 采样显存测试：构造随机 conds，走 comfy 真实 KSampler + context windows 路径
# 用法: uv run python my/attention_vram_test.py   （在仓库根目录跑，盯任务管理器看 CUDA 锯齿）
# 目的: 复刻 server 采样行为，把 1024×512 时 CUDA 0%~100% 锯齿的根因钉死
#       （假设: 激活翻倍挤占 lowvram 权重预算 → aimdo 逐层流送权重 → PCIe 搬运与计算交替）
# 安全护栏: 后台线程每 2 秒查设备剩余显存，低于地板值立即杀进程（防 WDDM 静默溢出蓝屏）

import os
import sys
import threading

# my/ 下有 comfy/ 副本会遮住仓库的 comfy 包，强制用仓库根目录的
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== 可改参数 =====
WIDTH = 1024        # 像素宽，8 的倍数
HEIGHT = 512        # 像素高
FRAMES = 151        # 目标像素帧数，latent 帧数 = (FRAMES-1)//4+1
MODEL_PATH = os.path.abspath("models/unet/Animate/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors")
# 工作流同款 LoRA。注意：脚本加载时空闲显存比 server 大（server 先 stage 了 TE/CLIP/Vision），
# 预算宽裕时 patch 会被物化进显存（torch 直占 3.3GB），和 server 的流送行为分叉。
# LoRA 不影响 attention 激活尺寸，测显存就默认不挂
LORAS = [
    # (os.path.abspath("models/loras/Animate/WanAnimate_relight_lora_fp16.safetensors"), 1.0),
    # (os.path.abspath("models/loras/Animate/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"), 1.0),
]
# context windows（照抄工作流 WanContextWindowsManual 节点: 81/30/standard_static/pyramid/retain_first_frame）
CONTEXT_FRAMES = 81     # 窗口像素帧数，4N+1
CONTEXT_OVERLAP = 30    # 窗口重叠像素帧数
STEPS = 2               # 锯齿单步内就会出现，2 步够看
SEED = 888
# 服务端启动参数快照（ComfyUI 控制台 [显存日志] 启动参数 那行），照抄到这里还原行为
SERVER_ARGS: list[str] = [
    "--disable-pinned-memory",   # server 必带（CLAUDE.md），否则 pinned 缓存被 WDDM 计进"共享 GPU 内存"一路涨
    # "--disable-dynamic-vram",
]
FREE_FLOOR_MB = 100  # 设备剩余显存低于此值就杀进程（文档建议值，给杀进程留余量）
RAM_FLOOR_MB = 2000   # RAM可用低于此值就杀进程（WDDM 溢出的终点是拖垮系统）
LOG_INTERVAL = 2.0    # 显存日志间隔（秒）
# ====================

if SERVER_ARGS:
    sys.argv = [sys.argv[0]] + SERVER_ARGS

# 让 logging.info 可见（文档差异点 6）：server 加载路径的关键日志
# （Model storage policy / prepared ... Staged / aimdo 原生）全走 logging.info，裸脚本默认丢弃
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

# 关键！main.py:2 会开参数解析，裸脚本不开的话 cli_args 走 parse_args([])，所有 SERVER_ARGS 被静默丢弃
import comfy.options

comfy.options.enable_args_parsing()

# 复刻 main.py:270-312 的动态显存初始化。必须齐全，缺任何一步权重都不走 aimdo 动态流送：
# control.init → init_devices → 替换 CoreModelPatcher → aimdo_enabled=True。
# control.init 必须在 comfy.utils/sd（连带 import comfy_aimdo.storage，import 时抓 lib）之前执行
import comfy.cli_args as cli_args

os.environ.setdefault('MIMALLOC_PURGE_DELAY', '0')
if cli_args.enables_dynamic_vram():
    import comfy_aimdo.control
    comfy_aimdo.control.init(
        simple_vram_headroom=None if cli_args.args.reserve_vram is None else int(cli_args.args.reserve_vram * 1024**3),
        nvml_pressure=not cli_args.args.disable_nvml_pressure)
    import comfy.model_management as mm
    try:
        aimdo_initialized = comfy_aimdo.control.init_devices(
            (d.index, int(cli_args.args.vram_headroom * 1024**3)) for d in mm.get_all_torch_devices())
    except TypeError:  # comfy-aimdo 0.4.9 协议
        aimdo_initialized = comfy_aimdo.control.init_devices(d.index for d in mm.get_all_torch_devices())
    if aimdo_initialized:
        import comfy.model_patcher
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
        print("[初始化] aimdo 动态显存已启用", flush=True)
    else:
        print("[初始化] init_devices 失败，退回 legacy ModelPatcher（16GB 权重会硬塞显存）", flush=True)

import torch

import comfy.context_windows
import comfy.sd
import comfy.sample
import comfy.utils
import comfy.model_management as mm
from types import SimpleNamespace

stop = threading.Event()
_last = {"alloc": -1e9, "reserved": -1e9, "free": -1e9}


def log(tag: str, force=False, min_delta_mb=64):
    import psutil
    proc = psutil.Process()
    alloc = torch.cuda.memory_allocated() / 2**20
    reserved = torch.cuda.memory_reserved() / 2**20
    free, _total = torch.cuda.mem_get_info()
    rss = proc.memory_info().rss / 2**20
    read_mb = proc.io_counters().read_bytes / 2**20
    if not force and all(abs(v - _last[k]) < min_delta_mb for k, v in
                         (("alloc", alloc), ("reserved", reserved), ("free", free), ("rss", rss))):
        return  # 变化不够大就不刷屏
    _last.update(alloc=alloc, reserved=reserved, free=free, rss=rss)
    mapped = sum(m.rss for m in proc.memory_maps() if "safetensors" in m.path.lower()) / 2**20
    print(f"[{tag}] torch占用={alloc:.0f}MB torch保留={reserved:.0f}MB 设备剩余={free/2**20:.0f}MB "
          f"RAM可用={psutil.virtual_memory().available/2**20:.0f}MB "
          f"进程RSS={rss:.0f}MB 读盘累计={read_mb:.0f}MB 权重映射驻留={mapped:.0f}MB", flush=True)


def guard():
    """设备级显存护栏：显存或RAM低于地板值立即退出，防蓝屏；顺带打变化驱动的显存日志。"""
    import psutil
    while not stop.is_set():
        free, _total = torch.cuda.mem_get_info()
        ram_avail = psutil.virtual_memory().available / 2**20
        if free / 2**20 < FREE_FLOOR_MB:
            print(f"\n!!! 设备剩余显存 {free/2**20:.0f}MB < {FREE_FLOOR_MB}MB，杀进程防溢出", flush=True)
            os._exit(1)
        if ram_avail < RAM_FLOOR_MB:
            print(f"\n!!! RAM可用 {ram_avail:.0f}MB < {RAM_FLOOR_MB}MB，杀进程防蓝屏", flush=True)
            os._exit(1)
        log("监测")
        stop.wait(LOG_INTERVAL)  # ponytail: log 内部有 64MB 变化阈值，实际不会 2 秒一行


def main():
    threading.Thread(target=guard, daemon=True).start()

    latent_frames = (FRAMES - 1) // 4 + 1
    latent_hw = (HEIGHT // 8, WIDTH // 8)
    print(f"[参数] {WIDTH}x{HEIGHT} 像素帧={FRAMES} latent帧={latent_frames} "
          f"context={CONTEXT_FRAMES}/{CONTEXT_OVERLAP} steps={STEPS}", flush=True)

    patcher = comfy.sd.load_diffusion_model(MODEL_PATH)
    for lora_path, strength in LORAS:
        patcher, _ = comfy.sd.load_lora_for_models(patcher, None, comfy.utils.load_torch_file(lora_path), strength, 0.0)
    model = patcher.model
    print(f"[模型] {type(model).__name__} dynamic={patcher.is_dynamic()} "
          f"fast_disk={getattr(patcher, 'fast_disk', None)} "
          f"权重约={mm.module_size(patcher.model) / 2**30:.1f}GB", flush=True)

    # 照抄 WanContextWindowsManual.execute：context_length/overlap 从像素帧换算到 latent 帧
    patcher.model_options["context_handler"] = comfy.context_windows.IndexListContextHandler(
        context_schedule=comfy.context_windows.get_matching_context_schedule("standard_static"),
        fuse_method=comfy.context_windows.get_matching_fuse_method("pyramid"),
        context_length=max(((CONTEXT_FRAMES - 1) // 4) + 1, 1),
        context_overlap=max(CONTEXT_OVERLAP // 4, 0),
        context_stride=1,
        closed_loop=False,
        dim=2,
        freenoise=False,
        cond_retain_index_list="0",   # retain_first_frame=True
        causal_window_fix=True,
    )
    comfy.context_windows.create_prepare_sampling_wrapper(patcher)

    # 随机构造 conds（语义无关，只对显存结构负责）。键名对应 WAN22_Animate.extra_conds 的读取
    with torch.inference_mode():  # server 在 execution.py 全程 inference_mode，必须对齐
        text_embed = torch.zeros(1, 512, 4096)   # umt5_xxl 输出: seq 512, dim 4096
        clip_fea = SimpleNamespace(penultimate_hidden_states=torch.zeros(1, 257, 1280))  # clip_vision_h
        pose_latents = torch.zeros(1, 16, latent_frames, *latent_hw)
        ref_latent = torch.zeros(1, 16, 1, *latent_hw)
        face_pixel_values = torch.zeros(1, 3, latent_frames * 4 - 3, HEIGHT, WIDTH)

        positive = [[text_embed, {
            "clip_vision_output": clip_fea,
            "pose_latents": pose_latents,
            "face_pixel_values": face_pixel_values,
            "reference_latents": [ref_latent],
        }]]
        negative = [[text_embed, {}]]

        latent_image = torch.zeros(1, 16, latent_frames, *latent_hw)
        noise = comfy.sample.prepare_noise(latent_image, SEED)

        log("采样前")
        samples = comfy.sample.sample(
            patcher, noise, steps=STEPS, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=positive, negative=negative,
            latent_image=latent_image, seed=SEED)
        log("采样后")
        print(f"[完成] 输出 shape={tuple(samples.shape)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        stop.set()
