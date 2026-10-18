"""修复 VFR 源视频导致的 Wan Animate 音画漂移。

原理：VHS_LoadVideo 的 cv 路径按帧索引（假设恒定 fps）抽帧，无视真实 pts。
抖音源是 VFR（平均 58.5fps，末尾 2s 掉到 30fps），导致姿势序列内容时间
与显示时间漂移：中间累计落后 ~0.6s，末尾 1s 内追回（看起来加速收尾）。

补救：按 VHS 算法精确复算每帧对应的真实内容时间 t(i)，用 PyAV 解码输出
视频后按 t(i) 重设每帧 pts 再编码，音频流直接复制。
"""
import subprocess
from fractions import Fraction

import av

INPUT = "input/抖音_16628_夏天太热啦_小仙摇_crop512_x300_y700.mp4"
TARGETS = [
    ("output/动作迁移_00020-audio.mp4", "output/动作迁移_00020-audio_修正.mp4"),
    ("output/动作迁移对比_00022-audio.mp4", "output/动作迁移对比_00022-audio_修正.mp4"),
]
FORCE_RATE = 30
POSE_FRAMES = 836  # pose/face 序列实际帧数，其余为末帧重复


def probe_pts(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout
    return [float(l.strip().rstrip(",")) for l in out.splitlines() if l.strip()]


def simulate_vhs(pts, fps):
    """逐行复现 VHS cv_frame_generator 的选帧，返回被选中帧的真实 pts。"""
    base, target = 1 / fps, 1 / FORCE_RATE
    held = 0          # 开头 grab() 已消费第 0 帧
    offset = target
    sel = []
    while len(sel) < POSE_FRAMES:
        if offset < target:
            held += 1
            if held >= len(pts):
                break
            offset += base
            continue
        offset -= target
        sel.append(pts[held])
    return sel


def resync(src, dst, content_ts):
    src_c = av.open(src)
    v_in = src_c.streams.video[0]
    a_in = src_c.streams.audio[0]

    out_c = av.open(dst, "w")
    v_out = out_c.add_stream("libx264", rate=FORCE_RATE)
    v_out.width, v_out.height, v_out.pix_fmt = v_in.width, v_in.height, "yuv420p"
    v_out.options = {"crf": "19"}
    # 显式细时基，避免 pts 被量化到 1/30 网格
    time_base = Fraction(1, 1000000)
    v_out.codec_context.time_base = time_base
    a_out = out_c.add_stream("aac", rate=a_in.rate, layout=a_in.layout)

    # 每帧真实内容时间；超出姿势序列的末帧按 1/30 延续
    ts = list(content_ts)
    count = 0
    for frame in src_c.decode(video=0):  # decode 按显示顺序吐帧
        if count >= len(ts):
            ts.append(ts[-1] + 1 / FORCE_RATE)
        frame.pts = int(ts[count] / time_base)
        frame.time_base = time_base
        for pkt in v_out.encode(frame):
            out_c.mux(pkt)
        count += 1
    for frame in av.open(src).decode(audio=0):  # 视频循环已耗尽容器，音频单独开
        for pkt in a_out.encode(frame):
            out_c.mux(pkt)
    for pkt in v_out.encode(None):
        out_c.mux(pkt)
    for pkt in a_out.encode(None):
        out_c.mux(pkt)
    out_c.close()
    src_c.close()
    print(f"  {src} -> {dst} ({len(ts)} 帧)")


def main():
    pts = probe_pts(INPUT)
    fps = len(pts) / pts[-1]  # cv2 CAP_PROP_FPS 报的就是这个均值
    print(f"源: {len(pts)} 帧, 平均 {fps:.2f} fps (VFR)")
    sel = simulate_vhs(pts, fps)
    print(f"复算抽帧数 {len(sel)} (应={POSE_FRAMES}), 内容时间 0 → {sel[-1]:.3f}s")
    for src, dst in TARGETS:
        resync(src, dst, sel)


if __name__ == "__main__":
    main()
