# TODO

## VAE 显存优化调查（2026-10-17）

背景：本地补丁（streaming-vae-attention-0310 分支）自带流式 VAE 编码 `wan_vae_stream.py`，想确认官方后续是否已解决同等问题、能否升级基线甩掉 VAE 补丁。

### 官方 VAE 显存优化时间线（035414e 之后 → master 2026-10）

| 提交 | 日期 | 内容 | 解决驻留？ |
|---|---|---|---|
| `035414e` #13014 | 03-17 | Wan VAE 编码 4 帧/批 → 2 帧/批；下采样"单帧 cork"进缓存（返回 None 延迟到下块）；全面去 `.clone()`；解码改递归逐帧上采样边走边释放 | 否，只降激活峰值 |
| `82b868a` #13073 | 03-19 | 修 tiler fallback 的 VRAM 泄漏（`soft_empty_cache` + `run_up` 移出闭包避免循环引用） | 否 |
| `25b6d1d` #13101 | 03-21 | 修 035414e 引入的 resample 过早拆分丢卷积 bug（第 5→6 帧跳色） | 否 |
| `fabed69` #13062 | 03-19 | **`comfy_has_chunked_io` 协议**（rattus 本人，先给 LTX 实现）：encode 像素留 CPU、模型内部分块搬；decode 预分配 CPU output_buffer 直写，消灭大 cat 双倍峰值 | **是，但 Wan 没接入** |

协议当前实现方：Lightricks / MiniMax / SeedVR 三家 VAE。**WanVAE 至今还是老接口**：
- encode：`sd.py` 仍整个视频 `.to(device)`（720p 900 帧 ≈ 5GB 输入直接上 GPU）
- decode：仍全量解码后再搬（`memory_used_decode` 对 latent 有含 t 的估算，但 batch 维度为 1 时无济于事）

另：decode 路径新增 `comfy_decode_estimate_is_reliable`（估算超显存直接跳到 tiled），算兜底。

### 结论

1. 官方解法形态 = 我们 zip 补丁的流式思路（输入留 CPU、分块喂、输出直写），方向被官方认可，只是做成了协议且只给了三家
2. **升级到新版 ≠ 解决 Wan 长视频 VAE 驻留**，`wan_vae_stream.py` 在 Wan 上仍是独一份，不升级基线
3. 低成本自救路线（留在当前基线）：需要时把 sd.py 的 chunked_io 协议移植过来 + 给 WanVAE 实现 `comfy_has_chunked_io`（实现体就是现有流式编码器换皮）
4. 眼下 decode 撞墙用 tiled VAE decode 节点绕过，encode 侧已被补丁覆盖

### 待办

- [ ] 催 rattus128 给 WanVAE 接入 `comfy_has_chunked_io` → 已提 [#17114](https://github.com/Comfy-Org/ComfyUI/issues/17114)（2026-10-17），等回复
- [ ] 720p 长视频实测：关注 RAM 余量（RAMAvail 只剩 0.6~0.8GB）与 VAE decode 峰值
- [ ] 备用方案未实测：`--disable-dynamic-vram`（RAM 缓慢流失疑似 aimdo vbar mmap 页）
- [ ] 未提交：`print_vram` 监控增强、`intermediate_dtype` shim（等用户发话）
- [ ] 旧分支清理：streaming-vae-attention、streaming-vae-attention-0317（空）、vaered（有 ef9ed07）
