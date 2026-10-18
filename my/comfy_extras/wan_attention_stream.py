import torch
import torch.nn as nn
import math
from einops import rearrange
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope1
import comfy.model_management as mm

# 显存追踪开关
DEBUG_MEM = True

def print_vram(label):
    if DEBUG_MEM and mm.get_torch_device().type == 'cuda':
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"DEBUG_VRAM [{label}]: Allocated: {mem:.2f}MB, Peak: {peak:.2f}MB")
        torch.cuda.reset_peak_memory_stats()

def tensor_info(name, t):
    if DEBUG_MEM:
        size_mb = t.element_size() * t.nelement() / (1024**2)
        print(f"DEBUG_TENSOR [{name}]: Shape: {list(t.shape)}, Dtype: {t.dtype}, Device: {t.device}, Size: {size_mb:.2f}MB")

def repeat_e(e, x):
    repeats = 1
    if e.size(1) > 1:
        repeats = x.size(1) // e.size(1)
    if repeats == 1:
        return e
    if repeats * e.size(1) == x.size(1):
        return torch.repeat_interleave(e, repeats, dim=1)
    else:
        return torch.repeat_interleave(e, repeats + 1, dim=1)[:, :x.size(1)]

class WanSelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 kv_dim=None,
                 operation_settings={}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        if kv_dim is None:
            kv_dim = dim

        ops = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")

        self.q = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.k = ops.Linear(kv_dim, dim, device=device, dtype=dtype)
        self.v = ops.Linear(kv_dim, dim, device=device, dtype=dtype)
        self.o = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype) if qk_norm else nn.Identity()
        self.norm_k = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype) if qk_norm else nn.Identity()

    def forward(self, x, freqs, transformer_options={}, modulation_params=None, norm_layer=None, residual_x=None, residual_gate=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        chunk_size = 4096

        # --- [手术步第八步：预分配 K/V 底座，消灭 cat 峰值] ---
        compute_device = self.q.weight.device
        if compute_device.type == 'cpu':
            if modulation_params is not None:
                compute_device = modulation_params[0].device
            else:
                compute_device = mm.get_torch_device()
                
        full_k = torch.empty((b, s, n, d), device=compute_device, dtype=x.dtype)
        full_v = torch.empty((b, s, n * d), device=compute_device, dtype=x.dtype)
        
        # 1. 第一轮循环：分块计算并填入 K 和 V
        for i in range(0, s, chunk_size):
            end = min(i + chunk_size, s)
            # 支持 CPU Latent
            x_chunk = x[:, i:end].to(compute_device)
            freqs_chunk = freqs[:, i:end] if freqs is not None else None
            
            if modulation_params is not None and norm_layer is not None:
                mod_chunk = torch.addcmul(
                    repeat_e(modulation_params[0], x_chunk), 
                    norm_layer(x_chunk), 
                    1 + repeat_e(modulation_params[1], x_chunk)
                )
            else:
                mod_chunk = x_chunk

            # 计算 K 并填入底座
            k_chunk = self.norm_k(self.k(mod_chunk)).view(b, x_chunk.shape[1], n, d)
            if freqs_chunk is not None:
                k_chunk = apply_rope1(k_chunk, freqs_chunk)
            full_k[:, i:end] = k_chunk
            
            # 计算 V 并填入底座
            full_v[:, i:end] = self.v(mod_chunk).view(b, x_chunk.shape[1], n * d)
            
        print_vram("Self-Attn full_KV Ready (No Cat Peak)")

        # 2. 第二轮循环：按 Query 分块执行 Attention
        for i in range(0, s, chunk_size):
            end = min(i + chunk_size, s)
            # --- [极致优化：支持 CPU Latent 拉取] ---
            x_chunk = x[:, i:end].to(full_k.device) 
            freqs_chunk = freqs[:, i:end] if freqs is not None else None
            
            if modulation_params is not None and norm_layer is not None:
                mod_chunk = torch.addcmul(
                    repeat_e(modulation_params[0], x_chunk), 
                    norm_layer(x_chunk), 
                    1 + repeat_e(modulation_params[1], x_chunk)
                )
            else:
                mod_chunk = x_chunk

            # 计算 Q_chunk
            q_chunk = self.norm_q(self.q(mod_chunk)).view(b, x_chunk.shape[1], n, d)
            if freqs_chunk is not None:
                q_chunk = apply_rope1(q_chunk, freqs_chunk)
            
            # 执行注意力
            attn_chunk = optimized_attention(
                q_chunk.view(b, q_chunk.shape[1], n * d),
                full_k.view(b, s, n * d),
                full_v,
                heads=self.num_heads,
                transformer_options=transformer_options,
            )
            
            # 直接投影加回 residual_x
            o_chunk = self.o(attn_chunk)
            if residual_x is not None and residual_gate is not None:
                gate = repeat_e(residual_gate, x_chunk)
                # 使用 += 确保如果 residual_x 在 CPU，则自动写回 CPU
                delta = (o_chunk * gate).to(residual_x.device)
                residual_x[:, i:end] += delta
            elif residual_x is not None:
                residual_x[:, i:end] += o_chunk.to(residual_x.device)
                
        print_vram("Self-Attn Chunked-Query Finished")
        return None







class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, transformer_options={}, norm_layer=None, residual_x=None, **kwargs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        chunk_size = 4096
        
        # 1. 预先计算全量的 K 和 V (Context 长度通常较短)
        full_k = self.norm_k(self.k(context))
        full_v = self.v(context)
        
        # 2. 循环处理 Query 分块
        compute_device = full_k.device
        if compute_device.type == 'cpu':
            compute_device = mm.get_torch_device()
        for i in range(0, s, chunk_size):
            # 支持 CPU Latent
            x_chunk = x[:, i:i+chunk_size].to(compute_device)
            q_in_chunk = norm_layer(x_chunk) if norm_layer is not None else x_chunk
            q_chunk = self.norm_q(self.q(q_in_chunk))
            
            # 分块执行 Attention
            attn_chunk = optimized_attention(
                q_chunk, full_k, full_v, 
                heads=self.num_heads, 
                transformer_options=transformer_options
            )
            
            # 立即投影并更新 residual_x
            o_chunk = self.o(attn_chunk)
            if residual_x is not None:
                residual_x[:, i:i+chunk_size] += o_chunk.to(residual_x.device)
            else:
                # 兼容模式：如果没传 residual_x，我们这里无法原地返回，只能报错或回滚。
                # 但目前 Block 级逻辑已保证会有 residual_x。
                pass
                
        return None

class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6, operation_settings={}):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)
        ops = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        self.k_img = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.v_img = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_k_img = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype) if qk_norm else nn.Identity()

    def forward(self, x, context, context_img_len, transformer_options={}, norm_layer=None, residual_x=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        context_img = context[:, :context_img_len]
        context = context[:, context_img_len:]
        chunk_size = 4096
        
        # 1. 预先计算全量的 K 和 V
        k = self.norm_k(self.k(context))
        v = self.v(context)
        k_img = self.norm_k_img(self.k_img(context_img))
        v_img = self.v_img(context_img)
        
        # 2. 循环处理 Query 分块
        compute_device = k.device
        if compute_device.type == 'cpu':
            compute_device = mm.get_torch_device()
        for i in range(0, s, chunk_size):
            # 支持 CPU Latent
            x_chunk = x[:, i:i+chunk_size].to(compute_device)
            q_in_chunk = norm_layer(x_chunk) if norm_layer is not None else x_chunk
            q_chunk = self.norm_q(self.q(q_in_chunk))
            
            # 分块执行两次 Attention (Context + Image Context)
            img_x_chunk = optimized_attention(q_chunk, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options)
            attn_chunk = optimized_attention(q_chunk, k, v, heads=self.num_heads, transformer_options=transformer_options)
            
            # 立即投影并更新 residual_x
            o_chunk = self.o(attn_chunk + img_x_chunk)
            if residual_x is not None:
                residual_x[:, i:i+chunk_size] += o_chunk.to(residual_x.device)
        
        return None





WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

class WanAttentionBlock(nn.Module):
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        ops = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")

        self.norm1 = ops.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)
        self.norm3 = ops.LayerNorm(dim, eps, elementwise_affine=True, device=device, dtype=dtype) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps, operation_settings=operation_settings)
        self.norm2 = ops.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
        self.ffn = nn.Sequential(
            ops.Linear(dim, ffn_dim, device=device, dtype=dtype), nn.GELU(approximate='tanh'),
            ops.Linear(ffn_dim, dim, device=device, dtype=dtype))

        self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=device, dtype=dtype))

    def forward(self, x, e, freqs, context, context_img_len=257, transformer_options={}):
        from comfy_extras.wan_attention_stream import print_vram, tensor_info
        # 审计权重与输入分布
        tensor_info("Block_Q_Weight", self.self_attn.q.weight)
        tensor_info("Block_Input_E", e)
        if freqs is not None: tensor_info("Block_Input_Freqs", freqs)
        if context is not None: tensor_info("Block_Input_Context", context)
        
        if e.ndim < 4:
            modulation = mm.cast_to(self.modulation, dtype=x.dtype, device=e.device)
            e_chunks = (modulation + e).chunk(6, dim=1)
        else:
            modulation = mm.cast_to(self.modulation, dtype=x.dtype, device=e.device).unsqueeze(0)
            e_chunks = (modulation + e).unbind(2)

        # self-attention
        x = x.contiguous()
        print_vram("Block Start")
        
        # --- [手术步第六步：Self-Attn 完全 In-place 重构] ---
        self.self_attn(
            x, freqs, 
            transformer_options=transformer_options,
            modulation_params=(e_chunks[0], e_chunks[1]),
            norm_layer=self.norm1,
            residual_x=x,
            residual_gate=e_chunks[2]
        )
        print_vram("Block Self-Attn (In-place) Done")
        print_vram("Block Residual 1 Done")


        # cross-attention
        chunk_size = 4096
        self.cross_attn(
            x, context, 
            context_img_len=context_img_len, 
            transformer_options=transformer_options,
            norm_layer=self.norm3,
            residual_x=x
        )
        print_vram("Block Cross-Attn Done")


        # --- [手术步第六步：FFN In-place 重构] ---
        # 确定 FFN 计算设备
        ffn_compute_device = self.ffn[0].weight.device
        if ffn_compute_device.type == 'cpu':
            ffn_compute_device = e_chunks[3].device
            
        # 彻底消灭 y_parts 和 torch.cat(y_parts)
        for i in range(0, x.size(1), chunk_size):
            # 支持 CPU Latent
            x_chunk = x[:, i:i+chunk_size].to(ffn_compute_device)
            mod_a = repeat_e(e_chunks[3], x_chunk)
            mod_b = repeat_e(e_chunks[4], x_chunk)
            mod_g = repeat_e(e_chunks[5], x_chunk)
            
            # 局部计算 FFN 输入
            ffn_in_chunk = torch.addcmul(mod_a, self.norm2(x_chunk), 1 + mod_b)
            # 局部计算 FFN 输出并直接加回 x
            res_ffn = self.ffn(ffn_in_chunk) * mod_g
            x[:, i:i+chunk_size] += res_ffn.to(x.device)
            
        print_vram("Block All Done (In-place)")
        return x


# --- [手术九步：FaceBlock 显存优化] ---

class FaceBlockStream(nn.Module):
    # 我们不直接继承 FaceBlock 以免引入 model_animate.py 的循环依赖
    # 而是手动实现其逻辑
    def __init__(self, hidden_size, heads_num, qk_norm, qk_norm_type, operations, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.head_dim = hidden_size // heads_num
        
        factory_kwargs = {"device": kwargs.get("device"), "dtype": kwargs.get("dtype")}
        
        self.linear1_kv = operations.Linear(hidden_size, hidden_size * 2, **factory_kwargs)
        self.linear1_q = operations.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear2 = operations.Linear(hidden_size, hidden_size, **factory_kwargs)
        
        if qk_norm_type == "rms":
            norm_cls = getattr(operations, "RMSNorm", None)
            if norm_cls is None:
                # Fallback implementation
                class RMSNormFallback(nn.Module):
                    def __init__(self, dim, eps=1e-6, elementwise_affine=True, device=None, dtype=None):
                        super().__init__()
                        self.eps = eps
                        if elementwise_affine:
                            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
                        else:
                            self.weight = None
                    def forward(self, x):
                        r_std = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                        return x * r_std * self.weight if self.weight is not None else x * r_std
                norm_cls = RMSNormFallback
        else:
            norm_cls = operations.LayerNorm
            
        self.q_norm = norm_cls(self.head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        self.k_norm = norm_cls(self.head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        
        self.pre_norm_feat = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.pre_norm_motion = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

    def forward(self, x, motion_vec, motion_mask=None, transformer_options={}):
        from comfy_extras.wan_attention_stream import tensor_info
        tensor_info("FaceBlock_X", x)
        tensor_info("FaceBlock_Motion_Vec", motion_vec)
        if motion_mask is not None: tensor_info("FaceBlock_Motion_Mask", motion_mask)
        
        B, T_comp, N, C = motion_vec.shape
        L_total = x.shape[1]
        S = L_total // T_comp # 每帧的 patch 数
        
        # 1. 预先计算全量的 K 和 V (针对 motion_vec)
        # motion_vec 通常很短 (例如 38x257)，显存占用极小 (~30MB)
        x_motion = self.pre_norm_motion(motion_vec)
        kv = self.linear1_kv(x_motion)
        k, v = rearrange(kv, "B L N (K H D) -> K B L N H D", K=2, H=self.heads_num)
        k = self.k_norm(k).to(v)
        # 准备好 (B*T) N H D 形状供后续分块使用
        k_full = rearrange(k, "B L N H D -> (B L) N H D")
        v_full = rearrange(v, "B L N H D -> (B L) N H D")
        
        # 2. 分块处理主序列 x (Query)
        # 注意：FaceBlock 是按帧对齐的 Cross-Attention。
        # 我们按帧的分块大小 S 来进行处理，确保每一块 Q 对应正确的 K[L]
        compute_device = k_full.device
        if compute_device.type == 'cpu':
            compute_device = mm.get_torch_device()
        for t in range(T_comp):
            start = t * S
            end = (t + 1) * S
            # 支持 CPU Latent
            x_chunk = x[:, start:end].to(compute_device)
            
            # Q 投影与 Norm
            x_feat_chunk = self.pre_norm_feat(x_chunk)
            q_chunk = self.linear1_q(x_feat_chunk)
            q_chunk = rearrange(q_chunk, "B S (H D) -> B S H D", H=self.heads_num)
            q_chunk = self.q_norm(q_chunk).to(v)
            
            # 这里对应原逻辑：(B L) S (H D)
            # 对于当前块，B=1, t=t
            q_ready = rearrange(q_chunk, "B S H D -> B S (H D)")
            k_ready = k_full[B*t : B*(t+1)]
            v_ready = v_full[B*t : B*(t+1)]
            
            # 分块 Attention
            attn_chunk = optimized_attention(
                q_ready, k_ready, v_ready, 
                heads=self.heads_num, 
                transformer_options=transformer_options
            )
            
            # 输出投影并原地累加到 x
            out_chunk = self.linear2(attn_chunk)
            
            if motion_mask is not None:
                mask_chunk = rearrange(motion_mask, "B T H W -> B (T H W)")[:, start:end].unsqueeze(-1)
                delta = (out_chunk * mask_chunk).to(x.device)
                x[:, start:end] += delta
            else:
                x[:, start:end] += out_chunk.to(x.device)
        
        print_vram("FaceBlock Chunked-Query Finished")
        return None # 返回 None 表示已原地更新，外层需要适配


# 为了让 model_animate.py 里的 FaceAdapter 能够使用它，我们需要一个工厂函数
def create_face_block_stream(hidden_size, heads_num, **kwargs):
    return FaceBlockStream(hidden_size, heads_num, **kwargs)
