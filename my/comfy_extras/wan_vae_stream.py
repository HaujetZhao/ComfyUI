import torch
import logging
import math
import comfy.model_management as mm
from comfy.ldm.wan.vae import count_conv3d

class WanVAEStreamEncoder:
    def __init__(self, vae_wrapper, tile_size=512, overlap=64, offload_to_cpu=True):
        self.vae_wrapper = vae_wrapper
        self.device = self.vae_wrapper.device
        self.dtype = self.vae_wrapper.vae_dtype
        self.first_stage = self.vae_wrapper.first_stage_model
        
        self.tile_size = tile_size
        self.overlap = overlap
        self.offload_to_cpu = offload_to_cpu
        self.tile_states = {} # (y, x) -> feat_map
        self.num_layers = count_conv3d(self.first_stage.encoder)

        self.latents = []
        self.unprocessed_pixels = None
        self.is_first = True

    def _encode_tensor(self, x, feat_cache):
        conv_idx = [0]
        with torch.no_grad():
            out_ = self.first_stage.encoder(
                x,
                feat_cache=feat_cache,
                feat_idx=conv_idx
            )
            mu, _ = self.first_stage.conv1(out_).chunk(2, dim=1)
            return mu.float()

    def add_chunk(self, pixels_input):
        T, H, W, C = pixels_input.shape

        if self.is_first:
            lat_0 = self._spatial_tiled_encode(pixels_input[:1])
            self.latents.append(lat_0)
            pixels_input = pixels_input[1:]
            self.is_first = False
            T -= 1

        if self.unprocessed_pixels is not None:
            pixels_input = torch.cat([self.unprocessed_pixels, pixels_input], dim=0)
            T = pixels_input.shape[0]

        remainder = T % 4
        if remainder > 0:
            self.unprocessed_pixels = pixels_input[-remainder:]
            pixels_input = pixels_input[:-remainder]
        else:
            self.unprocessed_pixels = None
            
        if pixels_input.shape[0] > 0:
            for i in range(pixels_input.shape[0] // 4):
                chunk_pixels = pixels_input[i*4:(i+1)*4]
                lat = self._spatial_tiled_encode(chunk_pixels)
                self.latents.append(lat)

    def _spatial_tiled_encode(self, chunk_pixels):
        T, H, W, C = chunk_pixels.shape
        out_T = 1
        out_H, out_W = H // 8, W // 8
        
        # If no tiling, process everything at once
        if self.tile_size <= 0:
            tile_pix_gpu = self.vae_wrapper.process_input(chunk_pixels).to(self.dtype).to(self.device).movedim(-1, 1).movedim(1, 0).unsqueeze(0)
            state_key = (0, 0)
            if state_key not in self.tile_states:
                self.tile_states[state_key] = [None] * self.num_layers
            
            if self.offload_to_cpu:
                current_state = [f.to(self.device).to(self.dtype) if isinstance(f, torch.Tensor) else f for f in self.tile_states[state_key]]
            else:
                current_state = self.tile_states[state_key]

            lat_tile = self._encode_tensor(tile_pix_gpu, current_state)
            
            if self.offload_to_cpu:
                self.tile_states[state_key] = [f.cpu() if isinstance(f, torch.Tensor) else f for f in current_state]
            else:
                self.tile_states[state_key] = current_state
            
            return lat_tile.cpu()

        # Tiled logic
        output_latent = torch.zeros((1, self.vae_wrapper.latent_channels, out_T, out_H, out_W), device="cpu")
        output_weights = torch.zeros((1, 1, out_T, out_H, out_W), device="cpu")

        ts = self.tile_size
        ol = self.overlap
        
        for y in range(0, H, ts - ol):
            for x in range(0, W, ts - ol):
                y_end = min(y + ts, H)
                x_end = min(x + ts, W)
                y_start = max(y, 0)
                x_start = max(x, 0)
                
                tile_pix = chunk_pixels[:, y_start:y_end, x_start:x_end, :]
                tile_pix_gpu = self.vae_wrapper.process_input(tile_pix).to(self.dtype).to(self.device).movedim(-1, 1).movedim(1, 0).unsqueeze(0)

                state_key = (y, x)
                if state_key not in self.tile_states:
                    self.tile_states[state_key] = [None] * self.num_layers
                
                if self.offload_to_cpu:
                    current_state = [f.to(self.device).to(self.dtype) if isinstance(f, torch.Tensor) else f for f in self.tile_states[state_key]]
                else:
                    current_state = self.tile_states[state_key]

                lat_tile = self._encode_tensor(tile_pix_gpu, current_state)
                
                if self.offload_to_cpu:
                    self.tile_states[state_key] = [f.cpu() if isinstance(f, torch.Tensor) else f for f in current_state]
                else:
                    self.tile_states[state_key] = current_state
                
                lat_tile = lat_tile.cpu()
                
                h_tile, w_tile = lat_tile.shape[3], lat_tile.shape[4]
                mask = torch.ones((1, 1, 1, h_tile, w_tile))
                feather = ol // 8
                if feather > 0:
                    for i in range(min(feather, h_tile // 2)):
                        v = (i + 0.5) / feather
                        if y > 0: mask[:, :, :, i, :] *= v
                        if y_end < H: mask[:, :, :, -1-i, :] *= v
                    for i in range(min(feather, w_tile // 2)):
                        v = (i + 0.5) / feather
                        if x > 0: mask[:, :, :, :, i] *= v
                        if x_end < W: mask[:, :, :, :, -1-i] *= v
                
                ly, lx = y_start // 8, x_start // 8
                output_latent[:, :, :, ly:ly+h_tile, lx:lx+w_tile] += lat_tile * mask
                output_weights[:, :, :, ly:ly+h_tile, lx:lx+w_tile] += mask
                del lat_tile, tile_pix_gpu
                if self.offload_to_cpu:
                    torch.cuda.empty_cache()
                
        return output_latent / (output_weights + 1e-8)

    def flush(self):
        self.unprocessed_pixels = None

    def get_result(self):
        self.flush()
        if len(self.latents) == 0:
            return None
        return torch.cat(self.latents, dim=2)

    def encode(self, image_cpu, chunk_size=16):
        if not hasattr(self.vae_wrapper.first_stage_model, 'encoder'):
            return self.vae_wrapper.encode(image_cpu)
        
        if self.tile_size > 0:
            fake_shape = (1, 3, 4, self.tile_size, self.tile_size)
        else:
            fake_shape = (1, 3, min(chunk_size, image_cpu.shape[0]), image_cpu.shape[1], image_cpu.shape[2])
            
        memory_used = self.vae_wrapper.memory_used_encode(fake_shape, self.vae_wrapper.vae_dtype)
        mm.load_models_gpu([self.vae_wrapper.patcher], memory_required=memory_used)
        
        logging.warning(f"[VRAM] WanVAE stream_encode Start - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Pre-process image to correct dimensions if needed
        pixels = self.vae_wrapper.vae_encode_crop_pixels(image_cpu)
        
        for i in range(0, pixels.shape[0], chunk_size):
            chunk = pixels[i:min(i+chunk_size, pixels.shape[0])]
            self.add_chunk(chunk)
            logging.warning(f"[VRAM] stream_encode Chunk {i//chunk_size} ({i}-{i+len(chunk)}) - Peak: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            
        res = self.get_result()
        logging.warning(f"[VRAM] WanVAE stream_encode End - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        return res
