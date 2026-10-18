# 验证 encode_image 分块与整批编码数学等价（用真实 clip_vision_h 模型）
import sys
sys.path.insert(0, ".")
import torch
import comfy.clip_vision

KEYS = ["last_hidden_state", "image_embeds", "image_sizes", "penultimate_hidden_states", "mm_projected"]

clip = comfy.clip_vision.load("models/clip_vision/clip_vision_h.safetensors")
torch.manual_seed(0)
imgs = torch.rand(3, 512, 512, 3)

with torch.no_grad():
    whole = clip.encode_image(imgs, crop=False)
    chunks = [clip.encode_image_chunk(imgs[i:i+1], crop=False) for i in range(3)]

for k in KEYS:
    if whole[k] is None:
        print(f"{k}: None，跳过")
        continue
    if k == "image_sizes":
        assert sum((c[k] for c in chunks), []) == whole[k], k
        print(f"{k}: OK")
        continue
    cat = torch.cat([c[k] for c in chunks], dim=0)
    diff = (whole[k].float() - cat.float()).abs().max().item()
    print(f"{k}: shape={tuple(whole[k].shape)} max_diff={diff:.2e}")
    assert diff < 1e-2, k
print("等价性验证通过")
