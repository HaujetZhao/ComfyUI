# 导出 kjnodes 的分类、输入显示名、输出显示名、返回类型清单（供第二轮翻译）
import sys, json, types
sys.path.insert(0, r"e:\ComfyUI")

import server
class Q:
    def put(self, *a, **k): pass
class Dummy:
    prompt_queue = Q()
    app = types.SimpleNamespace(router=types.SimpleNamespace(frozen=True))
server.PromptServer.instance = Dummy()

import nodes
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "comfyui_kjnodes", r"e:\ComfyUI\custom_nodes\comfyui-kjnodes\__init__.py")
kj = importlib.util.module_from_spec(_spec)
sys.modules["comfyui_kjnodes"] = kj
_spec.loader.exec_module(kj)

mappings = kj.NODE_CLASS_MAPPINGS

# 1) 分类
cats = set()
for cls in mappings.values():
    c = getattr(cls, "CATEGORY", None)
    if c:
        cats.add(c)
print("分类数:", len(cats))
for c in sorted(cats):
    print(" ", c)

# 2) 输入显示名 + 输出显示名/类型
slots = {}
all_types = set()
for key, cls in mappings.items():
    entry = {}
    try:
        it = cls.INPUT_TYPES()
        inputs = {}
        for section in ("required", "optional"):
            for name, spec in (it.get(section) or {}).items():
                raw_t = spec[0] if isinstance(spec, tuple) else spec
                t = raw_t if isinstance(raw_t, str) else str(raw_t)
                if isinstance(raw_t, str):
                    all_types.add(raw_t)
                # 显示名：后端没给 display_name 时前端回退参数名本身
                disp = spec[1].get("display_name") if isinstance(spec, tuple) and len(spec) > 1 and isinstance(spec[1], dict) else None
                inputs[name] = {"type": t, "display_name": disp}
        entry["inputs"] = inputs
    except Exception as e:
        entry["inputs_error"] = str(e)
    try:
        entry["outputs"] = [str(t) for t in cls.RETURN_TYPES]
        all_types |= {str(t) for t in cls.RETURN_TYPES}
        rn = getattr(cls, "RETURN_NAMES", None)
        if rn:
            entry["output_names"] = [str(n) for n in rn]
    except Exception as e:
        entry["outputs_error"] = str(e)
    slots[key] = entry

json.dump(slots, open(r"e:\ComfyUI\my\kj_slots_manifest.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
json.dump(sorted(cats), open(r"e:\ComfyUI\my\kj_categories.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)

# 3) 自定义类型（官方 zh main.json 已覆盖的不算）
official = json.load(open(r"E:\repos\ComfyUI_frontend\src\locales\zh\main.json", encoding="utf8")).get("dataTypes", {})
custom_types = sorted(t for t in all_types if t not in official and not t.startswith(("COMFY", "*")))
json.dump(custom_types, open(r"e:\ComfyUI\my\kj_custom_types.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("输入参数总数:", sum(len(v.get("inputs", {})) for v in slots.values()))
print("自定义类型:", len(custom_types), custom_types[:15])
