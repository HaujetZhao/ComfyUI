# 导出 rgthree 全部节点的权威清单（键/显示名/描述/输入/输出），供翻译用
import sys, json, types
sys.path.insert(0, r"e:\ComfyUI")

import server
class Q:
    def put(self, *a, **k): pass
from aiohttp.web import RouteTableDef
class Dummy:
    prompt_queue = Q()
    app = types.SimpleNamespace(router=types.SimpleNamespace(frozen=True))
    routes = RouteTableDef()
server.PromptServer.instance = Dummy()

import nodes
import comfy_extras.nodes_post_processing  # 预热，避免 rgthree 触发循环导入
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "rgthree_comfy", r"e:\ComfyUI\custom_nodes\rgthree-comfy\__init__.py")
rg = importlib.util.module_from_spec(_spec)
sys.modules["rgthree_comfy"] = rg
_spec.loader.exec_module(rg)

mappings = rg.NODE_CLASS_MAPPINGS
manifest = {}
for key, cls in mappings.items():
    entry = {"display_name": getattr(cls, "NAME", cls.__name__)}
    try:
        it = cls.INPUT_TYPES()
        inputs = {}
        for section in ("required", "optional"):
            for name, spec in (it.get(section) or {}).items():
                if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                    opts = spec[1]
                else:
                    opts = {}
                inputs[name] = {
                    "tooltip": opts.get("tooltip"),
                    "display_name": opts.get("display_name"),
                }
        entry["inputs"] = inputs
    except Exception as e:
        entry["inputs_error"] = str(e)
    try:
        entry["outputs"] = [str(t) for t in cls.RETURN_TYPES]
        rn = getattr(cls, "RETURN_NAMES", None)
        if rn:
            entry["output_names"] = [str(n) for n in rn]
    except Exception as e:
        entry["outputs_error"] = str(e)
    entry["description"] = (cls.DESCRIPTION or "").strip() if getattr(cls, "DESCRIPTION", None) else ""
    manifest[key] = entry

json.dump(manifest, open(r"e:\ComfyUI\my\rgthree_manifest.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("节点总数:", len(manifest))
print("有 description 的:", sum(1 for v in manifest.values() if v.get("description")))
print("有 tooltip 的输入:", sum(1 for v in manifest.values() for i in v.get("inputs", {}).values() if i.get("tooltip")))
print("inputs_error:", [k for k, v in manifest.items() if "inputs_error" in v])
print("节点键:", sorted(manifest))
