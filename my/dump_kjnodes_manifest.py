# 导出 kjnodes 全部节点的权威清单（键/显示名/描述/输入/输出），供翻译分片用
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
manifest = {}
for key, cls in mappings.items():
    entry = {"display_name": kj.NODE_DISPLAY_NAME_MAPPINGS.get(key, cls.__name__)}
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
                    "type": spec[0] if isinstance(spec, tuple) else str(spec),
                }
        entry["inputs"] = inputs
    except Exception as e:
        entry["inputs_error"] = str(e)
    try:
        entry["outputs"] = [str(t) for t in cls.RETURN_TYPES]
        if getattr(cls, "OUTPUT_TOOLTIPS", None):
            entry["outputs_tooltips"] = list(cls.OUTPUT_TOOLTIPS)
    except Exception as e:
        entry["outputs_error"] = str(e)
    entry["description"] = (cls.DESCRIPTION or "").strip() if getattr(cls, "DESCRIPTION", None) else ""
    manifest[key] = entry

json.dump(manifest, open(r"e:\ComfyUI\my\kjnodes_manifest.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("节点总数:", len(manifest))
print("有 description 的:", sum(1 for v in manifest.values() if v.get("description")))
print("有 tooltip 的输入:", sum(1 for v in manifest.values() for i in v.get("inputs", {}).values() if i.get("tooltip")))
