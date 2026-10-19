# 校验 rgthree 的 locales/zh/nodeDefs.json 与真实注册表
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

import comfy_extras.nodes_post_processing
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "rgthree_comfy", r"e:\ComfyUI\custom_nodes\rgthree-comfy\__init__.py")
rg = importlib.util.module_from_spec(_spec)
sys.modules["rgthree_comfy"] = rg
_spec.loader.exec_module(rg)

mappings = rg.NODE_CLASS_MAPPINGS
trans = json.load(open(r"e:\ComfyUI\custom_nodes\rgthree-comfy\locales\zh\nodeDefs.json", encoding="utf8"))

# 纯前端节点：服务端注册表没有，由前端合成 nodeDef，键名合法
FRONTEND_ONLY = {
    "Dynamic Context (rgthree)", "Dynamic Context Switch (rgthree)",
    "Mute / Bypass Relay (rgthree)", "Mute / Bypass Repeater (rgthree)",
    "Fast Muter (rgthree)", "Fast Bypasser (rgthree)",
    "Fast Groups Muter (rgthree)", "Fast Groups Bypasser (rgthree)",
    "Fast Actions Button (rgthree)", "Label (rgthree)",
    "Power Conductor (rgthree)", "Node Collector (rgthree)",
    "Reroute (rgthree)", "Random Unmuter (rgthree)", "Bookmark (rgthree)",
}
valid_keys = set(mappings) | FRONTEND_ONLY

errors = []
for k in sorted(set(trans) - valid_keys):
    errors.append(f"多余节点键: {k}")
for k in sorted(valid_keys - set(trans)):
    errors.append(f"漏翻节点: {k}")
n_desc = n_name = n_tip = n_oname = 0
for key, node in trans.items():
    if key not in mappings:
        n_desc += bool(node.get("description"))
        continue
    cls = mappings[key]
    it = cls.INPUT_TYPES()
    valid = set(it.get("required", {})) | set(it.get("optional", {}))
    for param, e in node.get("inputs", {}).items():
        if param not in valid:
            errors.append(f"{key}: 输入名不在 INPUT_TYPES: {param}")
        n_tip += bool(e.get("tooltip")); n_name += bool(e.get("name"))
    for idx in node.get("outputs", {}):
        if int(idx) >= len(cls.RETURN_TYPES):
            errors.append(f"{key}: 输出索引越界 {idx}")
        n_oname += 1
    n_desc += bool(node.get("description"))

if errors:
    print("\n".join(errors)); sys.exit(1)
print(f"校验通过: 节点 {len(trans)}/{len(mappings)}，description {n_desc}，输入名 {n_name}，输入tooltip {n_tip}，输出名 {n_oname}")
