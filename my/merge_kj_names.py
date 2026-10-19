# 合并显示名翻译（4 分片 + 1 补翻）进 nodeDefs.json，并对真实注册表校验
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

path = r"e:\ComfyUI\custom_nodes\comfyui-kjnodes\locales\zh\nodeDefs.json"
nodeDefs = json.load(open(path, encoding="utf8"))
mappings = kj.NODE_CLASS_MAPPINGS

n_name = n_oname = 0
for i in range(4):
    part = json.load(open(rf"e:\ComfyUI\my\kj_names_part{i}.json", encoding="utf8"))
    for key, node in part.items():
        assert key in mappings, f"未知节点键: {key}"
        for param, name in node.get("inputs", {}).items():
            nodeDefs[key].setdefault("inputs", {}).setdefault(param, {})["name"] = name
            n_name += 1
        for idx, name in node.get("outputs", {}).items():
            nodeDefs[key].setdefault("outputs", {}).setdefault(idx, {})["name"] = name
            n_oname += 1
fixed = json.load(open(r"e:\ComfyUI\my\kj_names_fixed.json", encoding="utf8"))
for key, node in fixed.items():
    for param, name in node.get("inputs", {}).items():
        nodeDefs[key].setdefault("inputs", {}).setdefault(param, {})["name"] = name
        n_name += 1

# 校验：输入参数名 ⊆ INPUT_TYPES，输出索引越界检查
errors = []
n_tip = 0
for key, node in nodeDefs.items():
    cls = mappings[key]
    it = cls.INPUT_TYPES()
    valid = set(it.get("required", {})) | set(it.get("optional", {}))
    for param, entry in node.get("inputs", {}).items():
        if param not in valid:
            errors.append(f"{key}: 输入名不在 INPUT_TYPES: {param}")
        n_tip += bool(entry.get("tooltip"))
    for idx in node.get("outputs", {}):
        if int(idx) >= len(cls.RETURN_TYPES):
            errors.append(f"{key}: 输出索引越界 {idx}")

if errors:
    print("\n".join(errors[:30]))
    sys.exit(1)

json.dump(nodeDefs, open(path, "w", encoding="utf8"), ensure_ascii=False, indent=2)
print(f"合并完成: 输入显示名 +{n_name}，输出显示名 +{n_oname}；tooltip 总数 {n_tip}；节点 {len(nodeDefs)}")
