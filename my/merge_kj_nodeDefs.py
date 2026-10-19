# 合并 4 个翻译分片 → locales/zh/nodeDefs.json，并对真实注册表硬校验
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

merged = {}
for i in range(4):
    part = json.load(open(rf"e:\ComfyUI\my\kj_nodeDefs_part{i}.json", encoding="utf8"))
    dup = set(merged) & set(part)
    assert not dup, f"分片间重复键: {dup}"
    merged.update(part)

mappings = kj.NODE_CLASS_MAPPINGS
reg_keys = set(mappings)
trans_keys = set(merged)

errors = []
for k in sorted(trans_keys - reg_keys):
    errors.append(f"多余节点键: {k}")
for k in sorted(reg_keys - trans_keys):
    errors.append(f"漏翻节点: {k}")

n_desc = n_in_tip = n_out_tip = 0
for key, node in merged.items():
    cls = mappings[key]
    try:
        it = cls.INPUT_TYPES()
        valid_inputs = set(it.get("required", {})) | set(it.get("optional", {}))
    except Exception:
        valid_inputs = None
    bad = set(node.get("inputs", {})) - valid_inputs if valid_inputs is not None else set()
    for b in bad:
        errors.append(f"{key}: 输入名不在 INPUT_TYPES: {b}")
    n_out = len(cls.RETURN_TYPES)
    for idx in node.get("outputs", {}):
        if int(idx) >= n_out:
            errors.append(f"{key}: 输出索引越界 {idx}/{n_out}")
    n_desc += bool(node.get("description"))
    n_in_tip += sum(1 for v in node.get("inputs", {}).values() if v.get("tooltip"))
    n_out_tip += sum(1 for v in node.get("outputs", {}).values() if v.get("tooltip"))

if errors:
    print("\n".join(errors))
    print(f"共 {len(errors)} 处问题，不写入")
    sys.exit(1)

out = r"e:\ComfyUI\custom_nodes\comfyui-kjnodes\locales\zh\nodeDefs.json"
json.dump(merged, open(out, "w", encoding="utf8"), ensure_ascii=False, indent=2)
print(f"校验通过并写入: 节点 {len(merged)}/{len(reg_keys)}，description {n_desc}，输入tooltip {n_in_tip}，输出tooltip {n_out_tip}")
