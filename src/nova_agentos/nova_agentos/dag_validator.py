# DAG 校验:id 唯一、type 合法、依赖存在、工具已注册、参数可解析、无环。
import json


class DagError(Exception):
    pass


def validate(graph: dict, tool_names: list[str]) -> list[str]:
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        raise DagError("任务图为空:缺少 nodes")

    ids = []
    seen = set()
    tool_set = set(tool_names)
    for n in nodes:
        nid = n.get("id")
        if not nid or nid in seen:
            raise DagError(f"节点 id 缺失或重复: {nid}")
        seen.add(nid)
        ids.append(nid)

        if n.get("type", "tool") != "tool":
            raise DagError(f"节点 {nid}: 不支持的 type '{n.get('type')}'")

        tool = n.get("tool_name")
        if tool not in tool_set:
            raise DagError(f"节点 {nid}: 工具 '{tool}' 未注册(可用: {', '.join(sorted(tool_set))})")

        deps = n.get("depends_on") or []
        if not isinstance(deps, list):
            raise DagError(f"节点 {nid}: depends_on 必须是列表")
        for dep in deps:
            if dep not in seen:
                raise DagError(f"节点 {nid}: 依赖 '{dep}' 不存在")

        params = n.get("params_json")
        if params:
            try:
                json.loads(params)
            except json.JSONDecodeError as exc:
                raise DagError(f"节点 {nid}: params_json 非法: {exc}")

    return topo_order(nodes)


# 拓扑排序,检测环
def topo_order(nodes: list[dict]) -> list[str]:
    order = []
    remaining = {n["id"]: set(n.get("depends_on") or []) for n in nodes}
    while remaining:
        ready = [nid for nid, deps in remaining.items() if not deps]
        if not ready:
            raise DagError(f"DAG 存在环: 剩余节点 {list(remaining)}")
        ready.sort()
        order.extend(ready)
        for nid in ready:
            remaining.pop(nid)
        for deps in remaining.values():
            deps.difference_update(ready)
    return order
