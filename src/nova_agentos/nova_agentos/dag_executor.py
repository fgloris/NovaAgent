# DAG 执行器:按拓扑序逐节点调用工具,支持 $ref 引用前序节点结果。
import json

from nova_agentos.dag_validator import topo_order


class DagExecutor:
    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def execute(
        self,
        graph: dict,
        task_id: str,
        on_step=None,
        on_before_node=None,
        on_after_node=None,
    ) -> dict[str, dict]:
        nodes = {n["id"]: n for n in graph["nodes"]}
        results: dict[str, dict] = {}
        for nid in topo_order(graph["nodes"]):
            n = nodes[nid]
            params = self._resolve_params(n.get("params_json") or "{}", results)
            extra = on_before_node(nid, n) if on_before_node else None
            if extra:
                params.update(extra)
            try:
                if on_step:
                    on_step(nid)
                result = self.adapter.execute(
                    n["tool_name"], params, trace_id=f"{task_id}:{nid}"
                )
            finally:
                # 异常路径也必须执行清理(on_after_node 只会在 on_before_node 返回非 None 时调用)
                if extra and on_after_node:
                    on_after_node(nid, n)
            results[nid] = result
        return results

    # 递归把 params 里的 "$ref": "<节点id>" 替换为对应节点执行结果
    @staticmethod
    def _resolve_params(params_json: str, results: dict[str, dict]):
        data = json.loads(params_json) if params_json else {}
        return DagExecutor._sub_ref(data, results)

    @staticmethod
    def _sub_ref(value, results):
        if isinstance(value, dict):
            if "$ref" in value and set(value.keys()) == {"$ref"}:
                ref = value["$ref"]
                if ref not in results:
                    raise ValueError(f"引用节点 {ref} 尚无结果(依赖顺序错误)")
                return results[ref]
            return {k: DagExecutor._sub_ref(v, results) for k, v in value.items()}
        if isinstance(value, list):
            return [DagExecutor._sub_ref(v, results) for v in value]
        return value
