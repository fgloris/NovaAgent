#!/usr/bin/env python3
# AgentOS 主节点:接收指令 -> skill 注入规划 -> DAG 校验 -> 调度执行。
# 对声明了 obs_bindings 的 VLA 工具,执行前通过 topic_router 建立专属命名空间转发,执行完撤销。
import json
import uuid
from pathlib import Path

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from nova_common.llm_client import LLMClient
from nova_interfaces.msg import TaskState
from nova_interfaces.srv import EnvInfo, MapTopics, RunTask, UnmapTopics

from nova_agentos.dag_executor import DagExecutor
from nova_agentos.dag_validator import validate
from nova_agentos.mcp_adapter import McpAdapter
from nova_agentos.planner import Planner
from nova_agentos.skill_store import SkillStore


class AgentosNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_agentos")
        self.declare_parameter("skills_dir", "")
        self.declare_parameter("list_tools_service", "/nova/executor_manager/list_tools")
        self.declare_parameter("execute_action", "/nova/executor_manager/execute")
        self.declare_parameter("run_task_service", "/nova/agentos/run")
        self.declare_parameter("map_topics_service", "/nova/topic_router/map")
        self.declare_parameter("unmap_topics_service", "/nova/topic_router/unmap")
        self.declare_parameter("env_info_service", "/nova/env/info")

        skills_dir = str(self.get_parameter("skills_dir").value)
        if not skills_dir:
            from ament_index_python.packages import get_package_share_directory
            skills_dir = str(Path(get_package_share_directory("nova_agentos")) / "skills")

        self.skills = SkillStore(skills_dir)
        self.llm = LLMClient()
        self.planner = Planner(self.llm, self.skills)
        self.adapter = McpAdapter(
            self,
            list_tools_srv=str(self.get_parameter("list_tools_service").value),
            execute_action=str(self.get_parameter("execute_action").value),
        )
        self.dag_executor = DagExecutor(self.adapter)
        self.state_pub = self.create_publisher(TaskState, "/nova/agentos/task_state", 10)

        # 独立 callback group:run 回调内同步等待响应,避免与默认组互斥卡死
        self._client_cg = MutuallyExclusiveCallbackGroup()
        self._map_client = self.create_client(
            MapTopics, str(self.get_parameter("map_topics_service").value), callback_group=self._client_cg
        )
        self._unmap_client = self.create_client(
            UnmapTopics, str(self.get_parameter("unmap_topics_service").value), callback_group=self._client_cg
        )
        self._env_info_client = self.create_client(
            EnvInfo, str(self.get_parameter("env_info_service").value), callback_group=self._client_cg
        )

        self._bindings: dict[str, dict] = {}
        self._env_info: dict | None = None
        self._active_mappings: dict[str, bool] = {}

        self.create_service(
            RunTask, str(self.get_parameter("run_task_service").value), self._run_task_cb
        )
        self.get_logger().info(f"AgentOS 就绪,skill 目录: {skills_dir}")

    def _run_task_cb(self, request, response):
        task_id = uuid.uuid4().hex[:8]
        instruction = request.instruction
        self._publish_state(task_id, instruction, "planning")
        try:
            descriptors = self.adapter.fetch_tools()
            self._cache_bindings(descriptors)
            tools_text = "\n".join(f"- {d.name}: {d.description}" for d in descriptors)
            graph = self.planner.plan(instruction, tools_text)
            validate(graph, [d.name for d in descriptors])
            plan_json = json.dumps(graph, ensure_ascii=False)
            self.get_logger().info(f"LLM DAG: {plan_json}")
            self._publish_state(task_id, instruction, "executing", plan_json)
            self._env_info = self._fetch_env_info()

            def on_step(nid):
                self._publish_state(task_id, instruction, "executing", plan_json, current_step=nid)

            def on_before_node(nid, n):
                bindings = self._bindings.get(n["tool_name"])
                if not bindings:
                    return None
                mapping_id = f"{task_id}_{nid}"
                # ROS 2 topic 名不允许冒号,也不允许 token 以数字开头,
                ns = f"/nova/session/s_{mapping_id}"
                src, dst, types, cam_selected = self._build_mapping(ns, bindings)
                if not src:
                    return None
                if not self._call_map_topics(mapping_id, src, dst, types):
                    raise RuntimeError(f"MapTopics 失败: {mapping_id}")
                self._active_mappings[mapping_id] = True
                extra = {"topic_namespace": ns}
                if cam_selected:
                    extra["camera_names"] = cam_selected
                return extra

            def on_after_node(nid, n):
                mapping_id = f"{task_id}_{nid}"
                if mapping_id in self._active_mappings:
                    self._call_unmap_topics(mapping_id)
                    self._active_mappings.pop(mapping_id, None)

            self.dag_executor.execute(
                graph,
                task_id,
                on_step=on_step,
                on_before_node=on_before_node,
                on_after_node=on_after_node,
            )
            self._publish_state(task_id, instruction, "done", plan_json)
            response.task_id = task_id
            response.success = True
            response.message = "任务完成"
        except Exception as exc:
            self._cleanup_mappings()
            self.get_logger().error(f"任务 {task_id} 失败: {exc}")
            self._publish_state(task_id, instruction, "failed", detail=str(exc))
            response.task_id = task_id
            response.success = False
            response.message = str(exc)
        return response

    # tool_name -> obs_bindings(JSON dict)
    def _cache_bindings(self, descriptors) -> None:
        self._bindings.clear()
        for d in descriptors:
            if not d.obs_bindings:
                continue
            try:
                self._bindings[d.name] = json.loads(d.obs_bindings)
            except json.JSONDecodeError:
                self.get_logger().warn(f"工具 {d.name} 的 obs_bindings 不是合法 JSON: {d.obs_bindings}")

    def _fetch_env_info(self) -> dict | None:
        if not self._env_info_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("env_info service 不可用,跳过仿真自发现")
            return None
        future = self._env_info_client.call_async(EnvInfo.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done():
            self.get_logger().warn("env_info 调用超时,跳过仿真自发现")
            return None
        resp = future.result()
        if not resp.success or not resp.spec_json:
            self.get_logger().warn(f"env_info 无有效 spec(success={resp.success}, spec_json={bool(resp.spec_json)})")
            return None
        info = json.loads(resp.spec_json)
        cameras = (info.get("obs_spec") or {}).get("cameras") or {}
        self.get_logger().info(f"env_info 就绪, cameras={list(cameras.keys())}")
        return info

    # 组装转发映射:bindings 声明相机(可为逻辑名)→ env_info 自发现的实际相机;动作回路固定。
    # 返回 (src, dst, types, cam_selected),cam_selected 为实际命中的相机名,供注入给 executor。
    def _build_mapping(self, ns: str, bindings: dict) -> tuple[list, list, list, list]:
        src, dst, types = [], [], []
        cameras = {}
        if self._env_info:
            cameras = (self._env_info.get("obs_spec") or {}).get("cameras") or {}
        cam_bindings = bindings.get("cameras") or []
        if cam_bindings:
            cam_selected = self._match_cameras(cam_bindings, cameras)
        else:
            cam_selected = list(cameras.keys())
        for cam in cam_selected:
            src.append(f"/nova/env/camera/{cam}/image_raw")
            dst.append(f"{ns}/camera/{cam}/image_raw")
            types.append("image")
        self.get_logger().info(
            f"build_mapping ns={ns} cameras(env)={list(cameras.keys())} "
            f"cam_bindings={cam_bindings} cam_selected={cam_selected}"
        )
        if bindings.get("state"):
            src.append("/nova/env/obs")
            dst.append(f"{ns}/state")
            types.append("string")
        src.append(f"{ns}/action_cmd")
        dst.append("/nova/env/action_cmd")
        types.append("float32multi")
        return src, dst, types, cam_selected

    # bindings 逻辑名 -> env 实际相机名:精确匹配优先,其次子串包含匹配(如 agentview -> video_robot0_agentview_left)
    def _match_cameras(self, cam_bindings: list, cameras: dict) -> list:
        matched = []
        seen = set()
        for b in cam_bindings:
            cands = [c for c in cameras if c == b] or [c for c in cameras if b in c]
            if not cands:
                self.get_logger().warn(f"bindings 相机 {b} 在 env_info 中未找到")
                continue
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    matched.append(c)
        return matched

    def _call_map_topics(self, mapping_id, src, dst, types) -> bool:
        if not self._map_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("topic_router MapTopics service 不可用")
        req = MapTopics.Request()
        req.mapping_id = mapping_id
        req.src_topics = src
        req.dst_topics = dst
        req.msg_types = types
        future = self._map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done():
            raise RuntimeError("MapTopics 调用超时")
        resp = future.result()
        if not resp.success:
            raise RuntimeError(f"MapTopics 失败: {resp.message}")
        return True

    def _call_unmap_topics(self, mapping_id) -> bool:
        if not self._unmap_client.wait_for_service(timeout_sec=5.0):
            return False
        req = UnmapTopics.Request()
        req.mapping_id = mapping_id
        future = self._unmap_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done():
            return False
        return bool(future.result().success)

    # 异常兜底:清理尚未走 on_after_node 的残留映射,避免 topic 泄漏
    def _cleanup_mappings(self) -> None:
        for mapping_id in list(self._active_mappings):
            try:
                self._call_unmap_topics(mapping_id)
            except Exception:
                self.get_logger().error(f"清理映射 {mapping_id} 失败")
        self._active_mappings.clear()

    def _publish_state(self, task_id, instruction, status, plan_json="", current_step="", detail=""):
        msg = TaskState()
        msg.task_id = task_id
        msg.instruction = instruction
        msg.status = status
        msg.plan_json = plan_json
        msg.current_step = current_step
        msg.detail = detail
        self.state_pub.publish(msg)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = AgentosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
