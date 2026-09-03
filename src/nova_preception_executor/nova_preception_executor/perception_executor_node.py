#!/usr/bin/env python3
# nova_preception_executor:感知类 MCP executor。
#   常驻订阅 /nova/env/camera/* 滚动缓存最新帧;工具调用时查 /nova/env/info 取相机投影矩阵,
#   用 VLM 多视图网格/像素定位获得物体 3D 世界坐标(工具名 locate_object_3d)。
import json
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from nova_common.llm_client import LLMClient
from nova_interfaces.action import MCPExecute
from nova_interfaces.msg import ExecutorHeartbeat, ToolDescriptor
from nova_interfaces.srv import EnvInfo

from nova_preception_executor.vlm_loop import VlmLocator

HEARTBEAT_TOPIC = "/nova/executors/heartbeat"
_CAM_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

DEFAULT_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right"]

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string", "description": "要定位的物体描述,如 '红色杯子'"},
        "camera_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "参与三角化的相机名(默认取节点 camera_names 参数)",
        },
        "grid_size": {"type": "integer", "description": "网格划分大小,默认取节点参数(默认 8,即 8x8)"},
        "max_rounds": {"type": "integer", "description": "最大调整轮数,默认取节点参数(默认 5)"},
        "max_restarts": {"type": "integer", "description": "重投影不一致时的最大重来次数,默认取节点参数(默认 2)"},
        "max_reproj_error_px": {"type": "number", "description": "重投影误差阈值(像素),超过则丢弃重来,默认取节点参数(默认 25)"},
    },
    "required": ["object"],
}


def _image_to_numpy(msg: Image) -> np.ndarray:
    return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))


class PerceptionExecutorNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_preception_executor")
        self.declare_parameter("camera_names", DEFAULT_CAMERAS)
        self.declare_parameter(
            "image_topics", [], ParameterDescriptor(dynamic_typing=True)
        )
        self.declare_parameter("env_ns", "/nova/env")
        self.declare_parameter("heartbeat_rate_hz", 1.0)
        self.declare_parameter("grid_size", 8)
        self.declare_parameter("max_rounds", 5)
        self.declare_parameter("max_restarts", 2)
        self.declare_parameter("max_reproj_error_px", 25.0)
        rate = float(self.get_parameter("heartbeat_rate_hz").value)
        self.camera_names = list(self.get_parameter("camera_names").value)
        self.env_ns = str(self.get_parameter("env_ns").value).rstrip("/")
        topics = self.get_parameter("image_topics").value
        topics = list(topics) if topics else []
        # 未显式给 image_topics 时,按 env_ns/camera/{name}/image_raw 推导,与 camera_names 一一对应
        if topics:
            if len(topics) != len(self.camera_names):
                raise RuntimeError(
                    f"image_topics({len(topics)}) 与 camera_names({len(self.camera_names)}) 数量不一致"
                )
            self.image_topics = dict(zip(self.camera_names, topics))
        else:
            self.image_topics = {
                cam: f"{self.env_ns}/camera/{cam}/image_raw" for cam in self.camera_names
            }
        self.env_info_srv = f"{self.env_ns}/info"
        self._grid_size = int(self.get_parameter("grid_size").value)
        self._max_rounds = int(self.get_parameter("max_rounds").value)
        self._max_restarts = int(self.get_parameter("max_restarts").value)
        self._max_reproj_error_px = float(self.get_parameter("max_reproj_error_px").value)

        self._frames: dict[str, np.ndarray] = {}
        for cam, topic in self.image_topics.items():
            self.create_subscription(
                Image, topic, self._make_cam_cb(cam), _CAM_QOS
            )

        # debug 话题配置:off=不发布;sub=有订阅者才发布;on=总是发布
        self.declare_parameter("vlm_debug_mode", "sub")
        mode = str(self.get_parameter("vlm_debug_mode").value).strip().lower()
        if mode not in ("off", "sub", "on"):
            raise RuntimeError(f"vlm_debug_mode 只支持 off|sub|on, 收到 {mode!r}")
        self._debug_mode = mode

        # 文本回合话题 + 每相机绘制图话题(实际话题 = {vlm_input_topic}/{相机名})
        self.declare_parameter("vlm_round_topic", "/nova/perception/vlm_round")
        self._round_topic = str(self.get_parameter("vlm_round_topic").value).strip()
        self.declare_parameter("vlm_input_topic", "/nova/perception/vlm_input")
        self._input_base = str(self.get_parameter("vlm_input_topic").value).strip().rstrip("/")

        self._info_cg = MutuallyExclusiveCallbackGroup()
        self._info_client = self.create_client(
            EnvInfo, self.env_info_srv, callback_group=self._info_cg
        )
        self._llm = LLMClient(vision=True)
        self._round_pub = None
        self._cam_pubs: dict[str, Any] = {}
        if self._debug_mode != "off":
            if self._round_topic:
                self._round_pub = self.create_publisher(String, self._round_topic, 1)
            if self._input_base:
                for cam in self.camera_names:
                    topic = f"{self._input_base}/{cam}"
                    self._cam_pubs[cam] = self.create_publisher(Image, topic, 5)

        self._action_server = ActionServer(
            self,
            MCPExecute,
            f"/{self.get_name()}/locate_object_3d/execute",
            self._execute_cb,
        )
        self._heartbeat_pub = self.create_publisher(ExecutorHeartbeat, HEARTBEAT_TOPIC, 1)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_heartbeat)
        self._publish_heartbeat()
        self.get_logger().info(
            f"感知 executor 就绪,相机={self.camera_names}, "
            f"话题={list(self.image_topics.values())}, LLM vision providers={[p.get('name') for p in self._llm.providers]}, "
            f"vlm_debug_mode={self._debug_mode}, vlm 图像话题={[f'{self._input_base}/{c}' for c in self._cam_pubs]}"
        )

    def _publish_heartbeat(self) -> None:
        hb = ExecutorHeartbeat()
        hb.executor_name = self.get_name()
        tool = ToolDescriptor()
        tool.name = "locate_object_3d"
        tool.description = (
            "多视图 VLM 3D 定位:读取 /nova/env/* 最新相机帧与投影矩阵,"
            "先让 VLM 在各图上用网格单元粗定位,再三角化并反投影,"
            "把上一轮点(灰)与重投影点(黑)画回图上让 VLM 迭代微调像素坐标,"
            "直到其满意,返回物体 3D 世界坐标(x,y,z)。"
        )
        tool.params_schema_json = json.dumps(TOOL_SCHEMA, ensure_ascii=False)
        tool.action_server_name = f"/{self.get_name()}/locate_object_3d/execute"
        hb.tools.append(tool)
        self._heartbeat_pub.publish(hb)

    def _make_cam_cb(self, cam: str):
        def cb(msg):
            self._frames[cam] = _image_to_numpy(msg)

        return cb

    def _fetch_projections(self) -> dict:
        if not self._info_client.service_is_ready():
            if not self._info_client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"env info service {self.env_info_srv} 不可用")
        future = self._info_client.call_async(EnvInfo.Request())
        deadline = time.time() + 5.0
        while rclpy.ok() and not future.done():
            if time.time() > deadline:
                raise RuntimeError("查询 env info 超时")
            time.sleep(0.02)
        response = future.result()
        if not response.success:
            raise RuntimeError(f"env info 失败: {response.message}")
        info = json.loads(response.spec_json)
        return info.get("cameras", {})

    # 请求相机名 -> 投影矩阵键:先精确匹配,再按包含关系兜底
    @staticmethod
    def _match_projection(cam: str, projections: dict) -> str | None:
        if cam in projections:
            return cam
        for key in projections:
            if cam in key or key in cam:
                return key
        return None

    def _execute_cb(self, goal_handle):
        goal = goal_handle.request
        try:
            params = json.loads(goal.params_json) if goal.params_json else {}
            result_json = self._locate(params, task_id=goal.trace_id)
            result = MCPExecute.Result()
            result.success = True
            result.result_json = json.dumps(result_json, ensure_ascii=False)
            result.error = ""
            goal_handle.succeed()
            self.get_logger().info(f"locate_object_3d 完成: {result_json.get('position')}")
            return result
        except Exception as exc:
            result = MCPExecute.Result()
            result.success = False
            result.result_json = ""
            result.error = str(exc)
            goal_handle.abort()
            self.get_logger().error(f"locate_object_3d 失败: {exc}")
            return result

    def _locate(self, params: dict, task_id: str = "") -> dict:
        object_desc = str(params.get("object", "")).strip()
        if not object_desc:
            return {"ok": False, "error": "缺少 object 参数"}

        projections = self._fetch_projections()
        if not projections:
            return {"ok": False, "error": "env info 未提供相机投影矩阵(需升级 robocasa_sim_server)"}

        cams = params.get("camera_names") or self.camera_names
        images, projs = {}, {}
        for cam in cams:
            key = self._match_projection(cam, projections)
            frame = self._frames.get(cam)
            if frame is None:
                self.get_logger().warn(f"相机 {cam} 未收到帧")
                continue
            if key is None:
                self.get_logger().warn(f"相机 {cam} 无投影矩阵(可用:{list(projections)})")
                continue
            images[cam] = frame
            # cameras 值形如 {"intrinsics":..., "projection": 3x4},三角化只用到 projection
            entry = projections[key]
            if isinstance(entry, dict) and "projection" in entry:
                entry = entry["projection"]
            projs[cam] = entry

        locator = VlmLocator(
            self._llm,
            grid_size=int(params.get("grid_size", self._grid_size)),
            max_rounds=int(params.get("max_rounds", self._max_rounds)),
            max_restarts=int(params.get("max_restarts", self._max_restarts)),
            max_reproj_error_px=float(params.get("max_reproj_error_px", self._max_reproj_error_px)),
        )
        return locator.locate(
            object_desc,
            images,
            projs,
            agent_context=str(params.get("_agent_context", "")),
            task_id=task_id,
            on_round=self._publish_vlm_round,
        )

    # debug 话题是否该发:on 恒发;sub 有订阅者才发;off 时 publisher 未创建
    def _want_debug_pub(self, pub) -> bool:
        if pub is None:
            return False
        if self._debug_mode == "on":
            return True
        if self._debug_mode == "sub":
            return pub.get_subscription_count() > 0
        return False

    # VlmLocator 每轮回调:绘制图按相机发到各自话题,回合文本发到 vlm_round
    def _publish_vlm_round(self, payload: dict) -> None:
        task_id = str(payload.get("task_id", ""))
        round_tag = str(payload.get("round", ""))
        if self._debug_mode != "off":
            prompt = str(payload.get("prompt", ""))
            reply = str(payload.get("reply", ""))
            self.get_logger().info(
                f"[vlm] task={task_id} round={round_tag}\nprompt: {prompt}\nreply: {reply}"
            )
        for cam, url in (payload.get("images") or {}).items():
            pub = self._cam_pubs.get(cam)
            if not self._want_debug_pub(pub):
                continue
            try:
                frame_id = f"{task_id}|{round_tag}|{cam}"
                pub.publish(self._data_url_to_image_msg(url, frame_id))
            except Exception as exc:
                self.get_logger().warn(f"发布 vlm 输入图 {cam} 失败: {exc}")
        if not self._want_debug_pub(self._round_pub):
            return
        try:
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._round_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"发布 vlm_round 失败: {exc}")

    @staticmethod
    def _data_url_to_image_msg(data_url: str, frame_id: str) -> Image:
        import base64
        import io

        from PIL import Image as PILImage

        b64 = data_url.split(",", 1)[1]
        jpeg = base64.b64decode(b64)
        image = np.asarray(PILImage.open(io.BytesIO(jpeg)).convert("RGB"))
        image = np.ascontiguousarray(image)
        msg = Image()
        msg.header.frame_id = frame_id
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(image.shape[1] * 3)
        msg.data = image.tobytes()
        return msg

    def destroy_node(self) -> bool:
        return super().destroy_node()


def main(args=None) -> int:
    rclpy.init(args=args)
    node = PerceptionExecutorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
