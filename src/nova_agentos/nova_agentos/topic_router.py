# topic_router:独立转发节点。AgentOS 在执行 VLA 类 executor 前调用 MapTopics,
# 把 /nova/env/* 的观测按需转发到 /nova/session/{task_id}/{nid}/* 专属命名空间,
# 并把 executor 的动作从 <ns>/action_cmd 回灌到 /nova/env/action_cmd;执行完 UnmapTopics 撤销。
import copy
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from nova_interfaces.srv import MapTopics, UnmapTopics

_IMAGE_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
_RELIABLE_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

_MSG_TYPES = {
    "image": Image,
    "string": String,
    "float32multi": Float32MultiArray,
    "float32": Float32,
    "bool": Bool,
}


class TopicRouter(Node):
    def __init__(self) -> None:
        super().__init__("nova_topic_router")
        self.declare_parameter("map_service", "/nova/topic_router/map")
        self.declare_parameter("unmap_service", "/nova/topic_router/unmap")
        self._mappings: dict[str, list] = {}
        self.create_service(
            MapTopics, str(self.get_parameter("map_service").value), self._map_cb
        )
        self.create_service(
            UnmapTopics, str(self.get_parameter("unmap_service").value), self._unmap_cb
        )
        self.get_logger().info("topic router ready")

    def _map_cb(self, request, response):
        try:
            if request.mapping_id in self._mappings:
                raise ValueError(f"mapping_id 已存在: {request.mapping_id}")
            n = len(request.src_topics)
            if len(request.dst_topics) != n or len(request.msg_types) != n:
                raise ValueError("src_topics/dst_topics/msg_types 长度必须一致")
            entries = []
            for src, dst, mtype in zip(request.src_topics, request.dst_topics, request.msg_types):
                cls = _MSG_TYPES.get(mtype)
                if cls is None:
                    raise ValueError(f"未知 msg_type: {mtype}")
                qos = _IMAGE_QOS if mtype == "image" else _RELIABLE_QOS
                pub = self.create_publisher(cls, dst, qos)
                sub = self.create_subscription(
                    cls, src, lambda msg, c=cls, p=pub: self._forward(c, p, msg), qos
                )
                entries.append((sub, pub))
                self.get_logger().info(f"map {src} -> {dst} ({mtype})")
            self._mappings[request.mapping_id] = entries
            # 动态 topic 需要 discovery 时间,稍等让订阅方就绪
            time.sleep(0.2)
            response.success = True
            response.message = f"mapped {len(entries)} topics"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _forward(self, cls, pub, msg) -> None:
        # Image 等含可变缓冲区的消息需要重建后再发布
        pub.publish(copy.deepcopy(msg))

    def _unmap_cb(self, request, response):
        entries = self._mappings.pop(request.mapping_id, None)
        if entries is None:
            response.success = False
            response.message = f"mapping_id 不存在: {request.mapping_id}"
            return response
        for sub, pub in entries:
            self.destroy_subscription(sub)
            self.destroy_publisher(pub)
        self.get_logger().info(f"unmap {request.mapping_id} ({len(entries)} topics)")
        response.success = True
        response.message = f"unmapped {len(entries)} topics"
        return response


def main(args=None) -> int:
    rclpy.init(args=args)
    node = TopicRouter()
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
