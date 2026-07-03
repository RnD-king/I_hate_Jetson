import re

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32


class TopicHzState:
    def __init__(self, node: Node, topic: str, prefix: str):
        self.topic = topic
        self.last_time_s = None

        self.gap_ms_pub = node.create_publisher(Float32, f"{prefix}/gap_ms", 10)
        self.inst_hz_pub = node.create_publisher(Float32, f"{prefix}/instant_hz", 10)


class HzTest(Node):
    def __init__(self):
        super().__init__("hz_test")

        self.declare_parameter("image_topics", ["/camera/color/image_raw", "/camera/depth/image_rect_raw"])
        self.declare_parameter("imu_topics", ["/camera/gyro/sample", "/camera/accel/sample"])
        self.declare_parameter("debug_prefix", "/debug/hz")

        self.image_topics = list(self.get_parameter("image_topics").value)
        self.imu_topics = list(self.get_parameter("imu_topics").value)
        self.debug_prefix = str(self.get_parameter("debug_prefix").value).rstrip("/")

        self.states = {}
        self.topic_subscriptions = []

        self._add_topic_subscriptions(self.image_topics, Image)
        self._add_topic_subscriptions(self.imu_topics, Imu)

        self.get_logger().info(
            f"Monitoring {len(self.image_topics)} image topics and {len(self.imu_topics)} imu topics"
        )

    def _add_topic_subscriptions(self, topics, msg_type):
        for topic in topics:
            suffix = self._topic_suffix(topic)
            prefix = f"{self.debug_prefix}/{suffix}"
            self.states[topic] = TopicHzState(self, topic, prefix)
            self.topic_subscriptions.append(
                self.create_subscription(msg_type, topic, lambda msg, t=topic: self.on_message(t), qos_profile_sensor_data)
            )
            self.get_logger().info(f"Monitoring {topic} -> {prefix}")

    @staticmethod
    def _topic_suffix(topic: str) -> str:
        suffix = topic.strip("/").replace("/", "_")
        suffix = re.sub(r"[^a-zA-Z0-9_]", "_", suffix)
        return suffix or "root"

    @staticmethod
    def _publish_float(pub, value: float):
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)

    def on_message(self, topic: str):
        state = self.states[topic]
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9

        if state.last_time_s is None:
            state.last_time_s = now_s
            return

        gap_s = max(0.0, now_s - state.last_time_s)
        state.last_time_s = now_s

        if gap_s <= 0.0:
            return

        instant_hz = 1.0 / gap_s

        self._publish_float(state.gap_ms_pub, gap_s * 1000.0)
        self._publish_float(state.inst_hz_pub, instant_hz)


def main(args=None):
    rclpy.init(args=args)
    node = HzTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
