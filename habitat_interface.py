#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import numpy as np
import sys, os
from threading import Thread
from std_msgs.msg import String
from textwrap import dedent
import argparse
import math

from scipy.spatial.transform import Rotation as R

from utils.socket_server import run_server, format_data
from utils.path_following_utils import WaypointFollower
from tf2_ros import Buffer, TransformListener, TransformException

from openai import OpenAI
import time

# Global shared state
_latest_rgb = None
_latest_depth = None
_latest_position = None
_latest_quat_xyzw = None
_scenario = None
_info = None
_host = "localhost"
_port = 12357


def _wrap_to_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _world_to_body(dx: float, dy: float, yaw: float):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c*dx - s*dy, s*dx + c*dy


def make_args():
    parser = argparse.ArgumentParser(description="Waypoint follower parser...")
    parser.add_argument("--waypoint_stride", type=int, default=15)
    parser.add_argument("--arrive_thresh", type=float, default=0.25)
    parser.add_argument("--max_v", type=float, default=0.6)
    parser.add_argument("--max_yaw_rate", type=float, default=1.0)
    parser.add_argument("--k_p_ang", type=float, default=1.5)
    parser.add_argument("--ky", type=float, default=0.0)
    parser.add_argument("--min_forward", type=float, default=0.05)
    parser.add_argument("--scene_id", type=str, default="test_sc")
    parser.add_argument("--episode_id", type=str, default="test_ep")
    parser.add_argument("--instruction", type=str, default="fireplace.")
    parser.add_argument("--robot_height", type=float, default=0.44)
    parser.add_argument("--hfov_deg", type=float, default=90.0)
    parser.add_argument("--metrics", type=str, default="{}")
    return parser.parse_args()


class HabitatROSBridge(Node):
    def __init__(self, args=None):
        super().__init__('habitat_ros_bridge')
        print("[HabitatROSBridge] Initializing Habitat ROS Bridge")
        self.args_cli = args or make_args()
        self.bridge = CvBridge()
        self._ros_publishers = {}
        self.old_task = None


        self.openai_api_key = "EMPTY"
        self.openai_api_base = "http://localhost:8000/v1"
        self.instruction = None
        self.prompt = dedent("""You are an expert AI at identifying the primary target object from a user's instruction. Your task is to extract or infer the goal object.
        The output must be a single word or short descriptive noun phrase. Do not include location details or general adjectives unless essential. Your Task: Instruction: 
        """)

        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_api_base)

        # wait for the openai client to be initialized
        while True:
            try:
                self.client.chat.completions.create(
                    model="Qwen/Qwen3-4B-Instruct-2507",
                    messages=[{"role": "user", "content": "Hello, world!"}],
                    max_tokens=10,
                )
                break
            except Exception as e:
                print(f"[HabitatROSBridge] Error initializing OpenAI client: {e}")
                time.sleep(1)

        # topic names
        self.robot_topic = "/spot"
        self.action_topic = f"{self.robot_topic}/cmd_vel"
        self.camera_topic = f"{self.robot_topic}/camera/frontright/image/compressed"
        self.depth_topic = f"{self.robot_topic}/depth/frontright/image"
        self.odom_topic = f"{self.robot_topic}/platform/odom"
        self.scenario_topic = f"/scenario"
        self.sim_control_topic = f"/sim_control"
        # Subscribers
        self.create_subscription(CompressedImage, self.camera_topic, self.rgb_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.subscription = self.create_subscription(String, self.scenario_topic, self.scenario_ros_callback, 10)

        qos = QoSProfile(depth=10)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.sim_control_pub = self.create_publisher(String, self.sim_control_topic, qos)
        cam_msg = String()
        cam_msg.data = self.camera_topic
        self.sim_control_pub.publish(cam_msg)

        # Timer for path follower
        self.path_follower_timer = self.create_timer(0.05, self.path_follower_callback)
        self.follower = WaypointFollower(
            device="cpu",
            lookahead_distance=0.5,
            max_vel=[0.1, 0.05, 0.2],
            min_lin_speed=0.05,
            min_ang_speed=0.03
        )

    def init_360(self):
        global _latest_quat_xyzw

        # Ensure we have odometry
        while _latest_quat_xyzw is None:
            rclpy.spin_once(self, timeout_sec=0.01)

        # Starting yaw
        start_yaw = R.from_quat(_latest_quat_xyzw).as_euler('xyz')[2]

        while rclpy.ok():
            # Process odom callback so yaw updates
            rclpy.spin_once(self, timeout_sec=0.001)

            current_yaw = R.from_quat(_latest_quat_xyzw).as_euler('xyz')[2]

            # Normalize angle to [-pi, pi]
            diff = (current_yaw - start_yaw + np.pi) % (2 * np.pi) - np.pi
            rotated = abs(diff)

            if rotated >= 2 * np.pi:
                break

            # Command rotation
            move_cmd = Twist()
            move_cmd.angular.z = -0.2
            self.publish(self.action_topic, move_cmd)

        # Stop after rotation
        self.publish(self.action_topic, Twist())
        
    # ---- Callbacks ----
    def rgb_callback(self, msg):
        global _latest_rgb
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            _latest_rgb = cv_image[..., ::-1].astype(np.uint8)
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def depth_callback(self, msg):
        global _latest_depth
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            _latest_depth = np.nan_to_num(cv_image, nan=0, posinf=0, neginf=0).astype(np.uint16)
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def odom_callback(self, msg):
        global _latest_position, _latest_quat_xyzw
        try:
            _latest_position = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ], dtype=np.float32)
            _latest_quat_xyzw = np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Odom callback error: {e}")

    def scenario_ros_callback(self, msg):
        global _scenario
        if msg.data != self.old_task:
            print(f"New scenario received: {msg.data}")
            self.old_task = msg.data
            self.instruction = msg.data
            llm_processed_task = self.client.chat.completions.create(
                model="Qwen/Qwen3-4B-Instruct-2507",
                messages=[{"role": "user", "content": self.prompt + msg.data}],
                max_tokens=300,
                temperature=0.6,
                top_p=0.95
            )
            print(f"[TaskListener] Processed task: {llm_processed_task.choices[0].message.content}")
            _scenario = llm_processed_task.choices[0].message.content

    def publish(self, topic_name, msg):
        if topic_name not in self._ros_publishers:
            self._ros_publishers[topic_name] = self.create_publisher(type(msg), topic_name, 10)
        self._ros_publishers[topic_name].publish(msg)


    # ---- Server callbacks ----
    def data_callback(self, request_type):
        global _info
        if request_type == "GET_SENSOR_DATA":
            _info = {
                "scene_id": "test_ep",
                "episode_id": 0,
                "instruction": _scenario,
                "robot_height": self.args_cli.robot_height,
                "hfov_deg": self.args_cli.hfov_deg,
                "metrics": {}
            }
            if _latest_rgb is None or _latest_depth is None or _latest_position is None or _latest_quat_xyzw is None:
                return None
            return format_data(_latest_rgb, _latest_depth, _latest_position, _latest_quat_xyzw, _info, "hab_interface")
        elif request_type == "GET_EPISODE_LIST":
            episode_set_list = {
                "all": ["test_ep_0"]*100,
                "mini_test": ["test_ep_0"]*100
            }
            return episode_set_list

    def action_callback(self, msg_type, message):
        if msg_type == 'VEL':
            move_command = Twist()
            move_command.linear.x = float(message['vy'])
            move_command.linear.y = float(message['vx'])
            move_command.angular.z = float(message['vw'])
            self.publish(self.action_topic, move_command)
        elif msg_type == 'WAYPOINT':
            print("Received waypoint message")
            if _latest_position is None or _latest_quat_xyzw is None:
                return

            # Convert waypoints to include yaw
            #waypoints = set_yaw_to_forward(_latest_position, message["waypoint"])

            # Assign waypoints and reset
            self.follower.waypoints = message["waypoint"]
            self.follower.arrived_at_goal = False
            self.follower.reset()
        elif msg_type == 'STOP':
            self.pub_task_submission()
            self.pub_task_complete()
        elif msg_type == 'EPISODE':
            # this will trigger a reset of the vln sim
            #TODO: implement episode loading
            pass
            #self.load_episode(message["episode_label"])

    def path_follower_callback(self):
        if _latest_position is None or _latest_quat_xyzw is None:
            # print("[PATH FOLLOWER] Waiting for position and orientation...")
            return
        
        if self.follower is not None and hasattr(self.follower, "waypoints"):# and not self.follower.arrived_at_goal:
            # print("[PATH FOLLOWER] Updating follower...")
            #print(f"[PATH FOLLOWER] Current waypoints (interface): {self.follower.waypoints}")
            # Pass the current waypoints explicitly
            cmd = self.follower.update(_latest_position, _latest_quat_xyzw, self.follower.waypoints, verbose=False)
            vx, vy, omega = cmd[0].cpu().numpy()

            move_command = Twist()
            move_command.linear.x = float(vx)
            move_command.linear.y = float(vy)
            move_command.angular.z = float(omega)
            self.publish(self.action_topic, move_command)

            if self.follower.arrived_at_goal:
                print("[PLAN] Goal reached.")

    # Once the simulation is running, your agent must publish task information on
    # the /task_submission topic. For example:
    #
    #   ros2 topic pub /task_submission std_msgs/msg/String \
    #       "data: '/spot/camera/head_rgb_left/image/compressed, (576, 192, 384, 0, 768, 384), top-right crop'"
    #
    # This message instructs the system to inspect the current observation from the
    # head_rgb_left camera, crop the image using the last four coordinates
    # (384, 0, 768, 384), and place a red dot at the location of the first two
    # coordinates (576, 192). In this example, the red dot ends up being at the center of the cropped image.
    def pub_task_submission(
        self,
        dot_x: int = 576,
        dot_y: int = 192,
        x_min: int = 384,
        y_min: int = 0,
        x_max: int = 768,
        y_max: int = 384,
        crop_description: str = "top-right crop",
    ):
        # ros2 topic pub /task_submission std_msgs/msg/String "data: '/spot/camera/head_rgb_left/image/compressed, (576, 192, 384, 0, 768, 384), top-right crop'"
        print("[TASK SUBMISSION] Submitting task")
        task_sub_msg = String()
        camera_source = "/spot/camera/head_rgb_right/image/compressed"
        task_sub_msg.data = f"{camera_source}, ({dot_x}, {dot_y}, {x_min}, {y_min}, {x_max}, {y_max}), {crop_description}"

        self.sim_control_pub.publish(task_sub_msg)

    # Once the agent determines that it has completed the task, call the following function
    # to complete the task. The simulation will then move to the next task, publishing it
    # to /scenario and saving all images and descriptions to a JSON.
    def pub_task_complete(self):
        print("[TASK COMPLETE] Task complete")
        task_complete_msg = String()
        task_complete_msg.data = "Task Complete"
        self.sim_control_pub.publish(task_complete_msg)


# Entrypoint
def main(args=None):
    rclpy.init(args=args)
    node = HabitatROSBridge()

    server_thread = Thread(target=run_server, kwargs={
        "data_cb": node.data_callback,
        "action_cb": node.action_callback,
        "host": _host,
        "port": _port
    })
    server_thread.daemon = True
    server_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()