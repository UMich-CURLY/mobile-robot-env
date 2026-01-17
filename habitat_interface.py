#!/usr/bin/env python3
from functools import partial
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage
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
from message_filters import Subscriber, ApproximateTimeSynchronizer
from openai import OpenAI
import time

# Global shared state
_latest_rgb = None
_latest_depth = None
_latest_rgb_by_source = {}
_latest_depth_by_source = {}
_latest_position = None
_latest_quat_xyzw = None
_latest_tfs = {}
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
    parser.add_argument("--dummy_llm", action="store_true")
    return parser.parse_args()


def _topic_to_key(topic_name: str) -> str:
    if "/image/" in topic_name:
        key = topic_name.split("/image/")[-1]
        key = key.replace("/compressed", "").replace("/image_raw", "")
        return key.strip("/")
    if "/depth/" in topic_name:
        key = topic_name.split("/depth/")[-1]
        key = key.replace("/image_raw", "")
        return key.strip("/")
    return topic_name.strip("/").split("/")[-1]


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
        if not self.args_cli.dummy_llm:
            while True:
                try:
                    self.client.chat.completions.create(
                        model="Qwen/Qwen3-4B-Instruct-2507",
                        messages=[{"role": "user", "content": "Hello, world!"}],
                        max_tokens=10,
                    )
                    break
                except Exception as e:
                    print(f"[HabitatROSBridge] Waiting for OpenAI client to be initialized...")
                    time.sleep(1)
            print("[HabitatROSBridge] OpenAI client initialized.")


        ## Cam Subscribers
        rgb_topics = {
            "frontright": "/spot/camera/frontright/image/compressed",
            "frontleft": "/spot/camera/frontleft/image/compressed",
            "left": "/spot/camera/left/image/compressed",
            "right": "/spot/camera/right/image/compressed",
            "back": "/spot/camera/back/image/compressed",
        }

        depth_topics = {
            "frontright": "/spot/depth/frontright/image",
            "frontleft": "/spot/depth/frontleft/image",
            "left": "/spot/depth/left/image",
            "right": "/spot/depth/right/image",
            "back": "/spot/depth/back/image",
        }
        
        # RGB subscriptions
        for name, topic in rgb_topics.items():
            self.create_subscription(
                CompressedImage,
                topic,
                partial(self.rgb_callback, source=name),
                10
            )

        # Depth subscriptions
        for name, topic in depth_topics.items():
            self.create_subscription(
                Image,
                topic,
                partial(self.depth_callback, source=name),
                10
        )
                  
        # topic names
        self.robot_topic = "/spot"
        self.action_topic = f"{self.robot_topic}/cmd_vel"
        self.camera_topic = f"{self.robot_topic}/camera/frontright/image/compressed"
        self.depth_topic = f"{self.robot_topic}/depth/frontright/image"
        self.odom_topic = f"{self.robot_topic}/platform/odom"
        self.scenario_topic = f"/scenario"
        self.sim_status_topic = f"/sim_status"
        self.sim_control_topic = f"/sim_control"
        self.sim_settings_topic = f"/sim_settings"
        self.task_submission_topic = f"/task_submission"
        # Subscribers
        self.rgb_sub = Subscriber(self, CompressedImage, self.camera_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.odom_sub = Subscriber(self, Odometry, self.odom_topic)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.odom_sub], 100, 0.1, allow_headerless=False)
        self.sync.registerCallback(self.sensor_callback)
        self.scenario_sub = self.create_subscription(String, self.scenario_topic, self.scenario_ros_callback, 10)

        # Map camera keys to TF child frames for consistent payloads
        self.tf_child_frame_by_camera = {
            "frontright": "head_right_rgbd_optical",
            "frontleft": "head_left_rgbd_optical",
            "left": "left_rgbd_optical",
            "right": "right_rgbd_optical",
            "back": "rear_rgbd_optical",
        }
        
        # TF subscriptions for camera transforms (static + dynamic)
        self.tf_sub = self.create_subscription(TFMessage, "/tf", self.tf_ros_callback, 10)
        self.tf_static_sub = self.create_subscription(TFMessage, "/tf_static", self.tf_ros_callback, 10)

        # wait for sim control topic to be available
        qos = QoSProfile(depth=10)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # wait for sim control topic to be available
        sim_status_data = None
        def sim_status_callback(msg):
            nonlocal sim_status_data
            sim_status_data = msg.data
        self.sim_status_sub = self.create_subscription(String, self.sim_status_topic, sim_status_callback, 10)
        while True:
            rclpy.spin_once(self, timeout_sec=0.01)
            if sim_status_data is not None:
                print(f"[HabitatROSBridge] Sim status topic available: {sim_status_data}")
                break
            else:
                print("[HabitatROSBridge] Waiting for sim status topic to be available...")
                time.sleep(1.0)
        self.destroy_subscription(self.sim_status_sub)

        # publish sim settings
        self.sim_settings_pub = self.create_publisher(String, self.sim_settings_topic, qos)
        settings_msg = String()
        settings_msg.data = "sensors: head_rgb_left head_rgb_right head_stereo_left head_stereo_right rear_rgb rear_depth, name: spot, model: hab_spot, policy: false, confirm settings"
        self.sim_settings_pub.publish(settings_msg)

        # publish sim control
        self.sim_control_pub = self.create_publisher(String, self.sim_control_topic, qos)
        cam_msg = String()
        cam_msg.data = self.camera_topic
        self.sim_control_pub.publish(cam_msg)

        # publish task submission
        self.task_submission_pub = self.create_publisher(String, self.task_submission_topic, qos)

        # Timer for path follower
        self.path_follower_timer = self.create_timer(0.05, self.path_follower_callback)
        self.follower = WaypointFollower(
            device="cpu",
            lookahead_distance=0.5,
            kp=[2.5, 1.0, 2.0],
            max_vel=[2.0, 1.5, 1.5],
            min_vel=[0.5, 0.5, 0.3],
            arrive_dist=0.1,
            arrive_yaw=np.pi/180.0*30.0,
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
    def sensor_callback(self, rgb_msg, depth_msg, odom_msg):
        self.rgb_callback(rgb_msg, source="frontright")
        self.depth_callback(depth_msg, source="frontright")
        self.odom_callback(odom_msg)

    def rgb_callback(self, msg, source="rgb_image"):
        global _latest_rgb, _latest_rgb_by_source
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb_image = cv_image[..., ::-1].astype(np.uint8)
            _latest_rgb_by_source[source] = rgb_image
            if source == "frontright":
                _latest_rgb = rgb_image
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def depth_callback(self, msg, source="depth_image"):
        global _latest_depth, _latest_depth_by_source
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_image = np.nan_to_num(cv_image, nan=0, posinf=0, neginf=0).astype(np.uint16)
            _latest_depth_by_source[source] = depth_image
            if source == "frontright":
                _latest_depth = depth_image
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")
    
    def tf_ros_callback(self, msg):
        global _latest_tfs
        for t in msg.transforms:
            parent_frame = t.header.frame_id
            child_frame = t.child_frame_id
            
            # Extract Translation (x, y, z)
            translation = {
                'x': float(t.transform.translation.x),
                'y': float(t.transform.translation.y),
                'z': float(t.transform.translation.z)
            }

            # Extract Rotation (x, y, z, w)
            rotation = {
                'x': float(t.transform.rotation.x),
                'y': float(t.transform.rotation.y),
                'z': float(t.transform.rotation.z),
                'w': float(t.transform.rotation.w)
            }
            
            # Store keyed by child_frame (e.g. "head_right_rgbd_optical")
            _latest_tfs[child_frame] = {
                'parent_frame': parent_frame,
                'translation': translation,
                'rotation': rotation
            }

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
            if self.args_cli.dummy_llm:
                keyword_map = {
                    "umbrella": "umbrella",
                    "soda, bread, cheese, and a bowl of fruit": "soda,bread,cheese,a bowl of fruit",
                    "hammer and something to measure length": "hammer,rule",
                    "step on the green then red floor panels": "green floor panel;red floor panel;watch",
                    "phone": "phone",
                    "wallet": "wallet",
                    "toothbrush": "toothbrush",
                    "chair": "chair",
                    "lunchbox": "lunchbox",
                    "one important item for sleeping": "sleeping bag"
                }
                for keyword, object_list in keyword_map.items():
                    if keyword in msg.data.lower():
                        _scenario = object_list
                        break
            else:
                llm_processed_task = self.client.chat.completions.create(
                    model="Qwen/Qwen3-4B-Instruct-2507",
                    messages=[{"role": "user", "content": self.prompt + msg.data}],
                    max_tokens=300,
                    temperature=0.6,
                    top_p=0.95
                )
                print(f"[TaskListener] Processed task: {llm_processed_task.choices[0].message.content}")
                _scenario = llm_processed_task.choices[0].message.content
            print(f"[HabitatROSBridge] Parsed scenario: {_scenario}")

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
                "metrics": {},
            }
            if (not _latest_rgb_by_source or not _latest_depth_by_source or
                    _latest_position is None or _latest_quat_xyzw is None):
                print("[HabitatROSBridge] Waiting for sensor data...")
                return None
            rgb_images = dict(_latest_rgb_by_source)
            depth_images = dict(_latest_depth_by_source)
            if _latest_tfs:
                tfs_by_camera = {}
                for cam_name, child_frame in self.tf_child_frame_by_camera.items():
                    tf_data = _latest_tfs.get(child_frame)
                    if tf_data is not None:
                        tfs_by_camera[cam_name] = tf_data
                if tfs_by_camera:
                    _info["tfs"] = tfs_by_camera
            return format_data(
                rgb_images,
                depth_images,
                _latest_position,
                _latest_quat_xyzw,
                _info,
                "hab_interface",
            )
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
            print("Received waypoint message, length: ", len(message["waypoint"]))
            if _latest_position is None or _latest_quat_xyzw is None:
                return

            # Convert waypoints to include yaw
            #waypoints = set_yaw_to_forward(_latest_position, message["waypoint"])

            # Assign waypoints and reset
            self.follower.waypoints = message["waypoint"]
            self.follower.arrived_at_goal = False
            self.follower.reset()
        elif msg_type == 'STOP':
            print(f"Received stop message: {message}")
            if not "goal_xyxy" in message:
                print(f"Warning: stop message does not contain goal_xyxy")
                return
            goal_xyxy = message["goal_xyxy"]
            x_min, y_min, x_max, y_max = [int(x) for x in goal_xyxy]
            dot_x = (x_min + x_max) // 2
            dot_y = (y_min + y_max) // 2
            self.pub_task_submission(
                dot_x=dot_x,
                dot_y=dot_y,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                description="goal"
            )
            self.pub_task_complete()
        elif msg_type == 'EPISODE':
            # this will trigger a reset of the vln sim
            #TODO: implement episode loading
            pass
            #self.load_episode(message["episode_label"])
        elif msg_type == 'RESET':
            self.pub_robot_reset()

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
        description: str = "goal",
    ):
        # ros2 topic pub /task_submission std_msgs/msg/String "data: '/spot/camera/head_rgb_left/image/compressed, (576, 192, 384, 0, 768, 384), top-right crop'"
        print("[TASK SUBMISSION] Submitting task")
        task_sub_msg = String()
        task_sub_msg.data = f"{self.camera_topic}, ({dot_x}, {dot_y}, {x_min}, {y_min}, {x_max}, {y_max}), {description}"

        self.task_submission_pub.publish(task_sub_msg)

    # Once the agent determines that it has completed the task, call the following function
    # to complete the task. The simulation will then move to the next task, publishing it
    # to /scenario and saving all images and descriptions to a JSON.
    def pub_task_complete(self):
        print("[TASK COMPLETE] Task complete")
        task_complete_msg = String()
        task_complete_msg.data = "task complete"
        self.sim_control_pub.publish(task_complete_msg)
    
    def pub_robot_reset(self):
        print("[ROBOT RESET] Resetting robot")
        robot_reset_msg = String()
        robot_reset_msg.data = "reset"
        self.sim_control_pub.publish(robot_reset_msg)


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