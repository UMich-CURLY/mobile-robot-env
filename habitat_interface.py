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
import time

from scipy.spatial.transform import Rotation as R
from functools import partial


#sys.path.append(os.path.abspath("SG-VLN"))
from utils.socket_server import run_server, format_data
from utils.path_following_utils import WaypointFollower

from tf2_msgs.msg import TFMessage

# Global shared state
_latest_rgb = {}
_latest_depth = {}
_latest_position = None
_latest_quat_xyzw = None
_scenario = None
_info = None
_host = "localhost"
_port = 12357
_tfs = {}
base_command = np.zeros(3)


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

        from openai import OpenAI
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_api_base)

        ## Subscribers
        rgb_topics = {
            "rgb_image": "/habitatsim/image/head_rgb_right/compressed", ## Assigned main camera
            "rgb_head_left": "/habitatsim/image/head_rgb_left/compressed",
            "rgb_right": "/habitatsim/image/right_rgb/compressed",
            "rgb_left": "/habitatsim/image/left_rgb/compressed",
        }

        depth_topics = {
            "depth_image": "/habitatsim/depth/head_stereo_right_depth/image_raw", ## Assigned main camera
            "depth_head_left":  "/habitatsim/depth/head_stereo_left_depth/image_raw",
            "depth_right":      "/habitatsim/depth/right_depth/image_raw",
            "depth_left":       "/habitatsim/depth/left_depth/image_raw",
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
            
        self.tf_names = {}
        for key, value in rgb_topics.items():
            # We split the string at '/image/' and take the part immediately following it.
            # Then we split that part at '/' to isolate the sensor name from anything following it (like 'compressed').
            try:
                sensor_name = value.split('/image/')[1].split('/')[0]
                self.tf_names[key] = sensor_name
            except IndexError:
                self.tf_names[key] = "Pattern not found"
        
        
        
        # # Forward right RGB and depth
        # self.create_subscription(CompressedImage, "/habitatsim/image/head_rgb_right/compressed", self.rgb_callback, 10)
        # self.create_subscription(Image, "/habitatsim/depth/head_stereo_right_depth/image_raw", self.depth_callback, 10)
        # # Forward left RGB and depth
        # self.create_subscription(CompressedImage, "/habitatsim/image/head_rgb_left/compressed", self.rgb_callback, 10)
        # self.create_subscription(Image, "/habitatsim/depth/head_stereo_left_depth/image_raw", self.depth_callback, 10)
        
        # # Right RGB and depth
        # self.create_subscription(CompressedImage, "/habitatsim/image/right_rgb/compressed", self.rgb_callback, 10)
        # self.create_subscription(Image, "/habitatsim/depth/right_depth/image_raw", self.depth_callback, 10)
        
        # # Left RGB and depth
        # self.create_subscription(CompressedImage, "/habitatsim/image/left_rgb/compressed", self.rgb_callback, 10)
        # self.create_subscription(Image, "/habitatsim/depth/left_depth/image_raw", self.depth_callback, 10)
        
        #Odometry and sim ctrl
        self.create_subscription(Odometry, "/habitatsim/platform/odom", self.odom_callback, 10)
        self.subscription = self.create_subscription(String, '/scenario', self.scenario_ros_callback, 10)
        
        self.tf_subscription = self.create_subscription(String, '/tf', self.tf_ros_callback, 10)


        qos = QoSProfile(depth=10)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.sim_control_pub = self.create_publisher(String, '/sim_control', qos)
        cam_msg = String()
        cam_msg.data = "/habitatsim/image/head_rgb_right/compressed"
        self.sim_control_pub.publish(cam_msg)

        # Timer for path follower
        self.path_follower_timer = self.create_timer(0.05, self.path_follower_callback)
        self.follower = None

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
            self.publish("/cmd_vel", move_cmd)

        # Stop after rotation
        self.publish("/cmd_vel", Twist())
        
    # ---- Callbacks ----
    def rgb_callback(self, msg, source):
        global _latest_rgb
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv_image[..., ::-1].astype(np.uint8)

            # Save each camera's RGB image separately
            _latest_rgb[source] = rgb
            #print(_latest_rgb.keys())

        except Exception as e:
            self.get_logger().error(f"RGB callback error from {source}: {e}")
        #print(_latest_rgb)

    def depth_callback(self, msg, source):
        global _latest_depth
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            depth = np.nan_to_num(
                cv_image,
                nan=0,
                posinf=0,
                neginf=0
            ).astype(np.uint16)

            _latest_depth[source] = depth
        except Exception as e:
            self.get_logger().error(f"Depth callback error ({source}): {e}")

    def tf_ros_callback(self, msg):
        global _tfs
        for t in msg.transforms:
            #Filter: We only care if the parent is 'base_link'
            if t.header.frame_id == 'base_link':
                
                child_frame = t.child_frame_id
                
                # Extract Translation (x, y, z)
                translation = {
                    'x': t.transform.translation.x,
                    'y': t.transform.translation.y,
                    'z': t.transform.translation.z
                }

                # Extract Rotation (x, y, z, w)
                rotation = {
                    'x': t.transform.rotation.x,
                    'y': t.transform.rotation.y,
                    'z': t.transform.rotation.z,
                    'w': t.transform.rotation.w
                }
                # If this sensor already exists, this overwrites it with the latest data.
                self.tfs[child_frame] = {
                    'timestamp': t.header.stamp, # Useful to know how fresh the data is
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
            llm_processed_task = self.client.chat.completions.create(
                model="Qwen/Qwen3-4B-Instruct-2507",
                messages=[{"role": "user", "content": self.prompt + msg.data}],
                max_tokens=300,
                temperature=0.6,
                top_p=0.95
            )
            print(f"[TaskListener] Processed task: {llm_processed_task.choices[0].message.content}")
            _scenario = "fireplace"

    def publish(self, topic_name, msg):
        if topic_name not in self._ros_publishers:
            self._ros_publishers[topic_name] = self.create_publisher(type(msg), topic_name, 10)
        self._ros_publishers[topic_name].publish(msg)


    # ---- Server callbacks ----
    def data_callback(self, request_type):
        global _info
        if request_type == "GET_SENSOR_DATA":
            _info = {
                "scene_id": self.args_cli.scene_id,
                "episode_id": self.args_cli.episode_id,
                "instruction": _scenario,
                "robot_height": self.args_cli.robot_height,
                "hfov_deg": self.args_cli.hfov_deg,
                "metrics": {}
            }
            if _latest_rgb is None or _latest_depth is None or _latest_position is None or _latest_quat_xyzw is None:
                return None
            #TODO add in timestamping from ros msg
            return format_data(_latest_rgb, _latest_depth, _latest_position, _latest_quat_xyzw, _info, "hab_interface", timestamp=time.time())
        elif request_type == "GET_EPISODE_LIST":
            return dict(all=["test_sc_test_ep"])
            # episode_set_list = load_episode_set(self.args.episode_folder)
            # episode_set_list["all"] = self.episode_label_list
            # return episode_set_list


    def action_callback(self, msg_type, message):
        global base_command
        if msg_type == 'VEL':
            move_command = Twist()
            move_command.linear.x = float(message['vy'])
            move_command.linear.y = float(message['vx'])
            move_command.angular.z = float(message['vw'])
            self.publish("/cmd_vel", move_command)
        elif msg_type == 'WAYPOINT':
            print("Received waypoint message")
            if _latest_position is None or _latest_quat_xyzw is None:
                return

            # Convert waypoints to include yaw
            #waypoints = set_yaw_to_forward(_latest_position, message["waypoint"])

            if self.follower is None:
                self.follower = WaypointFollower(device="cpu", lookahead_distance=0.5, max_vel=[0.01, 0.01, 0.01],)

            # Assign waypoints and reset
            self.follower.waypoints = message["waypoint"]
            self.follower.arrived_at_goal = False
            self.follower.reset()
        elif msg_type == 'STOP':
            self.action_callback('VEL', {"vx": 0.0, "vy": 0.0, "vw": 0.0})
        elif msg_type == 'EPISODE':
            # this will trigger a reset of the vln sim
            #TODO: implement episode loading
            pass
            #self.load_episode(message["episode_label"])

    def path_follower_callback(self):
        if _latest_position is None or _latest_quat_xyzw is None:
            print("[PATH FOLLOWER] Waiting for position and orientation...")
            return
        
        if self.follower is not None:# and not self.follower.arrived_at_goal:
            print("[PATH FOLLOWER] Updating follower...")
            #print(f"[PATH FOLLOWER] Current waypoints (interface): {self.follower.waypoints}")
            # Pass the current waypoints explicitly
            cmd = self.follower.update(_latest_position, _latest_quat_xyzw, self.follower.waypoints)
            vx, vy, omega = cmd[0].cpu().numpy()

            move_command = Twist()
            move_command.linear.x = float(vx)
            move_command.linear.y = float(vy)
            move_command.angular.z = float(omega)
            self.publish("/cmd_vel", move_command)

            if self.follower.arrived_at_goal:
                print("[PLAN] Goal reached.")



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