import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import socket
import pickle
import threading
import struct
import base64
import sys
import os

# Get the current working directory
utils_directory = os.getcwd()+'/..'

# Add it to sys.path if it's not already there
if utils_directory not in sys.path:
    sys.path.append(utils_directory)
from utils.socket_server import run_server,format_data,compress_payload
from utils.pcd import get_distance
from scipy.spatial.transform import Rotation
import utils.planner as pl
# --- Configuration ---
SOCKET_HOST = '0.0.0.0'  # Listen on all available interfaces
SOCKET_PORT = 12300
MAX_BUFFER_SIZE = 4096 # For receiving request from client, and sending pickled data length
import lcm

from unitree_go1_deploy.websocket.rc_command_lcmt_relay import rc_command_lcmt_relay 

LCM_URL     = "udpm://239.255.76.67:7667?ttl=255"
LCM_CHANNEL = "rc_command_relay"

collision_threshold = 0.5

lc = lcm.LCM(LCM_URL)
x,y,w = 0,0,0
def publish_lcm(lin_x,lin_y,yaw):
    global x,y,w
    x,y,w = lin_x,lin_y,yaw
    print(f"publishing {lin_x} {lin_y} {yaw}")
    lcm_msg = rc_command_lcmt_relay()
    lcm_msg.mode = 0
    lcm_msg.left_stick  = [lin_y, lin_x]
    lcm_msg.right_stick = [yaw, 0.0]
    lcm_msg.knobs       = [0.0, 0.0]

    lcm_msg.left_upper_switch = \
    lcm_msg.left_lower_left_switch = \
    lcm_msg.left_lower_right_switch = \
    lcm_msg.right_upper_switch = \
    lcm_msg.right_lower_left_switch = \
    lcm_msg.right_lower_right_switch = 0

    lc.publish(LCM_CHANNEL, lcm_msg.encode())

class SensorDataManager:
    """Holds the latest sensor data and provides thread-safe access."""
    def __init__(self, logger):
        self._latest_data = {
            "rgb": None,
            "depth": None,
            "position": None,
            "quat_wxyz": None
        }
        self.info = {
            "scene_id": "real_world",
            "episode_id": time.strftime("%m%d_%H%M"),
            "robot_height": 0.72,
            "hfov_deg": 54.75,
        }
        
        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()
        self.logger = logger
        self.data_ready = False # Flag to indicate if all data has been received at least once
        self.init_pos = None
        self.init_quat = None
        self.init_rotmat = None

        self.useplanner = False
        # self.planner = pl.Planner(max_vx=0.4, min_vx=-0.3, max_vy=0.2, max_vw=0.4, cruise_vel=0.5, Kp_x=0.5, Kp_w=0.5)
        self.planner = pl.Planner(max_vx=0.4, min_vx=-0.3, max_vy=0.1, max_vw=0.5, cruise_vel=0.4, Kp_x=0.5, Kp_w=0.5)
        self.distance = 5
    def rgb_callback(self, msg: CompressedImage):
        try:
            cv_img = self.cv_bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
            with self.lock:
                self._latest_data["rgb"] = cv_img
        except Exception as e:
            self.logger.error(f'RGB callback error: {e}')

    def depth_callback(self, msg: CompressedImage):
        global x,y,w
        try:
            png_bytes = msg.data[12:]
            depth      = cv2.imdecode(
                        np.frombuffer(png_bytes, np.uint8),
                        cv2.IMREAD_UNCHANGED) 
            # print("depth ts: ", msg.header.stamp.sec, msg.header.stamp.nanosec) 

            # depth = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self._latest_data["depth"] = depth
                self.distance = get_distance(self._latest_data["depth"].astype(float)/1000)
                if self.distance<collision_threshold:
                    x = np.clip(x,-0.5,0)
        except Exception as e:
            self.logger.error(f'Depth callback error: {e}')
            import traceback
            traceback.print_exc()

    def pose_callback(self, msg: PoseStamped):
        print("pose callback")
        with self.lock:
            self._latest_data["position"] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            self._latest_data["quat_wxyz"] = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        # self.logger.info('Received Pose', throttle_duration_sec=5)
        self.publish_planner_action()
    
    def publish_planner_action(self):
        global x,y,w
        # if hasattr(self,'_last_publish_time'):
        #     print("last publish time:", time.time() - self._last_publish_time)
        self._last_publish_time = time.time()

        if self.useplanner:
            position = self._latest_data["position"]
            o = self._latest_data["quat_wxyz"]
            yaw = Rotation.from_quat([o[1],o[2],o[3],o[0]]).as_euler('zyx')[0]
            x,y,w = self.planner.step(position[0],position[1],yaw)
            if self.distance<collision_threshold:
                x = np.clip(x,-0.5,0)
            publish_lcm(x, -y, w)
        
            
    def get_latest_data(self):
        with self.lock:
            data_ready = True
            for key in self._latest_data:
                if self._latest_data[key] is None:
                    print(f"data not ready for key: {key}")
                    data_ready = False
                    break
            self.data_ready = data_ready
            if not self.data_ready:
                return {}
            # Return copies to avoid issues if data is updated while pickling
            return self._latest_data

    def _compressed_depth_to_image(self, msg, frame_id='camera_depth_frame'):
        """
        decode `compressedDepth` → `sensor_msgs/Image` (16UC1).
        """
        raw_bytes = base64.b64decode(msg.data)

        png_start = raw_bytes.find(b'\x89PNG')
        # if png_start == -1:
        #     self.get_logger().warn('PNG header not found in compressedDepth')
        #     return None

    def _to_compressed_image(self, msg, frame_id='camera_frame'):
        image_msg = CompressedImage()
        image_msg.header.stamp = msg.header.stamp
        image_msg.header.frame_id = frame_id
        image_msg.format = msg.format
        image_msg.data = base64.b64decode(msg.data)
        return image_msg

class SensorServerNode(Node):
    def __init__(self, data_manager: SensorDataManager):
        super().__init__('sensor_socket_server_node')
        self.data_manager = data_manager
        self.get_logger().info("Sensor Socket Server ROS Node Started")

        # --- Subscribers to sensor topics ---
        self.rgb_subscriber = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed', self.data_manager.rgb_callback, 10)
        self.depth_subscriber = self.create_subscription(
            CompressedImage, '/camera/aligned_depth_to_color/image_raw/compressedDepth', self.data_manager.depth_callback, 10)
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/orb_slam3/camera_pose',
            self.data_manager.pose_callback,
            qos_profile_sensor_data) 

        # --- Dummy Publishers (to simulate sensor data for this example) ---
        # self.dummy_rgb_pub = self.create_publisher(CompressedImage, 'sensor/rgb_image', 10)
        # self.dummy_depth_pub = self.create_publisher(CompressedImage, 'sensor/depth_image', 10)
        # self.dummy_pose_pub = self.create_publisher(Odometry, 'sensor/pose', 10)
        # self.timer = self.create_timer(0.1, self.publish_dummy_data) # Publish dummy data at 10Hz
        self.frame_count = 0
        self.get_logger().info("Dummy sensor publishers started.")

    def publish_dummy_data(self):
        now = self.get_clock().now().to_msg()

        rgb_arr = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(rgb_arr, f"RGB ROS Frame: {self.frame_count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        rgb_msg = self.data_manager.cv_bridge.cv2_to_imgmsg(rgb_arr, encoding="bgr8")
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = "camera_rgb_optical_frame"
        self.dummy_rgb_pub.publish(rgb_msg)

        depth_arr = np.full((480, 640), 1500, dtype=np.uint16) # 1.5 meters in mm
        # Add some pattern to depth
        cv2.circle(depth_arr, (320, 240), 50, int(1000 + (self.frame_count % 10) * 50) , -1) # Varying depth circle
        depth_msg = self.data_manager.cv_bridge.cv2_to_imgmsg(depth_arr, encoding="16UC1")
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = "camera_depth_optical_frame"
        self.dummy_depth_pub.publish(depth_msg)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.position = Point(x=1.0 + self.frame_count * 0.01, y=2.0, z=0.0)
        pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.dummy_pose_pub.publish(pose_msg)

        self.frame_count += 1

def main(args=None):
    rclpy.init(args=args)
    
    # Using a temporary node to get a logger for SensorDataManager
    temp_node_for_logger = rclpy.create_node('temp_logger_node_sensor_server')
    data_manager = SensorDataManager(temp_node_for_logger.get_logger())
    temp_node_for_logger.destroy_node() # Don't need it anymore

    sensor_server_ros_node = SensorServerNode(data_manager)

    def data_callback():
        sensor_data = data_manager.get_latest_data()
        print("data callback")
        if sensor_data == {}:
            return {
                "success": False,
                "message": "data not ready"
            }
        return format_data(
            sensor_data["rgb"],
            sensor_data["depth"],
            sensor_data["position"],
            sensor_data["quat_wxyz"],
            data_manager.info,
            server_name="go1_server"
        )
    
    def action_callback(msg_type, message):
        global x,y,w
        if msg_type == 'VEL':
            x,y,w = message['vx'],message['vy'],message['vw']
            if data_manager.distance<collision_threshold:
                x = np.clip(x,-0.5,0)
            data_manager.useplanner = False
            # print(f"position: {position} quat: {quat}")
        if msg_type == 'WAYPOINT':
            waypoints = np.vstack((message["x_list"], message["y_list"])).T
            data_manager.useplanner = True
            # for visualizer in visualizers:
            #     visualizer.set_visibility(False)
            # visualizers.clear()
            translations = np.hstack((waypoints,np.ones((len(waypoints),1))*0.2))
            print("first waypoint raw: %s" % str(translations[0]))
            data_manager.planner.update_waypoints(translations[:,:2])
        if msg_type == 'STOP':
            x,y,w = 0,0,0
            print("STOP")
            data_manager.useplanner = False
    def planner_callback():
        col = int(data_manager.distance<collision_threshold)
        try:
            planner = data_manager.planner
            ex,ey,_ = planner.get_tracking_error()
            return {'err_x':ex,'err_y':ey,"vx":planner.cmd_x,"vy":planner.cmd_y,"w":planner.cmd_w,"collision":col}
        except:
            return {"collision":col}
        
    def lcm_sender():
        global x,y,w
        while True:
            if not data_manager.useplanner:
                publish_lcm(x,-y,w)
                time.sleep(0.05)
    
    # Start socket server in a separate thread
    # Pass the logger from the ROS node to the socket thread for consistent logging

    server_thread = threading.Thread(target=run_server,kwargs={"data_cb":data_callback,"action_cb":action_callback,"planner_cb":planner_callback,"port":SOCKET_PORT,"host":SOCKET_HOST})
    server_thread.start()
    publishing_thread = threading.Thread(target=lcm_sender)
    publishing_thread.start()
    

    try:
        rclpy.spin(sensor_server_ros_node)
    except KeyboardInterrupt:
        sensor_server_ros_node.get_logger().info("Keyboard interrupt received, shutting down ROS node.")
    except Exception as e:
        sensor_server_ros_node.get_logger().error(f"Exception in rclpy.spin: {e}", exc_info=True)
    finally:
        sensor_server_ros_node.get_logger().info("Shutting down sensor server ROS node...")
        sensor_server_ros_node.destroy_node()
        rclpy.shutdown()
        # The socket thread is a daemon, so it will exit when the main thread (rclpy.spin) exits.
        # Or, if rclpy.ok() is used in its loop, it will exit when rclpy is shutdown.
        # We can also add a more explicit shutdown mechanism if needed (e.g., using an event).
        if server_thread.is_alive():
            # Give it a moment to close gracefully based on rclpy.ok()
            server_thread.join(timeout=2.0) 
            if server_thread.is_alive():
                print("Socket thread did not shut down gracefully via rclpy.ok(). It will be terminated as daemon.")
        print("Sensor server fully shut down.")


if __name__ == '__main__':
    main()
