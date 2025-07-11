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
from utils.server import run_server,format_data,compress_payload
from utils.pcd import get_distance
from scipy.spatial.transform import Rotation
import utils.planner as pl
# --- Configuration ---
SOCKET_HOST = '0.0.0.0'  # Listen on all available interfaces
SOCKET_PORT = 12345
MAX_BUFFER_SIZE = 4096 # For receiving request from client, and sending pickled data length
import lcm

from unitree_go1_deploy.websocket.rc_command_lcmt_relay import rc_command_lcmt_relay 

LCM_URL     = "udpm://239.255.76.67:7667?ttl=255"
LCM_CHANNEL = "rc_command_relay"

collision_threshold = 0.4

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
        self.latest_rgb_cv_image = None
        self.latest_depth_cv_image = None
        self.latest_pose_dict = None
        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()
        self.logger = logger
        self.data_ready = False # Flag to indicate if all data has been received at least once
        self.init_pos = None
        self.init_quat = None
        self.init_rotmat = None

        self.useplanner = None
        self.planner = pl.Planner(max_vx=1,min_vx=-0.2,max_vy = 0.1,max_vw=1,cruise_vel=0.6,Kp_w=2.5)

        self.distance = 5
    def rgb_callback(self, msg: CompressedImage):
        try:
            cv_img = self.cv_bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_rgb_cv_image = cv_img
            self.check_data_ready()
        except Exception as e:
            self.logger.error(f'RGB callback error: {e}')

    def depth_callback(self, msg: CompressedImage):
        try:
            png_bytes = msg.data[12:]
            depth      = cv2.imdecode(
                        np.frombuffer(png_bytes, np.uint8),
                        cv2.IMREAD_UNCHANGED) 
            # print("depth ts: ", msg.header.stamp.sec, msg.header.stamp.nanosec) 

            # depth = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self.latest_depth_cv_image = depth
                self.distance = get_distance(self.latest_depth_cv_image.astype(float)/1000)
                if(self.distance<collision_threshold):
                    x = np.clip(x,-0.5,0)
            self.check_data_ready()
        except Exception as e:
            self.logger.error(f'Depth callback error: {e}')


    # def pose_callback(self, msg: Odometry):
    #     with self.lock:
    #         print("pose ts: ", msg.header.stamp.sec, msg.header.stamp.nanosec) 
    #         self.latest_pose_dict = {
    #             "header": {
    #                 "stamp_sec": msg.header.stamp.sec,
    #                 "stamp_nanosec": msg.header.stamp.nanosec,
    #                 "frame_id": msg.header.frame_id
    #             },
    #             "pose": {
    #                 "position": {"x": msg.pose.pose.position.x, "y": msg.pose.pose.position.y, "z": msg.pose.pose.position.z},
    #                 "orientation": {"x": msg.pose.pose.orientation.x, "y": msg.pose.pose.orientation.y,
    #                                 "z": msg.pose.pose.orientation.z, "w": msg.pose.pose.orientation.w}
    #             }
    #         }
    #     # self.logger.info('Received Pose', throttle_duration_sec=5)
    #     self.check_data_ready()
    #     self.publish_planner_action()

    def pose_callback(self, msg: PoseStamped):
        with self.lock:
            # print("pose ts: ", msg.header.stamp.sec, msg.header.stamp.nanosec) 
            self.latest_pose_dict = {
                "header": {
                    "stamp_sec": msg.header.stamp.sec,
                    "stamp_nanosec": msg.header.stamp.nanosec,
                    "frame_id": msg.header.frame_id
                },
                "pose": {
                    # The path is now simpler: msg.pose instead of msg.pose.pose
                    "position": {"x": msg.pose.position.x, "y": msg.pose.position.y, "z": msg.pose.position.z},
                    "orientation": {"x": msg.pose.orientation.x, "y": msg.pose.orientation.y,
                                    "z": msg.pose.orientation.z, "w": msg.pose.orientation.w}
                }
            }
        # self.logger.info('Received Pose', throttle_duration_sec=5)
        self.check_data_ready()
        self.publish_planner_action()

    def check_data_ready(self):
        # print(self.distance)

        if not self.data_ready:
            with self.lock:
                if self.latest_rgb_cv_image is not None and \
                   self.latest_depth_cv_image is not None and \
                   self.latest_pose_dict is not None:
                    self.data_ready = True
                    self.logger.info("Initial set of sensor data received. Server is ready for client requests.")
    
    def publish_planner_action(self):
        global x,y,w
        # if hasattr(self,'_last_publish_time'):
        #     print("last publish time:", time.time() - self._last_publish_time)
        self._last_publish_time = time.time()

        if self.data_ready and self.useplanner:
            pose = self.latest_pose_dict['pose']
            position = pose['position']
            o = pose['orientation']
            yaw = Rotation.from_quat([o['x'],o['y'],o['z'],o['w']]).as_euler('zyx')[0]
            x,y,w = self.planner.step(position['x'],position['y'],yaw)
            if(self.distance<collision_threshold):
                vx = np.clip(vx,-0.5,0)
        publish_lcm(x, y, w)
        
            
    def get_latest_data(self):
        with self.lock:
            if not self.data_ready:
                return None
            # Return copies to avoid issues if data is updated while pickling
            return {
                "rgb_image": self.latest_rgb_cv_image.copy() if self.latest_rgb_cv_image is not None else None,
                "depth_image": self.latest_depth_cv_image.copy() if self.latest_depth_cv_image is not None else None,
                "pose": self.latest_pose_dict.copy() if self.latest_pose_dict is not None else None,
                "timestamp_server_ns": time.time()*1e9
            }

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
        sensor_data['success'] = True
        sensor_data['message'] = "data from go1 robot"
        return compress_payload(sensor_data)
    
    def action_callback(message):
        global x,y,w
        if message.type == 'VEL':
            if(data_manager.distance<collision_threshold):
                message.x = np.clip(message.x,-0.5,0)

            x,y,w = message.x,message.y,message.omega
            print(message)
            data_manager.useplanner = False
            # print(f"position: {position} quat: {quat}")
        if message.type == 'WAYPOINT':
            waypoints = np.vstack((message.x,message.y)).T
            data_manager.useplanner = True
            # for visualizer in visualizers:
            #     visualizer.set_visibility(False)
            # visualizers.clear()
            translations = np.hstack((waypoints,np.ones((len(waypoints),1))*0.2))
            print("first waypoint raw: %s" % str(translations[0]))
            data_manager.planner.update_waypoints(translations[:,:2])

    def planner_callback():
        col = int(data_manager.distance<collision_threshold)
        try:
            planner = data_manager.planner
            ex,ey,_ = planner.get_tracking_error()
            return {'err_x':ex,'err_y':ey,"vx":planner.cmd_x,"vy":planner.cmd_y,"w":planner.cmd_w,"collision":col}
        except:
            return {"collision":col}
    
    # Start socket server in a separate thread
    # Pass the logger from the ROS node to the socket thread for consistent logging

    server_thread = threading.Thread(target=run_server,kwargs={"data_cb":data_callback,"action_cb":action_callback,"planner_cb":planner_callback})
    server_thread.start()

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
