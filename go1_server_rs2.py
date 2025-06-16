import threading
import time
import logging
from typing import Optional, Dict, Any

import numpy as np

from utils.server import run_server, format_data, compress_payload
from utils.pcd import get_distance
import utils.planner as pl

from rs2_utils import RealSenseSystem

# LCM imports (same as go1_server.py)
import lcm
from unitree_go1_deploy.websocket.rc_command_lcmt_relay import rc_command_lcmt_relay
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Configuration constants – adjust to match your hardware setup
# ---------------------------------------------------------------------------
D435_SERIAL = "827312072741"
T265_SERIAL = "146322110342"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
JSON_PRESET_PATH: Optional[str] = None  # Path to a JSON preset for the depth cam

LCM_URL = "udpm://239.255.76.67:7667?ttl=255"
LCM_CHANNEL = "rc_command_relay"

collision_threshold = 0.4  # metres

# ---------------------------------------------------------------------------
# LCM helper
# ---------------------------------------------------------------------------
lc = lcm.LCM(LCM_URL)
_latest_cmd_x = 0.0
_latest_cmd_y = 0.0
_latest_cmd_w = 0.0

def publish_lcm(vx: float, vy: float, w: float) -> None:
    global _latest_cmd_x, _latest_cmd_y, _latest_cmd_w
    _latest_cmd_x, _latest_cmd_y, _latest_cmd_w = vx, vy, w

    msg = rc_command_lcmt_relay()
    msg.mode = 0
    msg.left_stick = [vy, vx]  # NOTE: order aligns with original script
    msg.right_stick = [w, 0.0]
    msg.knobs = [0.0, 0.0]
    msg.left_upper_switch = 0
    msg.left_lower_left_switch = 0
    msg.left_lower_right_switch = 0
    msg.right_upper_switch = 0
    msg.right_lower_left_switch = 0
    msg.right_lower_right_switch = 0

    lc.publish(LCM_CHANNEL, msg.encode())

# ---------------------------------------------------------------------------
# Sensor data manager (no ROS, uses RealSenseSystem)
# ---------------------------------------------------------------------------
class SensorDataManagerRS2:
    """Thread-safe container for the latest RealSense frames & pose."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.lock = threading.Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_pose: Optional[Dict[str, Any]] = None

        self.data_ready = False
        self.distance = 5.0  # metres (initial value)

        # Planner & collision avoidance
        self.useplanner: bool = False
        self.planner = pl.Planner(max_vx=1, min_vx=-0.2, max_vy=0.1,
                                   max_vw=1, cruise_vel=0.6, Kp_w=2.5)

    # ---------------------------------------------------------------------
    def update(self, rgb: Optional[np.ndarray], depth: Optional[np.ndarray], pose: Optional[Dict[str, Any]]) -> None:
        with self.lock:
            if rgb is not None:
                self.latest_rgb = rgb
            if depth is not None:
                self.latest_depth = depth
                # Depth image is in mm – convert to metres for get_distance
                depth_m = depth.astype(np.float32) / 1000.0
                self.distance = get_distance(depth_m)
            if pose is not None:
                self.latest_pose = pose

        # Check readiness and planner action outside lock to minimise hold time.
        self._check_data_ready()
        self._publish_planner_action()

    # ------------------------------------------------------------------
    def _check_data_ready(self) -> None:
        if self.data_ready:
            return
        with self.lock:
            if self.latest_rgb is not None and self.latest_depth is not None and self.latest_pose is not None:
                self.data_ready = True
                self.logger.info("Initial set of RealSense data received. Server is ready.")

    # ------------------------------------------------------------------
    def _publish_planner_action(self) -> None:
        if not (self.data_ready and self.useplanner):
            return
        pose = self.latest_pose["pose"]
        pos = pose["position"]
        o = pose["orientation"]
        yaw = Rotation.from_quat([o["x"], o["y"], o["z"], o["w"]]).as_euler('zyx')[0]
        vx, vy, vw = self.planner.step(pos["x"], pos["y"], yaw)
        if self.distance < collision_threshold:
            vx = np.clip(vx, -0.5, 0)
        publish_lcm(vx, vy, vw)

    # ------------------------------------------------------------------
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        if not self.data_ready:
            return None
        with self.lock:
            rgb_copy = self.latest_rgb.copy() if self.latest_rgb is not None else None
            depth_copy = self.latest_depth.copy() if self.latest_depth is not None else None
            pose_copy = self.latest_pose.copy() if self.latest_pose is not None else None

        return {
            "rgb_image": rgb_copy,
            "depth_image": depth_copy,
            "pose": pose_copy,
            "timestamp_server_ns": time.time_ns(),
        }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Logger setup
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("go1_server_rs2")

    # RealSense system
    rs_system = RealSenseSystem(
        d435_serial=D435_SERIAL,
        t265_serial=T265_SERIAL,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=FPS,
        json_preset=JSON_PRESET_PATH,
        reset_before_start=False,
    )

    data_manager = SensorDataManagerRS2(logger)

    # ------------------------------------------------------------------
    # Separate acquisition threads for RGB-D and pose
    # ------------------------------------------------------------------

    def rgbd_loop():
        start_ts = time.time()
        frames = 0
        while True:
            try:
                color, depth = rs_system.get_rgbd()
                if color is None or depth is None:
                    continue  # wait for valid frame

                data_manager.update(color, depth, None)

                frames += 1
                now = time.time()
                if now - start_ts >= 1.0:
                    fps = frames / (now - start_ts)
                    print(f"[RGBD] FPS: {fps:.1f}")
                    frames = 0
                    start_ts = now
            except Exception as exc:
                logger.error(f"RGBD thread error: {exc}")
                time.sleep(0.05)

    def pose_loop():
        start_ts = time.time()
        frames = 0
        while True:
            try:
                pose = rs_system.get_pose()
                if pose is None:
                    time.sleep(0.005)
                    continue

                data_manager.update(None, None, pose)

                frames += 1
                now = time.time()
                if now - start_ts >= 1.0:
                    fps = frames / (now - start_ts)
                    print(f"[Pose] FPS: {fps:.1f}")
                    frames = 0
                    start_ts = now
            except Exception as exc:
                logger.error(f"Pose thread error: {exc}")
                time.sleep(0.05)

    threading.Thread(target=rgbd_loop, daemon=True).start()
    threading.Thread(target=pose_loop, daemon=True).start()

    # Callbacks for socket server (re-use utils.server implementation)
    def data_callback():
        now = time.time()
        if not hasattr(data_callback, "_last_ts"):
            data_callback._last_ts = now  # type: ignore
        else:
            dt = now - data_callback._last_ts  # type: ignore
            if dt > 0:
                print(f"[Socket] Request FPS: {1.0 / dt:.1f}")
            data_callback._last_ts = now  # type: ignore

        data = data_manager.get_latest_data()
        if data is None:
            return {"success": False, "message": "Data not ready"}
        data["success"] = True
        data["message"] = "data from go1 robot (RealSense)"
        return compress_payload(data)

    def action_callback(message):
        if message.type == 'VEL':
            if data_manager.distance < collision_threshold:
                message.x = np.clip(message.x, -0.5, 0)
            publish_lcm(message.x, message.y, message.omega)
            logger.info(f"VEL command: {message}")
            data_manager.useplanner = False

        elif message.type == 'WAYPOINT':
            waypoints = np.vstack((message.x, message.y)).T
            data_manager.useplanner = True
            translations = np.hstack((waypoints, np.ones((len(waypoints), 1)) * 0.2))
            logger.info(f"Received waypoints – first: {translations[0] if len(translations) else 'none'}")
            data_manager.planner.update_waypoints(translations[:, :2])

    def planner_callback():
        collision_flag = int(data_manager.distance < collision_threshold)
        planner = data_manager.planner
        try:
            ex, ey, _ = planner.get_tracking_error()
            return {
                "err_x": ex,
                "err_y": ey,
                "vx": planner.cmd_x,
                "vy": planner.cmd_y,
                "w": planner.cmd_w,
                "collision": collision_flag,
            }
        except Exception:
            return {"collision": collision_flag}

    # Start socket server in its own (blocking) thread so we can capture KeyboardInterrupt cleanly
    server_thread = threading.Thread(
        target=run_server,
        kwargs={"data_cb": data_callback, "action_cb": action_callback, "planner_cb": planner_callback},
        daemon=True,
    )
    server_thread.start()

    try:
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received – shutting down.")
    finally:
        rs_system.stop()
        logger.info("Server terminated.")


if __name__ == "__main__":
    main() 