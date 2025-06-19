import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
try:
    # Import pyrealsense2 with the full module path to avoid name clashes
    import pyrealsense2.pyrealsense2 as rs  # type: ignore
except ImportError as exc:
    raise ImportError(
        "pyrealsense2 is required for rs2_utils but was not found. "
        "Install with `pip install pyrealsense2` and ensure a RealSense SDK "
        "is available on the system.") from exc


# -----------------------------------------------------------------------------
# Helper functions (largely adopted from rs2_test.py)
# -----------------------------------------------------------------------------
def set_global_time_enabled(enable: bool):
    """
    Sets the RS2_OPTION_GLOBAL_TIME_ENABLED on compatible sensors.

    Args:
        enable (bool): True to enable global time, False to disable.
    """
    ctx = rs.context()
    devices = ctx.query_devices()

    if not devices:
        print("No RealSense device connected.")
        return

    for dev in devices:
        print(f"Device: {dev.get_info(rs.camera_info.name)} (Serial: {dev.get_info(rs.camera_info.serial_number)})")
        found_sensor_for_option = False
        for sensor in dev.query_sensors():
            sensor_name = sensor.get_info(rs.camera_info.name)
            if sensor.supports(rs.option.global_time_enabled):
                found_sensor_for_option = True
                print(f"  Sensor '{sensor_name}' supports global_time_enabled.")
                try:
                    current_value = sensor.get_option(rs.option.global_time_enabled)
                    print(f"    Current global_time_enabled: {bool(current_value)}")

                    desired_value = 1.0 if enable else 0.0
                    if current_value != desired_value:
                        sensor.set_option(rs.option.global_time_enabled, desired_value)
                        new_value = sensor.get_option(rs.option.global_time_enabled)
                        print(f"    Set global_time_enabled to: {bool(new_value)}")
                        if bool(new_value) != enable:
                            print(f"    WARNING: Failed to set global_time_enabled to {enable} on {sensor_name}.")
                    else:
                        print(f"    global_time_enabled is already set to: {bool(current_value)}")

                except Exception as e:
                    print(f"    Error setting global_time_enabled on '{sensor_name}': {e}")
            else:
                print(f"  Sensor '{sensor_name}' does NOT support global_time_enabled.")
        
        if not found_sensor_for_option:
            print(f"  No sensor on this device supports global_time_enabled.")
        print("-" * 30)

def _load_json_preset(device: rs.device, json_path: Path) -> None:
    """Upload a depth-camera JSON preset (requires advanced-mode)."""
    if not json_path or not json_path.is_file():
        print(f"[WARN] JSON preset not found: {json_path}")
        return

    adv = rs.rs400_advanced_mode(device)
    if not adv.is_enabled():
        print("[INFO] Enabling advanced mode …")
        adv.toggle_advanced_mode(True)
        time.sleep(2)  # Wait for reconnect
        # Re-acquire handle
        ctx = rs.context()
        sn = device.get_info(rs.camera_info.serial_number)
        device = next(d for d in ctx.devices
                      if d.get_info(rs.camera_info.serial_number) == sn)
        adv = rs.rs400_advanced_mode(device)

    adv.load_json(json_path.read_text())
    print(f"[INFO] Loaded JSON preset from {json_path}")


def _reset_device(serial: Optional[str]) -> None:
    """Perform a hardware reset on the RealSense device with the given serial."""
    ctx = rs.context()
    devices = ctx.devices
    if not devices:
        print("[WARN] No RealSense devices found for reset.")
        return

    target = None
    if serial:
        for d in devices:
            if d.get_info(rs.camera_info.serial_number) == serial:
                target = d
                break
        if target is None:
            print(f"[WARN] Device with serial {serial} not found for reset.")
            return
    else:
        if len(devices) == 1:
            target = devices[0]
        else:
            print("[WARN] Multiple devices present, specify serial to reset a specific one.")
            return

    sn = target.get_info(rs.camera_info.serial_number)
    print(f"[INFO] Resetting RealSense device {sn} …")
    target.hardware_reset()
    time.sleep(3)  # Allow USB re-enumeration


# -----------------------------------------------------------------------------
# Core RealSense System wrapper
# -----------------------------------------------------------------------------
class RealSenseSystem:
    """Convenience wrapper that starts D435/D455 (color + depth) and optional
    T265 (pose) pipelines and exposes a unified `grab_frames` interface.

    Attributes
    ----------
    has_pose : bool
        Indicates whether pose data is expected/available (T265 present).
    """

    def __init__(
        self,
        d435_serial: Optional[str] = None,
        t265_serial: Optional[str] = None,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        json_preset: Optional[str] = None,
        reset_before_start: bool = False,
    ) -> None:
        # Optional resets first (prevents "resource busy" errors)
        if reset_before_start:
            if d435_serial:
                _reset_device(d435_serial)
            if t265_serial:
                _reset_device(t265_serial)
            set_global_time_enabled(True)

        # ---------------- D435 / depth cam ----------------
        self.d435_pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        if d435_serial is not None:
            self.d435_pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(d435_serial)
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            profile = self.d435_pipeline.start(cfg)
            depth_sensor = profile.get_device().first_depth_sensor()
            scale = depth_sensor.get_depth_scale()
            print("[INFO] depth scale: ", scale)
            # Align depth to color for convenience
            self.align = rs.align(rs.stream.color)
            if json_preset:
                print(f"[INFO] Loading JSON preset from {json_preset}")
                _load_json_preset(profile.get_device(), Path(json_preset))
            print("[INFO] D435/D455 pipeline started.")

        # ---------------- T265 / pose cam -----------------
        self.t265_pipeline: Optional[rs.pipeline] = None
        if t265_serial is not None:
            self.t265_pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(t265_serial)
            cfg.enable_stream(rs.stream.pose)
            self.t265_pipeline.start(cfg)
            print("[INFO] T265 pipeline started (pose).")

        self.has_pose: bool = self.t265_pipeline is not None
        # Caches for latest frames (thread-safe access not handled here)
        self._last_color: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_pose: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------------
    def _fetch_d435_frames(self, timeout_ms: int = 500) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.d435_pipeline is None:
            return None, None
        frames = self.d435_pipeline.wait_for_frames(timeout_ms)
        if self.align is not None:
            frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_img = np.asanyarray(depth_frame.get_data())  # uint16 depth in mm
        color_img = np.asanyarray(color_frame.get_data())  # BGR
        return color_img, depth_img

    def _fetch_t265_pose(self) -> Optional[Dict[str, Any]]:
        if self.t265_pipeline is None:
            return None
        frames = self.t265_pipeline.poll_for_frames()
        if not frames:
            return None
        pose_frame = frames.get_pose_frame()
        if not pose_frame:
            return None
        data = pose_frame.get_pose_data()
        pose_dict = {
            "header": {
                "stamp_sec": int(time.time()),
                "stamp_nanosec": int((time.time_ns()) % 1_000_000_000),
                "frame_id": "t265_odom",
            },
            "pose": {
                "position": {
                    "x": data.translation.x,
                    "y": data.translation.y,
                    "z": data.translation.z,
                },
                "orientation": {
                    "x": data.rotation.x,
                    "y": data.rotation.y,
                    "z": data.rotation.z,
                    "w": data.rotation.w,
                },
            },
        }
        return pose_dict

    # ---------------------------------------------------------------------
    def grab_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Blocking fetch of the latest frames (color, depth, pose).

        Returns
        -------
        color_img : np.ndarray | None
            BGR image or `None` if unavailable.
        depth_img : np.ndarray | None
            uint16 depth image (millimetres) or `None`.
        pose_dict : dict | None
            Pose dictionary (same format as ROS2 message in go1_server) or `None`.
        """
        color, depth = self._fetch_d435_frames()
        pose = self._fetch_t265_pose() if self.has_pose else None
        # Update caches (could be None)
        if color is not None:
            self._last_color = color
        if depth is not None:
            self._last_depth = depth
        if pose is not None:
            self._last_pose = pose
        # Fallback to last known if current fetch failed (non-blocking semantics)
        return (
            self._last_color,
            self._last_depth,
            self._last_pose,
        )

    # ---------------------------------------------------------------------
    def stop(self) -> None:
        if self.d435_pipeline is not None:
            self.d435_pipeline.stop()
        if self.t265_pipeline is not None:
            self.t265_pipeline.stop()
        print("[INFO] RealSense pipelines stopped.")

    # ---------------------------------------------------------------------
    def __del__(self):
        # Ensure resources are released when the object is garbage-collected.
        try:
            self.stop()
        except Exception:
            # Suppress any exceptions during interpreter shutdown.
            pass

    # ---------------------------------------------------------------------
    def poll_once(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Non-blocking update of cached frames.

        This method queries the pipelines with `poll_for_frames()` which returns
        immediately (or `None`) if no new frames are available. When new frames
        do arrive, the internal caches (`_last_color`, `_last_depth`,
        `_last_pose`) are updated. The method always returns the *current*
        cached values – thus callers can call it at high frequency to keep the
        cache fresh while obtaining the latest available data at any time.
        """

        # Initialise FPS tracking attrs on first call
        if not hasattr(self, "_fps_last_ts"):
            self._fps_last_ts = time.time()
            self._fps_frames = 0

        new_rgb = False  # flag for FPS counting

        # ---------------- D435 ----------------
        if self.d435_pipeline is not None:
            frames = self.d435_pipeline.poll_for_frames()
            if frames:
                if self.align is not None:
                    frames = self.align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if depth_frame and color_frame:
                    self._last_depth = np.asanyarray(depth_frame.get_data())
                    self._last_color = np.asanyarray(color_frame.get_data())
                    new_rgb = True

        # ---------------- T265 ----------------
        if self.t265_pipeline is not None:
            frames = self.t265_pipeline.poll_for_frames()
            if frames:
                pose_frame = frames.get_pose_frame()
                if pose_frame:
                    data = pose_frame.get_pose_data()
                    self._last_pose = {
                        "header": {
                            "stamp_sec": int(time.time()),
                            "stamp_nanosec": int((time.time_ns()) % 1_000_000_000),
                            "frame_id": "t265_odom",
                        },
                        "pose": {
                            "position": {
                                "x": data.translation.x,
                                "y": data.translation.y,
                                "z": data.translation.z,
                            },
                            "orientation": {
                                "x": data.rotation.x,
                                "y": data.rotation.y,
                                "z": data.rotation.z,
                                "w": data.rotation.w,
                            },
                        },
                    }

        # ---------------- FPS logging ----------------
        if new_rgb:
            self._fps_frames += 1
            now = time.time()
            duration = now - self._fps_last_ts
            if duration >= 1.0:
                fps_val = self._fps_frames / duration
                print(f"[RealSense] Capture FPS: {fps_val:.1f}")
                self._fps_frames = 0
                self._fps_last_ts = now

        return self._last_color, self._last_depth, self._last_pose

    # ---------------------------------------------------------------------
    def get_rgbd(self, timeout_ms: int = 5000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Blocking wait for the next aligned RGB-D frame pair.

        This is intended for a dedicated thread that only interacts with the
        D435/D455 pipeline. Internally uses `wait_for_frames()` so the call
        blocks until a new set of frames is available (or until `timeout_ms`).
        """
        if self.d435_pipeline is None:
            return None, None
        while True:
            try:
                frames = self.d435_pipeline.wait_for_frames(timeout_ms)
                if frames:
                    # frames_ts = time.time()
                    break 
            except Exception as e:
                print(f"[RealSense] D435 Error: {e}")
                time.sleep(0.01)
                continue

        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        raw_frames_ts = depth_frame.get_timestamp()
        if not hasattr(self, "_init_depth_ts"):
            self._init_depth_ts = raw_frames_ts
        frames_ts = raw_frames_ts - self._init_depth_ts + time.time()
        # frames_ts = depth_frame.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)*1000 
        #backend timestamp is same as system? except need *1000

        # print("--------------------------------")
        # print("color_frame.get_timestamp()", color_frame.get_timestamp())
        # print("color_frame.get_frame_timestamp_domain()", color_frame.get_frame_timestamp_domain())
        # print("depth_frame.get_timestamp()", depth_frame.get_timestamp())
        # print("time_of_arrival", depth_frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival))

        # print("depth_frame.get_frame_timestamp_domain()", depth_frame.get_frame_timestamp_domain())
        # print("--------------------------------")
        if not depth_frame or not color_frame:
            return None, None

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        # Update caches
        self._last_color = color_img
        self._last_depth = depth_img

        return color_img, depth_img, frames_ts

    # ---------------------------------------------------------------------
    def get_pose(self, timeout_ms: int = 0) -> Optional[Dict[str, Any]]:
        """Fetch the latest pose frame from T265.

        If `timeout_ms` is 0 (default) the method uses `poll_for_frames()` and
        returns immediately if no new pose is available. For a blocking wait
        specify a positive timeout—the function will call `wait_for_frames()`.
        """
        if self.t265_pipeline is None:
            return None

        if timeout_ms == 0:
            frames = self.t265_pipeline.poll_for_frames()
            if not frames:
                return None
        else:
            frames = self.t265_pipeline.wait_for_frames(timeout_ms)
        # frames_ts = time.time()

        pose_frame = frames.get_pose_frame()
        raw_frames_ts = pose_frame.get_timestamp()
        #backend_timestamp,frame_timestamp,sensor_timestamp,time_of_arrival
        # frames_ts = pose_frame.get_frame_metadata(rs.frame_metadata_value.frame_timestamp)
        # print(f"pose_ts {frames_ts}")

        if not hasattr(self, "_init_pose_ts"):
            self._init_pose_ts = raw_frames_ts
        frames_ts = raw_frames_ts - self._init_pose_ts + time.time()
        # print(f"[Pose] time diff: {frames_ts - self._init_ts_pose}, time.time(): {time.time() - self._init_ts}")
        # print("pose_frame.get_timestamp()", pose_frame.get_timestamp(), flush=True)
        if not pose_frame:
            return None

        data = pose_frame.get_pose_data()
        # change (x: right, z: backward, y: up)
        # to x: forward, y: left, z: up
        pose_dict = {
            "header": {
                "stamp_sec": int(time.time()),
                "stamp_nanosec": int((time.time_ns()) % 1_000_000_000),
                "frame_id": "t265_odom",
            },
            "pose": {
                "position": {
                    "x": -data.translation.z,
                    "y": -data.translation.x,
                    "z": data.translation.y,
                },
                "orientation": {
                    "x": -data.rotation.z,
                    "y": -data.rotation.x,
                    "z": data.rotation.y,
                    "w": data.rotation.w,
                },
            },
        }

        self._last_pose = pose_dict
        return pose_dict, frames_ts