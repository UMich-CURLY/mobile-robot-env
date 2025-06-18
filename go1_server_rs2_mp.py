import multiprocessing as mp
import time
import logging
from typing import Optional, Dict, Any

import numpy as np
import threading

from utils.server import run_server, compress_payload
from utils.pcd import get_distance
import utils.planner as pl

from rs2_utils import RealSenseSystem

import lcm
from unitree_go1_deploy.websocket.rc_command_lcmt_relay import rc_command_lcmt_relay
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
D435_SERIAL = "827312072741"  # Update with your serial; None to auto-select
T265_SERIAL = "146322110342"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
# ALIGN_PRESET_PATH: Optional[str] = "/home/curlynuc/ros2_ws/src/realsense-ros/realsense2_camera/launch/HighAccuracyPreset.json"
ALIGN_PRESET_PATH: Optional[str] = "/home/unitree/HighAccuracyPreset.json"

LCM_URL = "udpm://239.255.76.67:7667?ttl=255"
LCM_CHANNEL = "rc_command_relay"
collision_threshold = 0.4  # metres

# ---------------------------------------------------------------------------
# LCM helper
# ---------------------------------------------------------------------------
lc = lcm.LCM(LCM_URL)
_latest_cmd_x = _latest_cmd_y = _latest_cmd_w = 0.0

def publish_lcm(vx: float, vy: float, w: float) -> None:
    global _latest_cmd_x, _latest_cmd_y, _latest_cmd_w
    _latest_cmd_x, _latest_cmd_y, _latest_cmd_w = vx, vy, w
    msg = rc_command_lcmt_relay()
    msg.mode = 0
    msg.left_stick = [vy, vx]  # (Y, X)
    msg.right_stick = [w, 0.0]
    msg.knobs = [0.0, 0.0]
    msg.left_upper_switch = msg.left_lower_left_switch = msg.left_lower_right_switch = 0
    msg.right_upper_switch = msg.right_lower_left_switch = msg.right_lower_right_switch = 0
    lc.publish(LCM_CHANNEL, msg.encode())

# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------

def rgbd_capture_proc(raw_q: mp.Queue, d435_serial: Optional[str], width: int, height: int, fps: int, preset_path: Optional[str]):
    """Capture raw RGB-D frames (no alignment) and push latest to queue."""
    rs = RealSenseSystem(
        d435_serial=d435_serial,
        t265_serial=None,
        width=width,
        height=height,
        fps=fps,
        json_preset=preset_path,
        reset_before_start=False,
    )
    # Disable automatic alignment inside RealSenseSystem for this process
    rs.align = None  # type: ignore

    frame_cnt = 0
    t0 = time.time()
    while True:
        color, depth, frame_ts = rs.get_rgbd()
        if color is None or depth is None:
            continue
        # Maintain only latest frame in queue
        while not raw_q.empty():
            try:
                raw_q.get_nowait()
            except mp.queues.Empty:
                break
        raw_q.put({"color": color, "depth": depth, "ts": frame_ts})
        frame_cnt += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps_val = frame_cnt / (now - t0)
            print(f"[RGBD Capture] FPS: {fps_val:.1f}")
            frame_cnt = 0
            t0 = now


def process_rgbd(raw_q: mp.Queue, processed_q: mp.Queue, dist_val: mp.Value):
    """Process RGB-D frames, update shared distance Value, push latest RGB-D to main queue."""
    # Create a standalone align helper using pyrealsense2
    import pyrealsense2.pyrealsense2 as rs
    align = rs.align(rs.stream.color)

    frame_cnt = 0
    t0 = time.time()

    while True:
        data = raw_q.get()  # Blocking
        color = data["color"]
        depth = data["depth"]

        # NOTE: Proper alignment would need RealSense frames; here we simply pass through.
        aligned_color = color  # already same resolution
        aligned_depth = depth  # for simplicity, assume aligned

        dist = 5.0
        try:
            dist = get_distance(aligned_depth.astype(float) / 1000.0)
        except Exception:
            pass

        # Update shared distance value
        dist_val.value = dist

        # Keep only latest in processed_q
        while not processed_q.empty():
            try:
                processed_q.get_nowait()
            except mp.queues.Empty:
                break
        processed_q.put({"color": aligned_color, "depth": aligned_depth, "distance": dist, "ts": data["ts"]})

        frame_cnt += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps_val = frame_cnt / (now - t0)
            print(f"[RGBD Proc] FPS: {fps_val:.1f}")
            frame_cnt = 0
            t0 = now


def pose_capture_proc(pose_q: mp.Queue, t265_serial: Optional[str]):
    rs = RealSenseSystem(
        d435_serial=None,
        t265_serial=t265_serial,
        reset_before_start=False,
    )
    frame_cnt = 0
    t0 = time.time()

    while True:
        pose, frames_ts = rs.get_pose(timeout_ms=1000)
        if pose is None:
            continue
        # while not pose_q.empty():
        #     try:
        #         pose_q.get_nowait()
        #     except mp.queues.Empty:
        #         break
        pose_q.put({"pose": pose, "ts": frames_ts})
        frame_cnt += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps_val = frame_cnt / (now - t0)
            print(f"[Pose Capture] FPS: {fps_val:.1f}")
            frame_cnt = 0
            t0 = now


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("go1_server_rs2_mp")

    mp.set_start_method("spawn", force=True)

    raw_q: mp.Queue = mp.Queue(maxsize=1)
    processed_q: mp.Queue = mp.Queue(maxsize=1)
    pose_q: mp.Queue = mp.Queue(maxsize=1)

    distance_shared = mp.Value('d', 5.0)
    stop_flag = mp.Value('b', False)

    procs = [
        mp.Process(target=rgbd_capture_proc, args=(raw_q, D435_SERIAL, FRAME_WIDTH, FRAME_HEIGHT, FPS, ALIGN_PRESET_PATH), daemon=True),
        mp.Process(target=process_rgbd, args=(raw_q, processed_q, distance_shared), daemon=True),
        mp.Process(target=pose_capture_proc, args=(pose_q, T265_SERIAL), daemon=True),
    ]

    for p in procs:
        p.start()

    # Planner and state
    planner = pl.Planner(max_vx=0.4, min_vx=-0.3, max_vy=0.2, max_vw=0.4, cruise_vel=0.5, Kp_x=0.5, Kp_w=0.5)
    direct_control_velocity = (0.0, 0.0, 0.0)
    direct_control_ts = time.time()
    use_planner = False

    latest_color: Optional[np.ndarray] = None
    latest_depth: Optional[np.ndarray] = None
    pose_list: list = []  # stores dicts
    pose_ts_list: list = []  # stores timestamps (float)
    latest_pose: Optional[Dict[str, Any]] = None
    latest_distance: float = 5.0
    rgbd_ts: float = 0.0

    # Thread-safe access to shared data
    data_lock = threading.Lock()

    def rgbd_worker():
        nonlocal latest_color, latest_depth, latest_distance, rgbd_ts
        while True:
            aligned_data = processed_q.get()  # blocks until next frame
            with data_lock:
                latest_color = aligned_data["color"]
                latest_depth = aligned_data["depth"]
                latest_distance = aligned_data.get("distance", latest_distance)
                rgbd_ts = aligned_data["ts"]

    def pose_worker():
        nonlocal pose_list, pose_ts_list, latest_pose, rgbd_ts
        import numpy as np  # local import to avoid issues with spawn
        while True:
            pose_data = pose_q.get()
            with data_lock:
                pose_list.append(pose_data["pose"])
                pose_ts_list.append(pose_data["ts"])

                # Keep only the latest 1000 poses
                if len(pose_list) > 1000:
                    pose_list = pose_list[-1000:]
                    pose_ts_list = pose_ts_list[-1000:]

                # Align pose to current RGB-D timestamp if available
                if rgbd_ts != 0 and len(pose_ts_list) > 0:
                    diffs = np.array(pose_ts_list) - rgbd_ts
                    offset = -100.0
                    idx = int(np.argmin(np.abs(diffs-offset)))
                    latest_pose = pose_list[idx]
                    print(f"[Pose Worker] diffs: {diffs[idx]} @ {idx}, offset: {offset}")

    threading.Thread(target=rgbd_worker, daemon=True).start()
    threading.Thread(target=pose_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Socket callbacks
    # ------------------------------------------------------------------
    def data_callback():
        # FPS tracking for client requests
        now = time.time()
        if not hasattr(data_callback, "_start_ts"):
            data_callback._start_ts = now  # type: ignore
            data_callback._cnt = 0  # type: ignore
        data_callback._cnt += 1  # type: ignore
        elapsed = now - data_callback._start_ts  # type: ignore
        if elapsed >= 1.0:
            fps_val = data_callback._cnt / elapsed  # type: ignore
            print(f"[Socket] Request FPS: {fps_val:.1f}")
            data_callback._cnt = 0  # type: ignore
            data_callback._start_ts = now  # type: ignore

        if latest_color is None or latest_depth is None or latest_pose is None:
            return {"success": False, "message": "Data not ready"}
        payload = {
            "rgb_image": latest_color,
            "depth_image": latest_depth,
            "pose": latest_pose,
            "timestamp_server_ns": time.time(),
            "success": True,
            "message": "data from go1 robot (RealSense mp)",
        }
        start_ts = time.time()
        compressed_payload = compress_payload(payload)
        end_ts = time.time()
        print(f"[Compress] Time: {end_ts - start_ts:.3f}s")
        return compressed_payload

    def action_callback(message):
        nonlocal use_planner, direct_control_velocity, direct_control_ts
        print(f"[Action] Message: {message}")
        if message.type == 'VEL':
            direct_control_velocity = (message.x, message.y, message.omega)
            use_planner = False
            direct_control_ts = time.time()
        elif message.type == 'WAYPOINT':
            waypoints = np.vstack((message.x, message.y)).T
            translations = np.hstack((waypoints, np.ones((len(waypoints), 1)) * 0.2))
            planner.update_waypoints(translations[:, :2])
            use_planner = True

    def planner_callback():
        nonlocal use_planner, stop_flag
        col = int(latest_distance < collision_threshold)
        try:
            if not use_planner:
                return {"collision": col}
            ex, ey, _ = planner.get_tracking_error()
            return {
                "err_x": ex,
                "err_y": ey,
                "vx": planner.cmd_x,
                "vy": planner.cmd_y,
                "w": planner.cmd_w,
                "collision": col,
            }
        except Exception as e:
            print("[Planner] Error: ",e)
            return {"collision": col}

    # ---------------- Planner Thread (200 Hz) ----------------
    def planner_loop():
        nonlocal use_planner, latest_pose, latest_distance, direct_control_velocity, direct_control_ts
        import numpy as np
        planner_loop._last_ts = time.time()
        count = 0
        while True:
            try:
                if use_planner and latest_pose is not None:
                    pose = latest_pose["pose"]
                    pos = pose["position"]
                    o = pose["orientation"]
                    yaw = Rotation.from_quat([o["x"], o["y"], o["z"], o["w"]]).as_euler('zyx')[0]

                    vx, vy, vw = planner.step(pos["x"], pos["y"], yaw)
                else:
                    if time.time() - direct_control_ts < 0.5:
                        vx, vy, vw = direct_control_velocity
                    else:
                        vx, vy, vw = 0.0, 0.0, 0.0
                max_acc_x = 0.4
                max_acc_y = 0.2
                vx_limit =  np.sqrt(max(latest_distance-collision_threshold, 0)*2*max_acc_x)
                vy_limit =  np.sqrt(max(latest_distance-collision_threshold, 0)*2*max_acc_y)
                vx_limit = min(vx_limit, planner.max_vx)
                vy_limit = min(vy_limit, planner.max_vy)
                vx = np.clip(vx, planner.min_vx, vx_limit)
                vy = np.clip(vy, -vy_limit, vy_limit)
                if latest_distance < collision_threshold:
                    vw = np.clip(vw, -0.5, 0.5)
                publish_lcm(vx, -vy, vw)
                # if count % 20 == 0:
                    # print(f"[Planner Loop] vx: {vx}, vy: {vy}, vw: {vw}, distance: {latest_distance}")
                sleep_time = max(0.005 - (time.time() - planner_loop._last_ts), 0.0)
                time.sleep(sleep_time) # 200 Hz
                count += 1
                planner_loop._last_ts = time.time()
            except Exception as e:
                print(f"[Planner Loop] Error: {e}")
                import traceback
                traceback.print_exc()
                stop_flag.value = True

    server_thread = threading.Thread(target=run_server, kwargs={"data_cb": data_callback, "action_cb": action_callback, "planner_cb": planner_callback, "stop_flag": stop_flag}, daemon=True)
    server_thread.start()

    threading.Thread(target=planner_loop, daemon=True).start()

    try:
        while True:
            if stop_flag.value:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt – shutting down.")
    finally:
        server_thread.join(timeout=1)
        for p in procs:
            p.terminate()
            p.join()


if __name__ == "__main__":
    main() 