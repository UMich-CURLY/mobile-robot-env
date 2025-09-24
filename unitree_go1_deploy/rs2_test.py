#!/usr/bin/env python3
"""
Unified RealSense viewer:

  d435 … aligned color + depth
  t265 … fisheye left/right + pose

Requirements:  pip install pyrealsense2 opencv-python numpy
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
import multiprocessing  # for running multiple cameras concurrently


# ────────────────────────── helpers ──────────────────────────
def load_json_preset(device, json_path: Path) -> None:
    """Upload a depth-camera JSON preset (requires advanced-mode)."""
    if not json_path or not json_path.is_file():
        print(f"[WARN] JSON preset not found: {json_path}")
        return
    adv = rs.rs400_advanced_mode(device)
    if not adv.is_enabled():
        print("[INFO] Enabling advanced mode …")
        adv.toggle_advanced_mode(True)
        time.sleep(2)  # wait for reconnect
        # reacquire handle
        ctx = rs.context()
        sn = device.get_info(rs.camera_info.serial_number)
        device = next(d for d in ctx.devices
                      if d.get_info(rs.camera_info.serial_number) == sn)
        adv = rs.rs400_advanced_mode(device)

    adv.load_json(json_path.read_text())
    print(f"[INFO] Loaded JSON preset from {json_path}")


# ────────────────────────── hardware reset helper ──────────────────────────

def reset_device(serial) -> None:
    """Perform a hardware reset on a RealSense device by serial number."""
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
        if not target:
            print(f"[WARN] Device with serial {serial} not found for reset.")
            return
    else:
        if len(devices) == 1:
            target = devices[0]
        else:
            print("[WARN] Multiple devices present, specify --serial to reset a specific one.")
            return

    sn = target.get_info(rs.camera_info.serial_number)
    print(f"[INFO] Resetting RealSense device {sn} …")
    target.hardware_reset()
    # Allow time for USB re-enumeration
    time.sleep(3)


# ────────────────────────── D435 / depth cam ──────────────────────────
def run_d435(args: argparse.Namespace) -> None:
    # Optional device reset before starting
    if getattr(args, "reset", False):
        reset_device(args.serial)

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)

    cfg.enable_stream(rs.stream.depth, args.width, args.height,
                      rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height,
                      rs.format.bgr8, args.fps)

    print("[INFO] Starting D435/D455 pipeline …")
    profile = pipeline.start(cfg)

    # optional JSON preset
    if args.json:
        load_json_preset(profile.get_device(), Path(args.json))

    align = rs.align(rs.stream.color)

    if args.no_gui:
        print("[INFO] Running D435 headless mode. Press Ctrl+C to quit.")
    else:
        print("[INFO] Press ESC/q to quit.")

    # FPS tracking
    frame_count = 0
    fps_start = time.time()
    d435_fps = 0.0
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_img = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())

            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

            # Combine images side-by-side
            combined = np.hstack((color_img, depth_vis))

            # FPS calculation
            frame_count += 1
            now = time.time()
            if now - fps_start >= 1.0:
                d435_fps = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now

                if args.no_gui:
                    print(f"[D435] FPS: {d435_fps:.1f}")

            if not args.no_gui:
                # Overlay FPS text
                cv2.putText(combined, f"FPS: {d435_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow("Color | Depth", combined)

                # Exit handling
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        pipeline.stop()
        if not args.no_gui:
            cv2.destroyAllWindows()


# ────────────────────────── T265 / tracking cam ──────────────────────────
def run_t265(args: argparse.Namespace) -> None:
    # Optional device reset before starting
    if getattr(args, "reset", False):
        reset_device(args.serial)

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)

    # Always enable pose stream
    cfg.enable_stream(rs.stream.pose)

    # Optionally enable fisheye streams
    if not getattr(args, "pose_only", False):
        cfg.enable_stream(rs.stream.fisheye, 1,
                          args.width, args.height, rs.format.y8, args.fps)
        cfg.enable_stream(rs.stream.fisheye, 2,
                          args.width, args.height, rs.format.y8, args.fps)

    print("[INFO] Starting T265 pipeline …")
    pipeline.start(cfg)

    if args.pose_only:
        print("[INFO] Running pose-only mode. Press Ctrl+C to quit.")
    else:
        print("[INFO] Press ESC/q to quit.")

    # FPS tracking
    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # Images ───────────────────────
            if not args.pose_only:
                fe1 = np.asanyarray(frames.get_fisheye_frame(1).get_data())
                fe2 = np.asanyarray(frames.get_fisheye_frame(2).get_data())
                cv2.imshow("T265 Fisheye Left | Right", np.hstack((fe1, fe2)))

            # Pose ─────────────────────────
            pose = frames.get_pose_frame()
            if pose:
                data = pose.get_pose_data()
                # Orientation as quaternion (x, y, z, w)
                rot = data.rotation
                # FPS calculation
                frame_count += 1
                if time.time() - fps_start >= 1.0:
                    t265_fps = frame_count / (time.time() - fps_start)
                    frame_count = 0
                    fps_start = time.time()
                else:
                    t265_fps = None

                msg = (f"Pos[{data.translation.x:+.2f}, {data.translation.y:+.2f}, {data.translation.z:+.2f}] m  "
                       f"Rot[qx={rot.x:+.2f}, qy={rot.y:+.2f}, qz={rot.z:+.2f}, qw={rot.w:+.2f}]")
                if t265_fps is not None:
                    msg += f"  [T265 FPS: {t265_fps:.1f}]"
                print(msg, end="\r", flush=True)

            # Exit handling
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print()  # newline after carriage-return


# ────────────────────────── Both cameras concurrently ──────────────────────────

def run_both(args: argparse.Namespace) -> None:
    """Run D435/D455 and T265 concurrently in separate processes."""

    # Optional resets first (in main process)
    if getattr(args, "reset", False):
        if args.d435_serial:
            reset_device(args.d435_serial)
        if args.t265_serial:
            reset_device(args.t265_serial)

    # Build argument namespaces for each worker
    d435_args = argparse.Namespace(
        serial=args.d435_serial,
        json=args.json,
        width=args.d435_width,
        height=args.d435_height,
        fps=args.fps,
        no_gui=getattr(args, "d435_no_gui", False),
        reset=args.reset,
    )

    t265_args = argparse.Namespace(
        serial=args.t265_serial,
        width=args.t265_width,
        height=args.t265_height,
        fps=args.fps,
        pose_only=args.pose_only,
        reset=args.reset,
    )

    # Ensure a start-method is set only once
    try:
        multiprocessing.set_start_method("fork")  # Works on Unix/macOS
    except RuntimeError:
        # already set -> ignore
        pass

    p_d435 = multiprocessing.Process(target=run_d435, args=(d435_args,))
    p_t265 = multiprocessing.Process(target=run_t265, args=(t265_args,))

    print("[INFO] Launching D435/D455 and T265 processes … Press Ctrl+C to stop.")
    p_d435.start()
    p_t265.start()

    try:
        # Wait for children
        while p_d435.is_alive() and p_t265.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received. Terminating camera processes …")
        p_d435.terminate()
        p_t265.terminate()
    finally:
        p_d435.join()
        p_t265.join()


# ────────────────────────── CLI ──────────────────────────
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RealSense viewer for depth-cams (D435/D455) "
                    "and tracking-cam (T265).")
    sub = p.add_subparsers(dest="mode", required=True)

    # depth-cam sub-parser
    pd = sub.add_parser("d435", help="Aligned color + depth (D435/D455)")
    pd.add_argument("--serial", help="Camera serial number")
    pd.add_argument("--json", help="Path to depth-camera JSON preset")
    pd.add_argument("--width", type=int, default=640)
    pd.add_argument("--height", type=int, default=480)
    pd.add_argument("--fps", type=int, default=30)
    pd.add_argument("--no-gui", action="store_true",
                    help="Disable OpenCV GUI; print FPS to console instead")
    pd.add_argument("--reset", action="store_true",
                    help="Perform hardware reset before starting")
    pd.set_defaults(func=run_d435)

    # t265 sub-parser
    pt = sub.add_parser("t265", help="Fisheye & pose (T265)")
    pt.add_argument("--serial", help="Camera serial number")
    pt.add_argument("--width", type=int, default=848,
                    help="Fisheye width (default 848)")
    pt.add_argument("--height", type=int, default=800,
                    help="Fisheye height (default 800)")
    pt.add_argument("--fps", type=int, default=30)
    pt.add_argument("--pose-only", action="store_true",
                    help="Output pose only (disable fisheye streams)")
    pt.add_argument("--reset", action="store_true",
                    help="Perform hardware reset before starting")
    pt.set_defaults(func=run_t265)

    # both sub-parser (D435 + T265)
    pb = sub.add_parser("both", help="Run D435/D455 and T265 simultaneously")
    pb.add_argument("--d435-serial", help="D435/D455 serial number")
    pb.add_argument("--t265-serial", help="T265 serial number")
    pb.add_argument("--json", help="Path to depth-camera JSON preset (for D435)")
    pb.add_argument("--d435-width", type=int, default=640)
    pb.add_argument("--d435-height", type=int, default=480)
    pb.add_argument("--t265-width", type=int, default=848)
    pb.add_argument("--t265-height", type=int, default=800)
    pb.add_argument("--fps", type=int, default=30)
    pb.add_argument("--pose-only", action="store_true",
                    help="Pose only for T265 (disable fisheye streams)")
    pb.add_argument("--d435-no-gui", action="store_true",
                    help="Disable OpenCV GUI for D435 (console FPS)")
    pb.add_argument("--reset", action="store_true",
                    help="Perform hardware reset on both devices before starting")
    pb.set_defaults(func=run_both)

    return p


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()