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


def get_frame_ts(frame):
    return frame.get_frame_metadata(rs.frame_metadata_value.frame_timestamp)

def get_sensor_ts(frame):
    return frame.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)

def get_backend_ts(frame):
    return frame.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)

def main() -> None:    
    pipeline = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(rs.stream.depth, 640,480,
                      rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640,480,
                      rs.format.bgr8, 30)

    print("[INFO] Starting D435/D455 pipeline …")
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)


    pipeline2 = rs.pipeline()
    # reset_device("146322110342")
    set_global_time_enabled(True)
    cfg2 = rs.config()
    cfg2.enable_device("146322110342")

    # Always enable pose stream
    cfg2.enable_stream(rs.stream.pose)
    print("[INFO] Starting T265 pipeline …")
    pipeline2.start(cfg2)


    frame_count = 0
    fps_start = time.time()
    d435_fps = 0.0
    try:
        while True:
            t0 = time.time()*1000
            frames_pose = pipeline2.wait_for_frames()

            frames_rgbd = pipeline.wait_for_frames()
            t1 = time.time()*1000
            t2 = time.time()*1000

            ts_rgbd = get_frame_ts(frames_rgbd)
            ts_pose = get_frame_ts(frames_pose)
            # set_global_time_enabled(True)
            # print(f"ts:{ts_rgbd}, sensor:{get_sensor_ts(frames_rgbd)-ts_rgbd}, backend: {get_backend_ts(frames_rgbd)-ts_rgbd*1000}, timestamp: {frames_rgbd.get_timestamp()-ts_rgbd}")
            # # print(frames_rgbd.get_frame_metadata(rs.frame_metadata_value.frame_timestamp))
            # print(f"ts:{ts_pose},  timestampe: {frames_pose.get_timestamp()*1000-ts_pose}")

            # print(ts_rgbd-ts_pose)
            frames_rgbd = align.process(frames_rgbd)
            t3 = time.time()*1000

            #print(f"{t1-t0} {t2-t1} {t3-t2}")
            print(f"pose domain: {frames_pose.frame_timestamp_domain} rgbd domain {frames_rgbd.frame_timestamp_domain}")

            print(f"{frames_pose.get_timestamp()-frames_rgbd.get_timestamp():.3f}")
            print(f"pose: {frames_pose.get_timestamp():.3f} rgbd: {frames_pose.get_timestamp():.3f}")

            depth_frame = frames_rgbd.get_depth_frame()
            color_frame = frames_rgbd.get_color_frame()
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

                print(f"[D435] FPS: {d435_fps:.1f}")

    finally:
        pipeline.stop()



if __name__ == "__main__":

    main()