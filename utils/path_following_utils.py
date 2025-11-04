import os
import json
import math
import torch
import numpy as np
from pxr import UsdGeom, Gf, Sdf
from scipy.spatial.transform import Rotation as R

def _infer_scene_id_from_usd(plan_data, scene_path):
    usd_base = os.path.basename(scene_path)
    candidates = [k for k in plan_data.keys() if k in (usd_base, usd_base.replace(".usd", ""))]
    return candidates[0] if candidates else next(iter(plan_data.keys()))


def load_plan(episode_path, scene_folder, episode_id, base_height=0.6):
    with open(episode_path, "r") as f:
        episodes = json.load(f)

    episode = None
    for ep in episodes:
        if ep["episode_id"] == episode_id:
            episode = ep
            break
    if episode is None:
        raise ValueError(f"Episode {episode_id} not found in {episode_path}")

    goal = episode["goals"][0]
    scale = episode.get("scene_scale", 1.0)

    path = [
        [p[0] * scale, p[1] * scale, p[2] * scale]
        for p in goal.get("reference_path", [])
    ]
    target = [goal["location"][0] * scale,
              goal["location"][1] * scale,
              goal["location"][2] * scale]

    return {
        "path": path,
        "target": target,
        "scene_path": f"{scene_folder}/{episode['scene_path']}",
        "scene_id": episode["scene_id"],
        "base_height": base_height,
        "scene_scale": scale,
    } 

def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))

def world_to_body(dx, dy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c*dx - s*dy, s*dx + c*dy


def follow_waypoints(
    cam_pos,
    cam_quat,
    device,
    waypoints_world,
    current_wp_idx,
    max_vx=1.0,
    max_vy=0.3,
    max_vw=1.0,
    kp_x=5.0,
    kp_y=3.0,
    kp_w=3.0,
    arrive_dist=0.3,
    arrive_yaw=np.pi/180.0*5.0,
    term_dist=0.05,
    term_yaw=np.pi/180.0*3.0,
    lookahead_distance=0.8,
):
    # Empty or invalid path
    if waypoints_world is None or len(waypoints_world) == 0:
        return torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32), current_wp_idx

    num_wps = len(waypoints_world)
    # Clamp current index to valid range (do not append/alter original path)
    if current_wp_idx >= num_wps:
        current_wp_idx = num_wps - 1

    base_xy = np.array(cam_pos[:2])
    base_yaw = R.from_quat(cam_quat).as_euler('ZYX')[0]
    waypoints_world = np.array(waypoints_world)

    # 1) Find nearest forward waypoint to avoid going backwards
    #    Only consider waypoints from current_wp_idx forward
    dists = [
        (i, np.linalg.norm(waypoints_world[i, :2] - base_xy))
        for i in range(num_wps)
    ]
    nearest_idx, nearest_dist = min(dists, key=lambda t: t[1])
    current_wp_idx = max(current_wp_idx, nearest_idx)

    # 2) Choose a lookahead target a few meters ahead along the path
    #    Use a simple rule: first waypoint at distance >= lookahead_distance from robot
    target_idx = num_wps - 1
    for i in range(current_wp_idx, num_wps):
        if dists[i][1] >= lookahead_distance:
            target_idx = i
            break
    target_wp = waypoints_world[target_idx]
    dist_to_goal = np.linalg.norm(target_wp[:2] - base_xy)

    # 3) Compute control to the lookahead target in body frame (pure-pursuit style)
    dx = target_wp[0] - base_xy[0]
    dy = target_wp[1] - base_xy[1]
    ang_err = wrap_to_pi(target_wp[2] - base_yaw)

    ex_b, ey_b = world_to_body(dx, dy, base_yaw)

    print("================================================")
    print(f"cam_pos: {cam_pos}")
    print(f"target_wp: [{target_idx}] {target_wp[:2]} {np.rad2deg(target_wp[2])}")
    print(f"base_xy: {base_xy} base_yaw: {np.rad2deg(base_yaw)}")
    print(f"dx: {dx}, dy: {dy}, ang_err: {np.rad2deg(ang_err)}")
    print(f"ex_b: {ex_b}, ey_b: {ey_b}")

    # Linear velocities: proportional to lookahead vector, clipped by given maxima
    vx = float(np.clip(kp_x * ex_b, -max_vx, max_vx))
    vy = float(np.clip(kp_y * ey_b, -max_vy, max_vy))
    vw = float(np.clip(kp_w * ang_err, -max_vw, max_vw))

    is_final_segment = (current_wp_idx >= num_wps - 2)
    if is_final_segment:
        # 4) Near-goal handling to avoid singularity and oscillation
        if dist_to_goal < term_dist and abs(ang_err) < term_yaw:
            vx, vy, vw = 0.0, 0.0, 0.0
            current_wp_idx = num_wps - 1
            print(f"[PLAN] Reached goal")
    else:
        # 5) Waypoint progression: progress when near the current waypoint
        if dist_to_goal < arrive_dist and abs(ang_err) < arrive_yaw:
            current_wp_idx += 1
            print(f"[PLAN] Reached waypoint {current_wp_idx}/{num_wps}")

    command = torch.tensor([[vx, vy, vw]], device=device, dtype=torch.float32)
    return command, current_wp_idx
