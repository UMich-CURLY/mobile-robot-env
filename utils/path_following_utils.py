import os
import json
import math
import torch
import numpy as np
from pxr import UsdGeom, Gf, Sdf


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

def get_base_xy_yaw(manager_env):
    pos = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy()
    quat = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy()
    w, x, y, z = quat
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return (float(pos[0]), float(pos[1])), yaw

def world_to_body(dx, dy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c*dx - s*dy, s*dx + c*dy


def follow_waypoints(
    manager_env,
    device,
    waypoints_world,
    current_wp_idx,
    max_vx=1.0,
    max_vy=0.3,
    max_yaw_rate=1.0,
    k_p_ang=2.0,
    arrive_dist=0.3,
    arrive_yaw=np.pi/180.0*60.0,
    term_dist=0.05,
    term_yaw=np.pi/180.0*30.0,
    lookahead_distance=0.8,
    min_lookahead=0.3
):
    # Empty or invalid path
    if waypoints_world is None or len(waypoints_world) == 0:
        return torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32), current_wp_idx

    num_wps = len(waypoints_world)
    # Clamp current index to valid range (do not append/alter original path)
    if current_wp_idx >= num_wps:
        current_wp_idx = num_wps - 1

    base_xy, base_yaw = get_base_xy_yaw(manager_env)

    # 1) Find nearest forward waypoint to avoid going backwards
    #    Only consider waypoints from current_wp_idx forward
    dists = [
        (i, math.hypot(waypoints_world[i][0] - base_xy[0], waypoints_world[i][1] - base_xy[1]))
        for i in range(current_wp_idx, num_wps)
    ]
    nearest_idx, nearest_dist = min(dists, key=lambda t: t[1])
    current_wp_idx = max(current_wp_idx, nearest_idx)

    # 2) Choose a lookahead target a few meters ahead along the path
    #    Use a simple rule: first waypoint at distance >= lookahead_distance from robot
    Ld = max(min_lookahead, float(lookahead_distance))
    target_idx = num_wps - 1
    for i in range(current_wp_idx, num_wps):
        di = math.hypot(waypoints_world[i][0] - base_xy[0], waypoints_world[i][1] - base_xy[1])
        if di >= Ld:
            target_idx = i
            break

    # For smoother approach, if we're on the final segment, adapt lookahead to remaining distance
    final_wp = waypoints_world[-1]
    dist_to_goal = math.hypot(final_wp[0] - base_xy[0], final_wp[1] - base_xy[1])
    if dist_to_goal < Ld:
        Ld = max(min_lookahead, dist_to_goal)

    # If no waypoint satisfies the Ld condition and we are not at the end,
    # try to pick a point between target_idx-1 and target_idx for continuity
    # Otherwise, simply target the last waypoint
    target_wp = waypoints_world[target_idx]

    # 3) Compute control to the lookahead target in body frame (pure-pursuit style)
    dx = target_wp[0] - base_xy[0]
    dy = target_wp[1] - base_xy[1]
    desired_yaw = math.atan2(dy, dx)
    ang_err = wrap_to_pi(desired_yaw - base_yaw)

    ex_b, ey_b = world_to_body(dx, dy, base_yaw)

    # Linear velocities: proportional to lookahead vector, clipped by given maxima
    vx = float(np.clip(ex_b, -max_vx, max_vx))
    vy = float(np.clip(ey_b, -max_vy, max_vy))

    # Angular velocity: proportional to heading error, clipped by given maxima
    vw = float(np.clip(k_p_ang * ang_err, -max_yaw_rate, max_yaw_rate))

    # 4) Near-goal handling to avoid singularity and oscillation
    is_final_segment = (current_wp_idx >= num_wps - 2)
    if is_final_segment:
        # Tighten arrival checks on the last segment
        arrive_dist_eff = term_dist
        arrive_yaw_eff = term_yaw
    else:
        arrive_dist_eff = arrive_dist
        arrive_yaw_eff = arrive_yaw

    # If we are very close to the final goal, command a gentle stop
    if dist_to_goal < term_dist and abs(ang_err) < arrive_yaw_eff:
        vx, vy, vw = 0.0, 0.0, 0.0
        current_wp_idx = num_wps - 1

    # 5) Waypoint progression: progress when near the current waypoint (not only at final)
    wp_now = waypoints_world[current_wp_idx]
    d_now = math.hypot(wp_now[0] - base_xy[0], wp_now[1] - base_xy[1])
    if d_now < arrive_dist_eff and current_wp_idx < num_wps - 1:
        current_wp_idx += 1
        print(f"[PLAN] Reached waypoint {current_wp_idx}/{num_wps}")

    command = torch.tensor([[vx, vy, vw]], device=device, dtype=torch.float32)
    return command, current_wp_idx
