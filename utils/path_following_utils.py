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

def _to_xyz(pt, default_z):
    if pt is None:
        return None
    if len(pt) == 3:
        return [float(pt[0]), float(pt[1]), float(pt[2])]
    return [float(pt[0]), float(pt[1]), float(default_z)]


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



def visualize_path(manager_env, path_xyz, target_xyz=None, dot_size=0.05, line_width=0.03):
    """Create USD prims to visualize waypoints and path line."""
    stage = manager_env.scene.stage
    root_path = Sdf.Path("/World/PathVis")
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)

    GREEN = Gf.Vec3f(0.0, 1.0, 0.0)
    RED   = Gf.Vec3f(1.0, 0.0, 0.0)

    if path_xyz and len(path_xyz) >= 1:
        pts = [Gf.Vec3f(p[0], p[1], p[2]) for p in path_xyz]

        pts_prim = UsdGeom.Points.Define(stage, root_path.AppendPath("Waypoints"))
        pts_prim.CreatePointsAttr(pts)
        pts_prim.CreateWidthsAttr([dot_size] * len(pts))
        pts_prim.CreateDisplayColorAttr([GREEN] * len(pts))

        curve = UsdGeom.BasisCurves.Define(stage, root_path.AppendPath("PathLine"))
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([len(pts)])
        curve.CreatePointsAttr(pts)
        curve.CreateWidthsAttr([line_width] * len(pts))
        curve.CreateDisplayColorAttr([RED])

    if target_xyz is not None:
        tprim = UsdGeom.Points.Define(stage, root_path.AppendPath("Target"))
        tprim.CreatePointsAttr([Gf.Vec3f(*target_xyz)])
        tprim.CreateWidthsAttr([dot_size * 2.5])
        tprim.CreateDisplayColorAttr([GREEN])
        
        


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


def follow_waypoints(manager_env, policy, obs, device, waypoints_world, current_wp_idx,
                     max_v=0.6, max_yaw_rate=1.0, k_p_ang=1.5, arrive_thresh=0.25):
    """Pure pursuit style waypoint follower. Returns updated obs and wp_idx."""
    if waypoints_world is None or len(waypoints_world) == 0:
        return obs, current_wp_idx

    if current_wp_idx >= len(waypoints_world):
        current_wp_idx = len(waypoints_world) - 1
        last_wp = waypoints_world[-1]
        waypoints_world.append([last_wp[0] + 1.0, last_wp[1], last_wp[2]])
# if waypoints_world is None or len(waypoints_world) == 0:
#     return obs, current_wp_idx

# if current_wp_idx >= len(waypoints_world):
#     current_wp_idx = len(waypoints_world) - 1  # stick to last waypoint


    base_xy, base_yaw = get_base_xy_yaw(manager_env)
    wp = waypoints_world[current_wp_idx]
    dx = wp[0] - base_xy[0]
    dy = wp[1] - base_xy[1]
    dist = math.hypot(dx, dy)

    desired_yaw = math.atan2(dy, dx)
    ang_err = wrap_to_pi(desired_yaw - base_yaw)

    arrive_dist = max(0.1, min(0.15, arrive_thresh))
    arrive_yaw = 0.25

    if abs(ang_err) > arrive_yaw:
        v, vy, w = 0.0, 0.0, max(-max_yaw_rate, min(max_yaw_rate, k_p_ang * ang_err))
    else:
        ex_b, ey_b = world_to_body(dx, dy, base_yaw)
        v = max(-max_v, min(max_v, 0.6 * ex_b))
        vy = 0.0
        w = max(-max_yaw_rate, min(max_yaw_rate, k_p_ang * ang_err))
        if abs(v) < 0.05:
            v = 0.05 * (1.0 if ex_b >= 0.0 else -1.0)

    if (dist < arrive_dist) and (abs(ang_err) < arrive_yaw):
        current_wp_idx += 1
        print(f"[PLAN] Reached waypoint {current_wp_idx}/{len(waypoints_world)}")

    command = torch.tensor([[v, vy, w]], device=device, dtype=torch.float32)

    return command, current_wp_idx
