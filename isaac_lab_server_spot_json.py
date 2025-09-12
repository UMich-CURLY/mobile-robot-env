
import argparse
import os
import sys
import json
import numpy as np
import torch
import io
import time
import math
from threading import Thread

# start simulation
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Isaac Lab Server for Spot robot with USD scene")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--scene_path", type=str, default="/home/junzhewu/data/isaac_scenes_v1/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd", help="Path to USD scene file")
parser.add_argument("--robot_pos", type=str, default="7.5799,0.06484971195459366,1", help="Robot initial position (x,y,z)")

# JSON plan args
parser.add_argument("--plan_json", type=str, default=None, help="Path to JSON plan file")
parser.add_argument("--plan_scene_id", type=str, default=None, help="Top-level key in the plan JSON; defaults to USD basename")
parser.add_argument("--plan_key", type=str, default=None, help="Inner key in the plan JSON (e.g., object name). Defaults to the first one found.")
parser.add_argument("--base_height", type=float, default=0.6, help="Height used when plan points are 2D")
parser.add_argument("--use_plan", action="store_true", help="If set, follow the path from the plan JSON")

# Waypoint follower params
parser.add_argument("--waypoint_stride", type=int, default=15, help="Keep every k-th waypoint from the JSON path (>=1).")
parser.add_argument("--arrive_thresh", type=float, default=0.25, help="Meters to consider waypoint reached")
parser.add_argument("--max_v", type=float, default=0.6, help="Max forward speed (m/s)")
parser.add_argument("--max_yaw_rate", type=float, default=1.0, help="Max yaw rate (rad/s)")
parser.add_argument("--k_p_ang", type=float, default=1.5, help="Heading P gain (for yaw rate)")

sys.path.append("/home/junzhewu/pohsun/IsaacLab/")
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Lab app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports
from pxr import Usd, UsdGeom, Gf
import carb
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

TASK = "Isaac-Velocity-Flat-Spot-v0"
RL_LIBRARY = "rsl_rl"

# Local imports
from utils.server import run_server, format_data
from robot.spot_flat_env_cfg import SpotFlatEnvCfg_PLAY

# ---------------------------- JSON PLAN HELPERS ---------------------------- #
def _infer_scene_id_from_usd(plan_data, scene_path):
    usd_base = os.path.basename(scene_path)
    candidates = [k for k in plan_data.keys() if k in (usd_base, usd_base.replace(".usd", ""))]
    if candidates:
        return candidates[0]
    # fallback: first key
    return next(iter(plan_data.keys()))

def _to_xyz(pt, default_z):
    if pt is None: return None
    if len(pt) == 3:
        return [float(pt[0]), float(pt[1]), float(pt[2])]
    return [float(pt[0]), float(pt[1]), float(default_z)]

def load_plan(plan_json_path, scene_path, plan_scene_id=None, plan_key=None, default_z=0.6):
    with open(plan_json_path, "r") as f:
        data = json.load(f)

    scene_id = plan_scene_id or _infer_scene_id_from_usd(data, scene_path)
    if scene_id not in data:
        raise KeyError(f"Scene id '{scene_id}' not found in plan JSON.")
    scene_block = data[scene_id]

    if plan_key is None:
        plan_key = next(iter(scene_block.keys()))
    if plan_key not in scene_block:
        raise KeyError(f"Plan key '{plan_key}' not found under scene '{scene_id}'.")

    route = scene_block[plan_key][0]
    start = _to_xyz(route.get("start_point"), default_z)
    target = _to_xyz(route.get("target_point"), default_z)
    path_raw = route.get("path", [])
    path_xyz = [_to_xyz(p, default_z) for p in path_raw]

    return {
        "scene_id": scene_id,
        "plan_key": plan_key,
        "start": start,
        "target": target,
        "path": path_xyz,
        "distance": route.get("distance", None),
    }

# ---------------------------- SERVER + GLOBALS ---------------------------- #
# Global variables
first_step = True
reset_needed = False
base_command = np.zeros(3)

# Shared buffers for server callbacks
_latest_rgb = None
_latest_depth = None
_latest_position = None
_latest_quat_wxyz = None

# Socket server integration (control Spot via external commands)
def action_callback(msg_type, message):
    global base_command
    # Expect messages of type 'VEL' with fields x, y, omega
    if msg_type == 'VEL':
        base_command = np.array([float(message.x), float(message.y), float(message.omega)])
        print("base_command: ", base_command)

def data_callback():
    global _latest_rgb, _latest_depth, _latest_position, _latest_quat_wxyz
    if _latest_rgb is None or _latest_depth is None or _latest_position is None or _latest_quat_wxyz is None:
        print("Data missing")
        return None
    return format_data(_latest_rgb, _latest_depth, _latest_position, _latest_quat_wxyz)

def planner_callback():
    # You could add current waypoint index, etc., here if you want.
    return {}

# Start socket server in background
server_thread = Thread(target=run_server, kwargs={
    "data_cb": data_callback,
    "action_cb": action_callback,
    "planner_cb": planner_callback
})
server_thread.daemon = True
server_thread.start()

# ---------------------------- ENV + POLICY SETUP ---------------------------- #
# Parse robot position (fallback if plan not provided)
cli_robot_pos = [float(x) for x in args_cli.robot_pos.split(',')]

env_cfg = SpotFlatEnvCfg_PLAY()
env_cfg.load_usd(args_cli.scene_path)

# Load plan if requested
plan = None
waypoints_world = None
current_wp_idx = 0

if args_cli.plan_json is not None:
    plan = load_plan(
        args_cli.plan_json,
        args_cli.scene_path,
        plan_scene_id=args_cli.plan_scene_id,
        plan_key=args_cli.plan_key,
        default_z=args_cli.base_height,
    )
    print(f"[PLAN] scene_id={plan['scene_id']} plan_key={plan['plan_key']}")
    # spawn at start if provided
    if plan["start"] is not None:
        spawn_pos = tuple(plan["start"])
    else:
        spawn_pos = tuple(cli_robot_pos)
else:
    spawn_pos = tuple(cli_robot_pos)

# Set init pose in CONFIG (ensures every reset sticks)
env_cfg.scene.robot.init_state.pos = spawn_pos
env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

env_cfg.sim.device = args_cli.device
env_cfg.curriculum = None
manager_env = ManagerBasedRLEnv(cfg=env_cfg)
print("[INFO]: Env setup complete...")

# Isaac Lab pretrained spot policy
agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
env = RslRlVecEnvWrapper(manager_env)
ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
ppo_runner.load(checkpoint)
policy = ppo_runner.get_inference_policy(device=args_cli.device)

# ---------------------------- PATH VISUALIZATION ---------------------------- #
from pxr import Sdf

def visualize_path(path_xyz, target_xyz=None, dot_size=0.05, line_width=0.03):
    """Create USD prims to visualize waypoints (small dots) and a thin polyline."""
    stage = manager_env.scene.stage
    root_path = Sdf.Path("/World/PathVis")
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)

    GREEN = Gf.Vec3f(0.0, 1.0, 0.0)
    RED   = Gf.Vec3f(1.0, 0.0, 0.0)

    if path_xyz and len(path_xyz) >= 1:
        pts = [Gf.Vec3f(p[0], p[1], p[2]) for p in path_xyz]

        # Waypoint dots (green)
        pts_prim = UsdGeom.Points.Define(stage, root_path.AppendPath("Waypoints"))
        pts_prim.CreatePointsAttr(pts)
        pts_prim.CreateWidthsAttr([dot_size] * len(pts))
        # set per-point displayColor so the renderer definitely uses green
        pts_prim.CreateDisplayColorAttr([GREEN] * len(pts))

        # Polyline (red)
        curve = UsdGeom.BasisCurves.Define(stage, root_path.AppendPath("PathLine"))
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([len(pts)])
        curve.CreatePointsAttr(pts)
        curve.CreateWidthsAttr([line_width] * len(pts))
        # constant red for the whole curve (single color entry is fine)
        curve.CreateDisplayColorAttr([RED])

    if target_xyz is not None:
        # Target: bigger green dot (or change color if you prefer)
        tprim = UsdGeom.Points.Define(stage, root_path.AppendPath("Target"))
        tprim.CreatePointsAttr([Gf.Vec3f(*target_xyz)])
        tprim.CreateWidthsAttr([dot_size * 2.5])
        tprim.CreateDisplayColorAttr([GREEN])
        
    

# Build waypoints from plan and visualize
if plan is not None and plan["path"] and args_cli.use_plan:
    raw = [[float(p[0]), float(p[1]), float(p[2])] for p in plan["path"]]

    # --- simple decimation: keep every k-th waypoint ---
    k = max(1, int(args_cli.waypoint_stride))
    thinned = raw[::k]
    if not thinned or thinned[-1] != raw[-1]:
        thinned.append(raw[-1])  # ensure we end at the true goal

    waypoints_world = thinned
    visualize_path(waypoints_world, target_xyz=plan["target"])
else:
    waypoints_world = None

# ---------------------------- HELPERS ---------------------------- #
def quat_wxyz_to_yaw(wxyz):
    w, x, y, z = wxyz
    # yaw (Z) from quaternion wxyz
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def pure_pursuit_command(base_xy, base_yaw, wp_xy, max_v, max_yaw_rate, k_p_ang):
    dx = wp_xy[0] - base_xy[0]
    dy = wp_xy[1] - base_xy[1]
    dist = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx)
    ang_err = math.atan2(math.sin(desired_yaw - base_yaw), math.cos(desired_yaw - base_yaw))
    v = max_v * max(0.0, math.cos(ang_err))  # slow if turned away
    yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, k_p_ang * ang_err))
    return v, 0.0, yaw_rate, dist

def wrap_to_pi(a): return math.atan2(math.sin(a), math.cos(a))

def get_base_xy_yaw():
    pos = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy()
    quat = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy()
    w, x, y, z = quat
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return (float(pos[0]), float(pos[1])), yaw

def world_to_body(dx, dy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c*dx - s*dy, s*dx + c*dy



# ---------------------------- MAIN SIM LOOP ---------------------------- #
print("[INFO]: Starting simulation")
while simulation_app.is_running():
    if first_step or reset_needed:
        obs, _ = env.reset()

        # Optional: terrain rescale (matches your original)
        # if env_cfg.usd_path is not None:
        #     terrain_prim = manager_env.scene.stage.GetPrimAtPath('/World/ground/terrain')
        #     if terrain_prim:
        #         terrain_prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(0.01, 0.01, 0.01))

        # After reset, enforce initial full root state once (pos, rot, zero vels)
        root_state = torch.zeros((env.num_envs, 13), device=args_cli.device, dtype=torch.float32)
        root_state[:, 0:3] = torch.tensor(spawn_pos, device=args_cli.device)
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=args_cli.device)
        manager_env.scene["robot"].write_root_state_to_sim(root_state)
        manager_env.sim.step()  # ensure sensors update

        first_step = False
        reset_needed = False
        print(f"[INFO]: Resetting robot state to {spawn_pos}..")

    with torch.inference_mode():
        # Choose command: follow plan or accept external velocity
        if args_cli.use_plan and waypoints_world and current_wp_idx < len(waypoints_world):
            base_xy, base_yaw = get_base_xy_yaw()
            wp = waypoints_world[current_wp_idx]
            dx = wp[0] - base_xy[0]
            dy = wp[1] - base_xy[1]
            dist = math.hypot(dx, dy)

            desired_yaw = math.atan2(dy, dx)
            ang_err = wrap_to_pi(desired_yaw - base_yaw)

            # tune these to be reasonably tight so we don't “cut” corners
            arrive_dist = max(0.05, min(0.15, args_cli.arrive_thresh))   # 5–15 cm
            arrive_yaw  = 0.25                                           # ~14 deg

            # 1) turn-in-place until roughly facing the waypoint
            if abs(ang_err) > arrive_yaw:
                v  = 0.0
                vy = 0.0            # set to 0 to avoid lateral shortcutting
                w  = max(-args_cli.max_yaw_rate, min(args_cli.max_yaw_rate, args_cli.k_p_ang * ang_err))
            else:
                # 2) move toward waypoint
                ex_b, ey_b = world_to_body(dx, dy, base_yaw)
                # Conservative gains to hug the path tightly
                kx = 0.6
                ky = 0.0            # keep vy=0 for strict “no cut”; set ky>0.5 if you want lateral nudging
                v  = max(-args_cli.max_v, min(args_cli.max_v, kx * ex_b))
                vy = max(-args_cli.max_v, min(args_cli.max_v, ky * ey_b))
                w  = max(-args_cli.max_yaw_rate, min(args_cli.max_yaw_rate, args_cli.k_p_ang * ang_err))

                # clamp very small forward speeds to avoid dithering
                if abs(v) < 0.05:
                    v = 0.05 * (1.0 if ex_b >= 0.0 else -1.0)

            # 3) strict arrival: need BOTH close + aligned
            if (dist < arrive_dist) and (abs(ang_err) < arrive_yaw):
                current_wp_idx += 1
                print(f"[PLAN] Reached waypoint {current_wp_idx}/{len(waypoints_world)}")

            command = torch.tensor([[v, vy, w]], device=args_cli.device, dtype=torch.float32)
        else:
            command = torch.tensor([[base_command[0], base_command[1], base_command[2]]],
                                device=args_cli.device, dtype=torch.float32)

        # inject BEFORE forward
        obs[:, 9:12] = command
        action = policy(obs)
        obs, _, _, _ = env.step(action)
        # print('command: ', command.tolist())

    # --- Capture camera data and robot pose ---
    # Get camera data
    rgb_t = manager_env.scene["pov_camera"].data.output['rgb']
    rgb_np = rgb_t[0].cpu().numpy()[..., :3].astype(np.uint8)

    # Convert depth to millimeters (single-channel)
    depth_m_t = manager_env.scene["pov_camera"].data.output['distance_to_image_plane']
    depth_m_np = depth_m_t[0].cpu().numpy()[..., 0]
    depth_mm = np.nan_to_num(depth_m_np, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
    depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)

    # Robot pose
    position = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy().astype(np.float32)
    quat_wxyz = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)

    # Update shared buffers
    _latest_rgb = rgb_np
    _latest_depth = depth_uint16
    _latest_position = position
    _latest_quat_wxyz = quat_wxyz

simulation_app.close()


