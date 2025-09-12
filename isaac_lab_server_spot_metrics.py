# python isaac_lab_server_spot_4.py --enable_cameras --scene_folder /data/isaac_scenes_v1/ --episode_path episodes/test.json
# Isaac Lab version of Spot robot server with Matterport scene


import argparse
import os
import sys
import numpy as np
import torch
import omni
import io
import time
import math
from threading import Thread
from pathlib import Path

# start simulation
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Isaac Lab Server for Spot robot with USD scene")

import utils.rsl_rl_cli_args as rsl_rl_cli_args  # isort: skip
import utils.vln_args as vln_cli_args

rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# Launch Isaac Lab app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import carb, os
settings = carb.settings.get_settings()

MDL_DIRS = [
    "/home/junzhewu/data/isaac_scenes_v1/home_scenes/Materials",
    "/home/junzhewu/data/isaac_scenes_v1/grscenes/Materials",
]
settings.set("/rtx/materials/mdl/searchPaths", MDL_DIRS)
settings.set("/rtx/mdl/searchPaths", MDL_DIRS)
settings.set("/rtx/materials/mdl/shader_search_paths", MDL_DIRS)


# Isaac Lab imports
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import carb
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.sim import SimulationContext, PhysicsMaterialCfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg

# Isaac Lab pretrained spot policy 
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

TASK = "Isaac-Velocity-Flat-Spot-v0"
RL_LIBRARY = "rsl_rl"

# Local imports
from utils.server import run_server, format_data
from utils.episode import VLNEpisodes
from utils.vln_env_wrapper import VLNEnvWrapper
from robot.spot_flat_env_cfg import SpotFlatEnvCfg_PLAY

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
    # No onboard planner state here
    return {}

# Start socket server in background
server_thread = Thread(target=run_server, kwargs={
    "data_cb": data_callback, 
    "action_cb": action_callback, 
    "planner_cb": planner_callback
})
server_thread.daemon = True
server_thread.start()
print("[INFO] Socket server started")

# Main simulation loop

# load episodes
episode_list = VLNEpisodes.from_json(args_cli.episode_path, args_cli.episode_type)
current_episode = episode_list[0]

# setup environment
env_cfg = SpotFlatEnvCfg_PLAY()
scene_folder = Path(args_cli.scene_folder)
# env_cfg.load_usd(args_cli.scene_path)
env_cfg.load_usd(scene_folder / current_episode["scene_path"])

env_cfg.scene.robot.init_state.pos = current_episode["start_position"]
env_cfg.scene.robot.init_state.rot = current_episode["start_rotation"]      

env_cfg.sim.device = args_cli.device
env_cfg.curriculum = None
manager_env = ManagerBasedRLEnv(cfg=env_cfg)

agent_cfg: RslRlOnPolicyRunnerCfg = rsl_rl_cli_args.parse_rsl_rl_cfg(TASK, args_cli)
env = RslRlVecEnvWrapper(manager_env)
ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
ppo_runner.load(checkpoint)
policy = ppo_runner.get_inference_policy(device=args_cli.device)

all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
env = VLNEnvWrapper(env, policy, "spot", current_episode, measure_names=all_measures)
print("[INFO] Env setup complete")

"""Main simulation loop"""
print("[INFO]: Starting simulation")
while simulation_app.is_running():
    if first_step or reset_needed:
        obs, _ = env.reset()
        scene_scale = current_episode["scene_scale"]
        if scene_scale != 1.0:
            terrain_prim = manager_env.scene.stage.GetPrimAtPath('/World/ground/terrain')
            terrain_prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(scene_scale, scene_scale, scene_scale))
        first_step = False
        reset_needed = False
        print(f"[INFO]: Resetting robot state..")

    with torch.inference_mode():
        # Policy forward pass
        vel_command = torch.tensor([base_command], device=args_cli.device, dtype=torch.float32)
        obs, _, done, info = env.step(vel_command)
        # action = policy(obs)
        # obs, _, _, _ = env.step(action)
        # obs[:, 9:12] = command
        print('command: ', vel_command)

    # --- Capture camera data and robot pose ---
    # Get camera data
    rgb_t = manager_env.scene["pov_camera"].data.output['rgb'] 
    rgb_np = rgb_t[0].cpu().numpy()[..., :3].astype(np.uint8)
    
    # Convert depth to millimeters
    depth_m_t = manager_env.scene["pov_camera"].data.output['distance_to_image_plane'] 
    depth_m_np = depth_m_t[0].cpu().numpy()[..., :3]
    depth_mm = np.nan_to_num(depth_m_np, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
    depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    
    # Get robot position and orientation from Isaac Lab articulation
    # This data is already batched, so we select the first environment's data
    position = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy().astype(np.float32)
    quat_wxyz = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
    # print("-------------------------------")
    # print("Received spot position: ", position)
    # print("Received spot rot: ", quat_wxyz)
    
    # Update shared buffers
    if(_latest_rgb is None or _latest_depth is None or _latest_position is None or _latest_quat_wxyz is None):
        print("Data missing")
    _latest_rgb = rgb_np
    _latest_depth = depth_uint16
    _latest_position = position
    _latest_quat_wxyz = quat_wxyz

simulation_app.close()