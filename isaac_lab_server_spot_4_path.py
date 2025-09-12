# cd pohsun/SG-VLN/robot_env
# python isaac_lab_server_spot_4_path.py --enable_cameras --scene_path /home/junzhewu/data/isaac_scenes_v1/nvidia_flatten/park_morning/park_morning_edit.usd --navmesh_path /home/junzhewu/pohsun/SG-VLN/robot_env/path_navmesh/pyrecast/navmesh_morning_parking.obj
# ../IsaacLab/isaaclab.sh -p robot_env/isaac_lab_server_spot_4_path.py --enable_cameras --scene_path /home/junzhewu/data/isaac_scenes_v1/nvidia_flatten/park_morning/park_morning_edit.usd --navmesh_path /home/junzhewu/pohsun/SG-VLN/robot_env/path_navmesh/pyrecast/navmesh_morning_parking.obj
# ../IsaacLab/isaaclab.sh -p robot_env/isaac_lab_server_spot_4_path.py --enable_cameras --scene_path /home/junzhewu/data/isaac_scenes_v1/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd --navmesh_path /home/junzhewu/pohsun/SG-VLN/robot_env/path_navmesh/pyrecast/navmesh_morning_parking.obj
# Isaac Lab version of Spot robot server with Matterport scene


import argparse
import sys
import os
import numpy as np
import torch
import omni
import io
import time
import math
from threading import Thread

# from path_navmesh import usd_utils
sys.path.append("/home/junzhewu/pohsun/SG-VLN/robot_env/path_navmesh")

# start simulation
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Isaac Lab Server for Spot robot with USD scene")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--scene_path", type=str, default="/home/junzhewu/data/isaac_scenes_v1/nvidia_flatten/park_morning/park_morning_edit.usd", help="Path to USD scene file")
parser.add_argument("--robot_pos", type=str, default="-152, 90, 1", help="Robot initial position (x,y,z)")
parser.add_argument("--navmesh_path", type=str, help="Path to preloaded navmesh obj file") #/home/junzhewu/pohsun/SG-VLN/robot_env/path_navmesh/pyrecast/navmesh_morning_parking.obj

sys.path.append("/home/junzhewu/pohsun/IsaacLab/")
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Lab app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb, os
settings = carb.settings.get_settings()

MDL_DIRS = [
    "/home/junzhewu/data/isaac_scenes_v1/grscenes_home/Materials",
    "/home/junzhewu/data/isaac_scenes_v1/grscenes_commercial/Materials",
]
settings.set("/rtx/materials/mdl/searchPaths", MDL_DIRS)
settings.set("/rtx/mdl/searchPaths", MDL_DIRS)
settings.set("/rtx/materials/mdl/shader_search_paths", MDL_DIRS)
settings.set_bool("/rtx/translucency/enabled", True)

# Isaac Lab imports
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import utils.navmesh_utils as navmesh_utils
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
from robot.spot_flat_env_cfg import SpotFlatEnvCfg_PLAY

# Parse robot position
robot_pos = [float(x) for x in args_cli.robot_pos.split(',')]

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

# Main simulation loop
print("[INFO]: Setup complete...")

# --- setup environment --- #
env_cfg = SpotFlatEnvCfg_PLAY()
env_cfg.load_usd(args_cli.scene_path)
env_cfg.sim.device = args_cli.device
env_cfg.curriculum = None
manager_env = ManagerBasedRLEnv(cfg=env_cfg)
print("[INFO]: Env setup complete...")


agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
env = RslRlVecEnvWrapper(manager_env)
ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
ppo_runner.load(checkpoint)
policy = ppo_runner.get_inference_policy(device=args_cli.device)


"""Main simulation loop"""
print("[INFO]: Starting simulation")
while simulation_app.is_running():
    if first_step or reset_needed:
        obs, _ = env.reset()
        if env_cfg.usd_path is not None:
            terrain_prim = manager_env.scene.stage.GetPrimAtPath('/World/ground/terrain')
            terrain_prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(0.01, 0.01, 0.01))
            
        # ----- Path planning using navmesh ----- #
        navmesh_file = args_cli.navmesh_path

        navmeshInterface = navmesh_utils.NavmeshInterface(up_axis='Z', stage=manager_env.scene.stage)
        
        if navmesh_file is None:
            selected_paths = ["/World/ground/terrain"]
            navmeshInterface.setup_navmesh(selected_paths)        
        else:
            navmeshInterface.setup_navmesh_from_file(navmesh_file)

        # Build the navmesh
        navmeshInterface.build_navmesh({
            "cellSize": 0.3,
            "cellHeight": 0.2,
            "agentHeight": 0.5,
            "agentRadius": 0.6,
            "agentMaxClimb": 0.5,
            "agentMaxSlope": 45.0,
            "regionMinSize": 8,
            "regionMergeSize": 20,
            "edgeMaxLen": 12.0,
            "edgeMaxError": 1.3,
            "vertsPerPoly": 6.0,
            "detailSampleDist": 6.0,
            "detailSampleMaxError": 1.0,
            "partitionType": 0
        })
        
        # Visualize the navmesh
        navmeshInterface.visualize_navmesh()
 
        # Find path between two points
        s = [-88.731, -40.245, 0.230012]
        e = [-55.9802, -57.2265, 0.318986]
        navmeshInterface.get_path_from_two_points(s, e)
        # ----- End of path planning using navmesh ----- #

        first_step = False
        reset_needed = False
        root_state = torch.tensor([s + [1.0, 0.0, 0.0, 0.0]], device=args_cli.device, dtype=torch.float32)
        manager_env.scene["robot"].write_root_pose_to_sim(root_state)
        print(f"[INFO]: Resetting robot state..")


    with torch.inference_mode():
        # Policy forward pass
        command = torch.tensor([[base_command[0], base_command[1], base_command[2]]], device=args_cli.device, dtype=torch.float32)
        action = policy(obs)
        obs, _, _, _ = env.step(action)
        obs[:, 9:12] = command
        # print('command: ', command)

    # print("random start: ", s)
    # print("random goal: ", e)
    # print(f"Path from robot to random goal: {navmeshInterface.path_points}")
    # ----- Capture camera data and robot pose ----- #
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