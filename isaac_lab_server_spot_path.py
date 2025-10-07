### Major changes:
# 1. Remove people from scene
# 2. Test navmesh generation using PyRecastDetour
### Usage:
# cd pohsun/SG-VLN
# ../IsaacLab/isaaclab.sh -p robot_env/isaac_lab_server_spot_path.py --enable_cameras --scene_path /home/junzhewu/pohsun/data_decimated/grscenes_commercial/scenes/MV4AFHQKTKJZ2AABAAAAADQ8_usd/start_result_navigation_no_people.usd
# ../IsaacLab/isaaclab.sh -p robot_env/isaac_lab_server_spot_path.py --enable_cameras --scene_path /home/junzhewu/data/isaac_scenes_v1/grscenes_commercial/scenes/MV4AFHQKTKJZ2AABAAAAADQ8_usd/start_result_navigation_no_people.usd
# ../IsaacLab/isaaclab.sh -p robot_env/isaac_lab_server_spot_path.py --enable_cameras --scene_path /home/junzhewu/pohsun/test_scene.usd
# DISPLAY=:2 python isaac_lab_server_spot_path.py --enable_cameras --scene_path /home/junzhewu/pohsun/test_scene.usd
# DISPLAY=:2 python isaac_lab_server_spot_path.py --enable_cameras --scene_path /home/junzhewu/data/isaac_scenes_v1/nvidia/park/park_morning_edit.usd
# python isaac_lab_server_spot_path.py --enable_cameras --scene_path /data/isaac_scenes_v1/nvidia/AECDemo_NVD@10012/Demos/AEC/BrownstoneDemo/test.usd
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

# start simulation
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Isaac Lab Server for Spot robot with USD scene")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--scene_path", type=str, default="/home/junzhewu/data/isaac_scenes_v1/nvidia_flatten/park_morning/park_morning_edit.usd", help="Path to USD scene file")
parser.add_argument("--robot_pos", type=str, default="7.5799,0.06484971195459366,1.2", help="Robot initial position (x,y,z)") # -152, 90, 1
parser.add_argument("--navmesh_path", type=str, default=None, help="Path to preloaded navmesh obj file") #/home/junzhewu/pohsun/SG-VLN/robot_env/path_navmesh/pyrecast/navmesh_morning_parking.obj

import utils.rsl_rl_cli_args as rsl_rl_cli_args
rsl_rl_cli_args.add_rsl_rl_args(parser)
import utils.vln_args as vln_cli_args

AppLauncher.add_app_launcher_args(parser)
args = vln_cli_args.parse_args(parser)

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
from utils.socket_server import run_server, format_data
from robot.spot_flat_env_cfg import SpotFlatEnvCfg_PLAY

# Parse robot position
robot_pos = [float(x) for x in args_cli.robot_pos.split(',')]

# Global variables
first_step = True
reset_needed = False
base_command = np.zeros(3)

# Main simulation loop
print("[INFO]: Setup complete...")

# --- setup environment --- #
env_cfg = SpotFlatEnvCfg_PLAY()
env_cfg.load_usd(args_cli.scene_path)
env_cfg.sim.device = args_cli.device
env_cfg.curriculum = None
manager_env = ManagerBasedRLEnv(cfg=env_cfg)


# Initialize navmesh interface
navmeshInterface = navmesh_utils.NavmeshInterface(up_axis='Z', stage=manager_env.scene.stage)


# remove people from scene
# def remove_people_folders(stage):
#     """
#     Recursively find and remove all folders/prim named 'person' in the stage.
#     """
#     for prim in stage.TraverseAll():
#         if prim.GetName().lower() == "person":
#             prim_path = prim.GetPath()
#             person_prim = stage.GetPrimAtPath(prim_path)
#             # Traverse children under person
#             for child in list(person_prim.GetChildren()):
#                 # Check if the prim actually has references
#                 refs = child.GetReferences()
#                 if refs:
#                     print(f"Clearing references for {child.GetPath()}")
#                     refs.ClearReferences()
                
#                 # Remove the child prim itself
#                 print(f"Removing prim {child.GetPath()}")
#                 stage.RemovePrim(child.GetPath())

#             # Finally remove the folder
#             print(f"Removing 'person' folder: {prim_path}")
#             stage.RemovePrim(prim_path)
#     print("[INFO]: Done removing all 'person' folders in the stage.")
# remove_people_folders(stage = manager_env.scene.stage)
print("[INFO]: Env setup complete...")

env_cfg.scene.robot.init_state.pos = robot_pos
env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

agent_cfg: RslRlOnPolicyRunnerCfg = rsl_rl_cli_args.parse_rsl_rl_cfg(TASK, args_cli)
env = RslRlVecEnvWrapper(manager_env)
ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
ppo_runner.load(checkpoint)
policy = ppo_runner.get_inference_policy(device=args_cli.device)


"""Main simulation loop"""
print("[INFO]: Starting simulation")
step_count = 0
while simulation_app.is_running():
    if first_step or reset_needed:
        obs, _ = env.reset()
        if env_cfg.usd_path is not None:
            pass
            # terrain_prim = manager_env.scene.stage.GetPrimAtPath('/World/ground/terrain')
            # terrain_prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(1.3, 1.3, 1.3))
        
        first_step = False
        reset_needed = False
        root_state = torch.tensor([robot_pos + [1.0, 0.0, 0.0, 0.0]], device=args_cli.device, dtype=torch.float32)
        manager_env.scene["robot"].write_root_pose_to_sim(root_state)
        print(f"[INFO]: Resetting robot state..")
        
    if step_count == 10:
        # ----- Path planning using navmesh ----- #
        navmesh_file = args_cli.navmesh_path

        # Setup navmesh from scene
        if navmesh_file is None:
            selected_paths = ["/World/ground/terrain"]
            start_time = time.time()
            navmeshInterface.setup_navmesh(selected_paths)
            navmeshInterface.build_navmesh()
            end_time = time.time()
            print(f"[INFO]: Navmesh build time: {end_time - start_time:.2f} seconds")
        else:
            start_time = time.time()
            navmeshInterface.load_navmesh(navmesh_file)
            end_time = time.time()
            print(f"[INFO]: Navmesh loading time: {end_time - start_time:.2f} seconds")
        # Visualize the navmesh
        navmeshInterface.visualize_navmesh()

        # Find path between two points
        points = navmeshInterface.sample_random_points(1000)
        navmesh_utils.create_points(points, prim_path="/World/RandomPoints", width=80.)
        for i in range(50):
            path = navmeshInterface.find_paths(points[2*i], points[2*i+1])
            navmesh_utils.create_curve(path, prim_path=f"/World/Path_{i}", width=40.)
        if navmesh_file is None:
            navmeshInterface.save_navmesh("episodes/navmesh.bin")

    with torch.inference_mode():
        # Policy forward pass
        command = torch.tensor([[base_command[0], base_command[1], base_command[2]]], device=args_cli.device, dtype=torch.float32)
        action = policy(obs)
        obs, _, _, _ = env.step(action)
        obs[:, 9:12] = command
        # print('command: ', command)
    
    step_count += 1
simulation_app.close()