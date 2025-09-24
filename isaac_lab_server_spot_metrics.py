
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

import utils.rsl_rl_cli_args as rsl_rl_cli_args
import utils.vln_args as vln_cli_args
import utils.wpfollowing_args as wpfollowing_cli_args

rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)
wpfollowing_cli_args.add_wpfollowing_args(parser)



AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sim_start_time = time.time()
# Launch Isaac Lab app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# Enable Extension
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.anim.navigation.bundle")
simulation_app.update()


import carb, os
settings = carb.settings.get_settings()
settings.set("/renderer/multiGPU/enabled", False)
settings.set("/renderer/activeGpu", 0)
settings.set("/rtx/post/dlss/execMode", 1) # 0: Performance, 1: Balanced, 2: Quality, 3: Auto

# Isaac Lab imports
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
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
from utils.path_following_utils import visualize_path, follow_waypoints, load_plan



# Isaac Lab pretrained spot policy 
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from omni.kit.viewport.utility import get_viewport_from_window_name

TASK = "Isaac-Velocity-Flat-Spot-v0"
RL_LIBRARY = "rsl_rl"

# Local imports
from utils.episode import VLNEpisodes
from utils.vln_env_wrapper import VLNEnvWrapper

from robot.spot_flat_env_cfg import SpotFlatEnvCfg_PLAY

from utils.server import run_server, format_data
from utils.innout_sim import InNOutSim

# Main simulation loop

# load episodes
episode_list = VLNEpisodes.from_json(args.episode_path, args.episode_type)
current_episode = episode_list[args.test_id]

# setup environment
env_cfg = SpotFlatEnvCfg_PLAY()
scene_folder = Path(args.scene_folder)
# env_cfg.load_usd(args.scene_path)
env_cfg.load_usd(scene_folder / current_episode["scene_path"])

env_cfg.scene.robot.init_state.pos = current_episode["start_position"]
env_cfg.scene.robot.init_state.rot = current_episode["start_rotation"]      
env_cfg.scene.num_envs = args.num_envs
# env_cfg.viewer.cam_prim_path = '/World/pov_camera'

env_cfg.sim.device = args.device
env_cfg.curriculum = None
manager_env = ManagerBasedRLEnv(cfg=env_cfg)

agent_cfg: RslRlOnPolicyRunnerCfg = rsl_rl_cli_args.parse_rsl_rl_cfg(TASK, args)
env = RslRlVecEnvWrapper(manager_env)
ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
ppo_runner.load(checkpoint)
policy = ppo_runner.get_inference_policy(device=args.device)

all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "SoftSPL", "OracleNavigationError", "OracleSuccess"]
env = VLNEnvWrapper(args, env, policy, "spot", current_episode, measure_names=all_measures)
print("[INFO] Env setup complete")

in_n_out_sim = InNOutSim(args, env)

sim_init = False

if args.use_plan:
    plan = load_plan(args.episode_path, args.scene_folder, args.test_id, args.base_height)
    waypoints_world = plan["path"][::args.waypoint_stride]
    visualize_path(manager_env, waypoints_world, target_xyz=plan["target"])
else:
    waypoints_world = None

current_wp_idx = 0
if args.use_plan:
    waypoints_world = current_episode["goals"][0]["reference_path"][::args.waypoint_stride]
    target = current_episode["goals"][0]["location"]
    visualize_path(manager_env, waypoints_world, target_xyz=target)
else:
    waypoints_world = None

"""Main simulation loop"""
print("[INFO]: Starting simulation")
start_time = time.time()
frame_count = 0
end_time = 0
while simulation_app.is_running():
    if not sim_init:
        obs, _ = env.reset()
        sim_init = True
        print(f"[INFO]: Resetting robot state..")

    with torch.inference_mode():
        # Policy forward pass
        obs, reward, done, info = env.step(in_n_out_sim.commands)
        in_n_out_sim.update_obs(obs, manager_env)
        # print("measures: ", info["measurements"])
        # print(f'[{in_n_out_sim.commands_source}] command: {in_n_out_sim.commands}')
    frame_count += 1
    if frame_count == 1:
        first_frame_time = time.time()
        print(f"[INFO]: First frame time: {first_frame_time - sim_start_time:.2f}s")
        pass
    if frame_count % 100 == 0:
        print(f"[INFO]: Frame count: {frame_count}, Time: {time.time() - start_time:.2f}s, FPS: {100 / (time.time() - start_time):.2f}")
        start_time = time.time()
    if args.use_plan and waypoints_world:
        command, current_wp_idx = follow_waypoints(
            manager_env, policy, obs, args.device, waypoints_world, current_wp_idx
        )
        obs, _, _, _ = env.step(command)  



simulation_app.close()