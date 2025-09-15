# python isaac_lab_server_spot_metrics.py --enable_cameras --scene_folder /data/isaac_scenes_v1/ --episode_path episodes/test.json
# python isaac_lab_server_spot_metrics.py --enable_cameras --scene_folder /home/junzhewu/data/isaac_scenes_v1 --episode_path episodes/test.json

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
    args_cli.scene_folder + "/grscenes_home/Materials",
    args_cli.scene_folder + "/grscenes_commercial/Materials",
    args_cli.scene_folder + "/nvidia_edit/Materials",
    args_cli.scene_folder + "/nvidia/Materials",
    args_cli.scene_folder + "/umich/Materials",
    args_cli.scene_folder + "/virtual_community/Materials",
    args_cli.scene_folder + "/vc/Materials",
]
settings.set("/rtx/materials/mdl/searchPaths", MDL_DIRS)
settings.set("/rtx/mdl/searchPaths", MDL_DIRS)
settings.set("/rtx/materials/mdl/shader_search_paths", MDL_DIRS)


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

class InNOutSim:
    def __init__(self, args_cli):
        self.args_cli = args_cli
        self.device = args_cli.device
        self.robot_index = 0
        
        # Shared buffers for server callbacks
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_position = None
        self._latest_quat_wxyz = None
        self.commands = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        self.commands_source = 'server'

        # initialize keyboard and server
        self.set_up_keyboard()
        self.start_server()

    # Socket server integration (control Spot via external commands)
    def action_callback(self, msg_type, message):
        # Expect messages of type 'VEL' with fields x, y, omega
        if msg_type == 'VEL':
            self.commands[self.robot_index] = torch.tensor([float(message.x), float(message.y), float(message.omega)], device=self.device)
            self.commands_source = 'server_vel'
        elif msg_type == 'WAYPOINTS':
            print("ERROR: Waypoints not supported yet")

    def data_callback(self):
        if self._latest_rgb is None or self._latest_depth is None or self._latest_position is None or self._latest_quat_wxyz is None:
            print("Data missing")
            return None
        return format_data(self._latest_rgb, self._latest_depth, self._latest_position, self._latest_quat_wxyz)

    def planner_callback(self):
        # No onboard planner state here
        return {}

    def start_server(self):
        # Start socket server in background
        server_thread = Thread(target=run_server, kwargs={
            "data_cb": self.data_callback, 
            "action_cb": self.action_callback, 
            "planner_cb": self.planner_callback
        })
        server_thread.daemon = True
        server_thread.start()
        print("[INFO] Socket server started")

    def update_obs(self, obs, manager_env):
        try:
            self._latest_rgb = obs[0, :, :, :3].cpu().numpy().astype(np.uint8)
            depth = obs[0, :, :, 3].cpu().numpy()
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
            depth = np.clip(depth, 0, 65535).astype(np.uint16)
            self._latest_depth = depth
            self._latest_position = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy().astype(np.float32)
            self._latest_quat_wxyz = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error updating obs: {e}")
            import traceback
            traceback.print_exc()

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 2
        R = 2
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([-T, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([0, 0.0, R], device=self.device),
            "RIGHT": torch.tensor([0, 0.0, -R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            if event.input.name in self._key_to_control:
                print("keyboard event: ", event.input.name)
                self.commands[self.robot_index] = self._key_to_control[event.input.name]
                self.commands_source = 'keyboard'
            # Escape key exits out of the current selected robot view
            # elif event.input.name == "ESCAPE":
            #     self._prim_selection.clear_selected_prim_paths()
            # C key swaps between third-person and perspective views
            # elif event.input.name == "C":
            #     if self._selected_id is not None:
            #         if self.viewport.get_active_camera() == self.camera_path:
            #             self.viewport.set_active_camera(self.perspective_path)
            #         else:
            #             self.viewport.set_active_camera(self.camera_path)
        # On key release, the robot stops moving
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.commands[self.robot_index] = self._key_to_control["ZEROS"]

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

# env_cfg.viewer.cam_prim_path = '/World/pov_camera'

env_cfg.sim.device = args_cli.device
env_cfg.curriculum = None
manager_env = ManagerBasedRLEnv(cfg=env_cfg)

agent_cfg: RslRlOnPolicyRunnerCfg = rsl_rl_cli_args.parse_rsl_rl_cfg(TASK, args_cli)
env = RslRlVecEnvWrapper(manager_env)
ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
ppo_runner.load(checkpoint)
policy = ppo_runner.get_inference_policy(device=args_cli.device)

all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "SoftSPL", "OracleNavigationError", "OracleSuccess"]
env = VLNEnvWrapper(env, policy, "spot", current_episode, measure_names=all_measures)
print("[INFO] Env setup complete")

in_n_out_sim = InNOutSim(args_cli)

"""Main simulation loop"""
print("[INFO]: Starting simulation")
while simulation_app.is_running():
    if first_step or reset_needed:
        obs, _ = env.reset()
        # scene_scale = current_episode["scene_scale"]
        # if scene_scale != 1.0:
        #     terrain_prim = manager_env.scene.stage.GetPrimAtPath('/World/ground/terrain')
        #     terrain_prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(scene_scale, scene_scale, scene_scale))
        first_step = False
        reset_needed = False
        print(f"[INFO]: Resetting robot state..")

    with torch.inference_mode():
        # Policy forward pass
        obs, reward, done, info = env.step(in_n_out_sim.commands)
        print("measures: ", info["measurements"])
        in_n_out_sim.update_obs(obs, manager_env)
        # action = policy(obs)
        # obs, _, _, _ = env.step(action)
        # obs[:, 9:12] = command
        print(f'[{in_n_out_sim.commands_source}] command: {in_n_out_sim.commands}')


    # --- Capture camera data and robot pose ---
    # # Get camera data
    # rgb_t = manager_env.scene["pov_camera"].data.output['rgb'] 
    # rgb_np = rgb_t[0].cpu().numpy()[..., :3].astype(np.uint8)
    
    # # Convert depth to millimeters
    # depth_m_t = manager_env.scene["pov_camera"].data.output['distance_to_image_plane'] 
    # depth_m_np = depth_m_t[0].cpu().numpy()[..., :3]
    # depth_mm = np.nan_to_num(depth_m_np, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
    # depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    
    # # Get robot position and orientation from Isaac Lab articulation
    # # This data is already batched, so we select the first environment's data
    # position = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy().astype(np.float32)
    # quat_wxyz = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
    
    # # Update shared buffers
    # if(_latest_rgb is None or _latest_depth is None or _latest_position is None or _latest_quat_wxyz is None):
    #     print("Data missing")
    # _latest_rgb = rgb_np
    # _latest_depth = depth_uint16
    # _latest_position = position
    # _latest_quat_wxyz = quat_wxyz

simulation_app.close()