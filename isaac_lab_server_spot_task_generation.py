# python isaac_lab_server_spot_metrics.py --enable_cameras --scene_folder /data/isaac_scenes_v1/ --episode_path episodes/test.json
# python isaac_lab_server_spot_sample.py --enable_cameras --scene_folder /home/junzhewu/data/isaac_scenes_v1 --episode_path episodes/test.json

import argparse
import torch
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
args = parser.parse_args()


# Launch Isaac Lab app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Enable Extension
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.anim.navigation.bundle")
simulation_app.update()
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
print(f"ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")

import carb, os
settings = carb.settings.get_settings()

MDL_DIRS = [
    args.scene_folder + "/grscenes_home/Materials",
    args.scene_folder + "/grscenes_commercial/Materials",
    args.scene_folder + "/nvidia_edit/Materials",
    args.scene_folder + "/nvidia/Materials",
    args.scene_folder + "/umich/Materials",
    args.scene_folder + "/virtual_community/Materials",
    args.scene_folder + "/vc/Materials",
]
settings.set("/rtx/materials/mdl/searchPaths", MDL_DIRS)
settings.set("/rtx/mdl/searchPaths", MDL_DIRS)
settings.set("/rtx/materials/mdl/shader_search_paths", MDL_DIRS)

# Isaac Lab pretrained spot policy 
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

TASK = "Isaac-Velocity-Flat-Spot-v0"
RL_LIBRARY = "rsl_rl"

# Local imports
from utils.episode import VLNEpisodes
from utils.vln_env_wrapper import VLNEnvWrapper
from robot.spot_flat_env_cfg import SpotFlatEnvCfg_PLAY
from utils.innout_sim import InNOutSim
from utils.task_generator import TaskGenerator

# Main simulation loop



# load episodes
task_generator = TaskGenerator(args)
episode_list = task_generator.generate_test_episodes()
current_episode = episode_list[args.test_id]

# setup environment
env_cfg = SpotFlatEnvCfg_PLAY()
scene_folder = Path(args.scene_folder)
# env_cfg.load_usd(args.scene_path)
# env_cfg.load_usd(scene_folder / current_episode["scene_path"])
env_cfg.scene.robot.init_state.pos = current_episode["start_position"]
env_cfg.scene.robot.init_state.rot = current_episode["start_rotation"]      

# env_cfg.viewer.cam_prim_path = '/World/pov_camera'

env_cfg.sim.device = args.device
env_cfg.curriculum = None
manager_env = ManagerBasedRLEnv(cfg=env_cfg)

agent_cfg = rsl_rl_cli_args.parse_rsl_rl_cfg(TASK, args)
env = RslRlVecEnvWrapper(manager_env)
ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
ppo_runner.load(checkpoint)
policy = ppo_runner.get_inference_policy(device=args.device)

all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "SoftSPL", "OracleNavigationError", "OracleSuccess"]
env = VLNEnvWrapper(args, env, policy, "spot", current_episode, measure_names=all_measures)
print("[INFO] Env setup complete")

in_n_out_sim = InNOutSim(args, env)


# Setup UI
from utils.ui import SimWindow
from isaacsim.gui.components import ui_utils
import omni.ui as ui

ui_window = SimWindow(manager_env)
ui_elements = ui_window.ui_elements

with ui_elements["main_frame"]:
    with ui.CollapsableFrame("Navmesh Settings"):
        ui_elements["usd_path"] = ui_utils.str_builder(
            "USD Path",
            use_folder_picker=True,
            default_val=args.scene_folder,
            folder_dialog_title="Select Scene USD",
            bookmark_path=args.scene_folder
        )
        ui_elements["scene_scale"] = ui_utils.combo_floatfield_slider_builder("Scene Scale")[0]
        ui_utils.btn_builder("Load Scene", lambda: in_n_out_sim.load_scene(ui_elements["usd_path"].model.as_string))
    # with omni.ui.CollapsableFrame("Navmesh Settings"):
    #     ui_window.create_float_drag("cellSize", 0.1, 100.0, 0.1, 1.0)
    #     ui_window.create_float_drag("cellHeight", 0.1, 10.0, 0.1, 1.0)
    #     ui_window.create_button("Build Navmesh", "Go!", lambda: build_navmesh())

def build_navmesh():
    print("[INFO]: Building navmesh...")
    print("cellSize: ", ui_window.values["cellSize"])
    print("cellHeight: ", ui_window.values["cellHeight"])

sim_init = False

"""Main simulation loop"""
print("[INFO]: Starting simulation")
while simulation_app.is_running():
    if not sim_init:
        # tg_success = task_generator.reset(env, current_episode["scene_id"])
        # if tg_success:
            # test episode
            # save episode
        obs, _ = env.reset()
        sim_init = True
        print(f"[INFO]: Resetting robot state..")

    with torch.inference_mode():
        # Policy forward pass
        obs, reward, done, info = env.step(in_n_out_sim.commands)
        # print("measures: ", info["measurements"])
        in_n_out_sim.update_obs(obs, manager_env)
        # task_generator.step()
        # print(f'[{in_n_out_sim.commands_source}] command: {in_n_out_sim.commands}')

simulation_app.close()