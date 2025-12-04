# python isaac_lab_server_spot_task_generation.py --enable_cameras --scene_folder /home/junzhewu/data/isaac_scenes_v1 --tg_config_path episodes/task_config.yaml --test_id test_generator

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
args = vln_cli_args.parse_args(parser)

# Launch Isaac Lab app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Enable Extension
# from isaacsim.core.utils.extensions import enable_extension
# enable_extension("omni.anim.navigation.bundle")
# simulation_app.update()
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# print(f"ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")

# Isaac Lab pretrained spot policy 
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

TASK = "Isaac-Velocity-Flat-Spot-v0"
RL_LIBRARY = "rsl_rl"

# Local imports
from utils.sim import VLNSim
from utils.task_generator import TaskGenerator
from utils.episode import VLNEpisode

# setup environment
default_scene_id = "test_generator"
task_generator = TaskGenerator(args)
current_episode = VLNEpisode(task_generator.get_scene_config(default_scene_id))
vln_sim = VLNSim(args)
vln_sim.reset(current_episode)
task_generator.bind_vln_sim(vln_sim)


# disable socket server
args.disable_socket_server = True
print(f"[INFO] Socket server disabled in task generation")


# Setup UI
from utils.ui import TaskGeneratorUI
task_generator_ui = TaskGeneratorUI(vln_sim, task_generator)
task_generator_ui.update_ui("scene_id", default_scene_id)


"""Main simulation loop"""
print("[INFO]: Starting simulation")
while simulation_app.is_running():
    vln_sim.step()

simulation_app.close()