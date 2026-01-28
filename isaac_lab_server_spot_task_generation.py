# python isaac_lab_server_spot_task_generation.py --enable_cameras --scene_folder /home/junzhewu/data/isaac_scenes_v1 --tg_config_path episodes/task_config.yaml --test_id test_generator
import argparse
from sympy.logic import true
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
# default_scene_id = "grCommercial_hostipal"
task_generator = TaskGenerator(args)
test_episode = VLNEpisode(task_generator.get_scene_config(default_scene_id))
vln_sim = VLNSim(args)
vln_sim.reset(test_episode)
vln_sim.step()
task_generator.bind_vln_sim(vln_sim)

# disable socket server
args.disable_socket_server = True
print(f"[INFO] Socket server disabled in task generation")


# Setup UI
from utils.ui import TaskGeneratorUI
task_generator_ui = TaskGeneratorUI(vln_sim, task_generator)
task_generator_ui.update_ui("scene_id", test_episode.scene_id)

# Generate task
import os
import sys
import signal

mode = "auto"
# mode = "manual"

if mode == "auto":
    try:
        if args.test_scene_id != "none":
            scene_id = args.test_scene_id
            scene_config = task_generator.get_scene_config(scene_id)
            test_episode = VLNEpisode(scene_config)
            task_generator_ui.update_ui("scene_id", scene_id)
            task_generator.save_status(scene_id, status="start")

            # check if task is already generated
            task_done = task_generator.load_status(scene_id)=="success"
            if task_done:
                print(f"[TG] Task already generated for {scene_id}, quitting...")
                task_generator.save_status(scene_id, status="success")
                os.kill(os.getpid(), signal.SIGKILL)

            # otherwise, load scene and generate task
            with task_generator.timing_status(scene_id, "load_scene"):
                vln_sim.reset(test_episode)
                vln_sim.step()
            with task_generator.timing_status(scene_id, "check_navmesh"):
                task_generator.check_navmesh(scene_id)
                task_generator_ui.teleport_robot()

            # generate BEV map
            with task_generator.timing_status(scene_id, "create_bev"):
                task_generator.create_bev_map(scene_id, file_name=f"bev_map", clip_range="ceiling", ceiling_height=scene_config["ceiling_height"])
                for i in range(200):
                    vln_sim.step()
                    if not task_generator.bev_camera_lock.locked():
                        print(f"[TG] BEV camera job completed")
                        break
                task_generator.create_bev_map(scene_id, file_name=f"nav_bev_map", clip_range="robot")
                for i in range(10000):
                    vln_sim.step()
                    if not task_generator.bev_camera_lock.locked():
                        print(f"[TG] BEV camera job completed")
                        break
            with task_generator.timing_status(scene_id, "generate_episodes"):
                task_generator.generate_episodes(scene_id)
                while True:
                    vln_sim.step()
                    if task_generator.generate_finished:
                        break
            task_generator.save_status(scene_id, status="success")
    except Exception as e:
        print(f"[TG] Error: {e}")
        task_generator.save_status(scene_id, status="error", error=str(e))
else:
    """Main simulation loop"""
    print("[INFO]: Starting simulation")
    while simulation_app.is_running():
        vln_sim.step()

os.kill(os.getpid(), signal.SIGKILL)