
import argparse
import numpy as np
import torch
import time
from pathlib import Path

# start simulation
from isaaclab.app import AppLauncher
import utils.rsl_rl_cli_args as rsl_rl_cli_args
import utils.vln_args as vln_cli_args

# Add command line arguments
parser = argparse.ArgumentParser(description="Benchmark")
rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = vln_cli_args.parse_args(parser)

# Launch Isaac Lab app
sim_start_time = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Enable Extension and setup settings
import carb
# from isaacsim.core.utils.extensions import enable_extension
# enable_extension("omni.anim.navigation.bundle")
simulation_app.update()
settings = carb.settings.get_settings()
settings.set("/renderer/multiGPU/enabled", False)
settings.set("/renderer/activeGpu", 0)
settings.set("/rtx/post/dlss/execMode", 1) # 0: Performance, 1: Balanced, 2: Quality, 3: Auto

# Isaac Lab Import
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# Local imports
from utils.episode import VLNEpisode
from utils.vln_env_wrapper import VLNEnvWrapper, init_env_cfg
from robot.spot_flat_env_cfg import SpotFlatEnvCfg_PLAY
from utils.sim import VLNSim

# load episodes
episode_list = VLNEpisode.from_json_folder(args.episode_folder)
episode_label_list = [x.episode_label for x in episode_list]
current_episode = episode_list[episode_label_list.index(args.episode_id)]

# isaac lab manager
env_cfg = SpotFlatEnvCfg_PLAY()
init_env_cfg(env_cfg, args, current_episode)
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device
env_cfg.curriculum = None
manager_env = ManagerBasedRLEnv(cfg=env_cfg)

# policy
TASK = "Isaac-Velocity-Flat-Spot-v0"
RL_LIBRARY = "rsl_rl"
agent_cfg: RslRlOnPolicyRunnerCfg = rsl_rl_cli_args.parse_rsl_rl_cfg(TASK, args)
env = RslRlVecEnvWrapper(manager_env)
ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
ppo_runner.load(checkpoint)
policy = ppo_runner.get_inference_policy(device=args.device)

# simulation
all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "SoftSPL", "OracleNavigationError", "OracleSuccess"]
env = VLNEnvWrapper(args, env, policy, "spot", measure_names=all_measures)
print("[INFO] Env setup complete")

# socket server & keyboard
vln_sim = VLNSim(args, env)

# Setup UI
import utils.ui as sim_ui
import utils.ui_utils as ui_utils
import omni.ui as ui
import omni.usd
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.bounds as bounds_utils
import isaaclab.sim as sim_utils
import json

ui_window = sim_ui.SimWindow(manager_env)
ui_elements = ui_window.ui_elements

with ui_elements["main_stack"]:
    with ui_window.create_frame("Debug Tools"):
        ui_elements["episode_label"] = ui_utils.dropdown_builder(
            "Episode Id",
            items=episode_label_list,
            on_clicked_fn=lambda x: update_ui("episode_label", x)
        )
        ui_elements["instruction"] = ui_utils.str_builder(
            "Instruction",
            default_val="instruction",
        )
        ui_elements["start_position"] = ui_utils.xyz_builder(
            "Start Position",
            default_val=[0.0, 0.0, 0.0]
        )
        ui_utils.btn_builder("Update States", text="Update", on_clicked_fn=lambda: save_settings("episode_runtime"))
        ui_utils.btn_builder("Follow Reference Path", text="Start", on_clicked_fn=lambda: start_following_waypoints())
        ui_utils.btn_builder("Stop Following", text="Stop", on_clicked_fn=lambda: stop_following_waypoints())
        ui_utils.btn_builder("Switch StopCalled State", text="Switch", on_clicked_fn=lambda: env.set_stop_called(not env.is_stop_called))
    with ui_window.create_frame("Episode Info"):
        ui_elements["episode_info"] = ui_utils.ui.Label(
            "Episode Info",
            style_type_name_override="Label::label",
            word_wrap=True,
            alignment=ui_utils.ui.Alignment.LEFT_TOP,
        )
ui_map = {
    "episode_label": ("episode_label", sim_ui.choice_func(episode_label_list)),
    "instruction": ("instruction", sim_ui.str_func),
    "episode_info": ("episode_info", sim_ui.label_func),
    "start_position": ("start_position", sim_ui.xyz_func),
}

def update_ui(settings_type, selected_value=None):
    global current_episode
    if settings_type == "episode_label":
        print(f"[INFO]: Updating episode id to {selected_value}")
        current_episode = episode_list[episode_label_list.index(selected_value)]
        for key, (value, _) in ui_map.items():
            ui_window.set_ui_value(ui_map, key, current_episode[value])
    elif settings_type == "episode_info":
        ui_window.set_ui_value(ui_map, "episode_info", current_episode["episode_info"])
update_ui("episode_label", current_episode["episode_label"])

sim_init = False
def reset_environment():
    global sim_init
    sim_init = False

def save_settings(settings_type):
    if settings_type == "episode_runtime":
        for key, (value, _) in ui_map.items():
            if key == "episode_info":
                continue
            current_episode[value] = ui_window.get_ui_value(ui_map, key)
        update_ui("episode_info")
        reset_environment()

def start_following_waypoints():
    goal_positions = np.array([x["location"] for x in current_episode["goals"]])
    dist_to_goals = np.linalg.norm(goal_positions - current_episode["start_position"], axis=1)
    closest_goal = current_episode["goals"][np.argmin(dist_to_goals)]
    ref_path = closest_goal["reference_path"]
    vln_sim.set_waypoints(ref_path[1:], visualize=True)

def stop_following_waypoints():
    vln_sim.clear_waypoints()
    remove_prim("/World/PathVis")

def remove_prim(rule):
    prim_list = prim_utils.find_matching_prim_paths(rule)
    for prim_path in prim_list:
        prim_utils.delete_prim(prim_path)

"""Main simulation loop"""
print("[INFO]: Starting simulation")
start_time = time.time()
frame_count = 0
end_time = 0
while simulation_app.is_running():
    with torch.inference_mode():
        if not sim_init:
            obs, _ = env.reset(current_episode)
            sim_init = True
            print(f"[INFO]: Resetting env state..")

        # Policy forward pass
        obs, reward, done, info = env.step(vln_sim.commands)
        vln_sim.update_obs(obs, info=info, current_episode=current_episode)
        # print("measures: ", info["measurements"])
        # print(f'[{vln_sim.commands_source}] command: {vln_sim.commands}')
    frame_count += 1
    if frame_count == 1:
        first_frame_time = time.time()
        print(f"[INFO]: First frame time: {first_frame_time - sim_start_time:.2f}s")
        pass
    if frame_count % 100 == 0:
        print(f"[INFO]: Frame count: {frame_count}, Time: {time.time() - start_time:.2f}s, FPS: {100 / (time.time() - start_time):.2f}")
        start_time = time.time()

simulation_app.close()