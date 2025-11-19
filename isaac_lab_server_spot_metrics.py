
import argparse
import numpy as np
import torch
import time

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
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/translucency/enabled", True)
settings.set("/rtx-flags/ecoMode/enabled", True)

# Local imports
from utils.sim import VLNSim

# setup simulation
vln_sim = VLNSim(args)
vln_sim.load_episode(args.episode_label)

# sim variables
episode_label_list = vln_sim.episode_label_list
episode_list = vln_sim.episode_list
manager_env = vln_sim.manager_env
env = vln_sim.env

# Setup UI
import utils.ui as sim_ui
import utils.ui_utils as ui_utils
import isaacsim.core.utils.prims as prim_utils

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
        ui_utils.btn_builder("Apply Settings", text="Update", on_clicked_fn=lambda: save_settings("episode_runtime"))
        ui_utils.btn_builder("Follow Reference Path", text="Start", on_clicked_fn=lambda: start_following_waypoints())
        ui_utils.btn_builder("Stop Following", text="Stop", on_clicked_fn=lambda: stop_following_waypoints())
        ui_utils.btn_builder("Switch StopCalled State", text="Switch", on_clicked_fn=lambda: env.set_stop_called(vln_sim.robot_index, not env.is_stop_called[vln_sim.robot_index]))
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
    if settings_type == "episode_label":
        print(f"[INFO]: Updating episode id to {selected_value}")
        current_episode = vln_sim.episode_list[vln_sim.episode_label_list.index(selected_value)]
        for key, (value, _) in ui_map.items():
            ui_window.set_ui_value(ui_map, key, current_episode[value])
    elif settings_type == "episode_info":
        ui_window.set_ui_value(ui_map, "episode_info", vln_sim.current_episode["episode_info"])
update_ui("episode_label", vln_sim.current_episode["episode_label"])

def save_settings(settings_type):
    current_episode = dict(vln_sim.current_episode)
    if settings_type == "episode_runtime":
        for key, (value, _) in ui_map.items():
            if key == "episode_info":
                continue
            current_episode[value] = ui_window.get_ui_value(ui_map, key)
        update_ui("episode_info")
        vln_sim.reset(current_episode)

def start_following_waypoints():
    current_episode = vln_sim.current_episode
    goal_positions = np.array([x["location"] for x in current_episode["goals"]])
    dist_to_goals = np.linalg.norm(goal_positions - current_episode["start_position"], axis=1)
    closest_goal = current_episode["goals"][np.argmin(dist_to_goals)]
    ref_path = closest_goal["reference_path"]
    vln_sim.set_waypoints(ref_path[1:], fix_yaw=True)

def stop_following_waypoints():
    vln_sim.clear_waypoints()
    remove_prim("/World/WaypointPath")
    remove_prim("/World/Arrow")

def remove_prim(rule):
    prim_list = prim_utils.find_matching_prim_paths(rule)
    for prim_path in prim_list:
        prim_utils.delete_prim(prim_path)

"""Main simulation loop"""
print("[INFO]: Starting simulation")
start_time = time.time()
start_episode_step = int(manager_env.episode_length_buf)
frame_count = 0
end_time = 0
while simulation_app.is_running():
    with torch.inference_mode():
        vln_sim.step()
        # print("measures: ", vln_sim.info["measurements"])
        # print(f'[{vln_sim.commands_source}] command: {vln_sim.commands}')
    frame_count += 1
    if frame_count == 1:
        first_frame_time = time.time()
        print(f"[INFO]: First frame time: {first_frame_time - sim_start_time:.2f}s")
        pass
    log_fps_interval = 100
    if frame_count % log_fps_interval == 0:
        duration = time.time() - start_time
        sim_time = int(manager_env.episode_length_buf - start_episode_step) * manager_env.step_dt
        print(f"[INFO]: Frame count: {frame_count}, Time: {duration:.2f}s, FPS: {log_fps_interval / duration:.2f}, Sim Time: {sim_time:.2f}s ({sim_time/duration:.2f}x)")
        start_time = time.time()
        start_episode_step = int(manager_env.episode_length_buf)

simulation_app.close()