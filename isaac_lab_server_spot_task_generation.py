# python isaac_lab_server_spot_task_generation.py --enable_cameras --scene_folder /home/junzhewu/data/isaac_scenes_v1 --tg_config_path episodes/task_config.yaml --test_id test_generator

import argparse
import json
import torch
from pathlib import Path
import time

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
args.disable_termination = True

# Launch Isaac Lab app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Enable Extension
# from isaacsim.core.utils.extensions import enable_extension
# enable_extension("omni.anim.navigation.bundle")
# simulation_app.update()
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# print(f"ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")

import carb, os
settings = carb.settings.get_settings()
settings.set("/rtx/post/dlss/execMode", 1) # 0: Performance, 1: Balanced, 2: Quality, 3: Auto
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/translucency/enabled", True)
settings.set("/rtx-flags/ecoMode/enabled", True)



# Isaac Lab pretrained spot policy 
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

TASK = "Isaac-Velocity-Flat-Spot-v0"
RL_LIBRARY = "rsl_rl"

# Local imports
from utils.episode import VLNEpisode, save_episodes
from utils.sim import VLNSim
from utils.task_generator import TaskGenerator
import utils.navmesh_utils as navmesh_utils
from utils.vis import visualize_points, visualize_curve

# Main simulation loop

# load episodes
def load_task_config(args, scene_id):
    task_generator = TaskGenerator(args)
    task_config = task_generator.task_config
    scene_config = task_generator.get_scene_config(scene_id)
    current_episode = VLNEpisode(scene_config)
    return task_generator, task_config, scene_config, current_episode

task_generator, task_config, scene_config, current_episode = load_task_config(args, args.test_scene_id)

# setup environment
vln_sim = VLNSim(args)
vln_sim.reset(current_episode)

# sim variables
manager_env = vln_sim.manager_env
env = vln_sim.env
scene_folder = Path(args.scene_folder)

# setup navmesh tools
navmesh_interface = navmesh_utils.NavmeshInterface(up_axis='Z', stage=manager_env.scene.stage)

# disable socket server
args.disable_socket_server = True
print(f"[INFO] Socket server disabled in task generation")


# Setup UI
import utils.ui as sim_ui
import utils.ui_utils as ui_utils
import omni.usd
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.bounds as bounds_utils
import isaaclab.sim as sim_utils

ui_window = sim_ui.SimWindow(manager_env)
ui_elements = ui_window.ui_elements

with ui_elements["main_stack"]:
    with ui_window.create_frame("Scene Settings"):
        ui_elements["scene_id"] = ui_utils.dropdown_builder(
            "Scene Id",
            items=task_generator.scene_id_list,
            on_clicked_fn=lambda x: update_ui("scene_id", x)
        )
        ui_utils.btn_builder("Save Scene Settings", text="Save", on_clicked_fn=lambda: save_settings("scene"))
        ui_utils.btn_builder("ReLoad Task Config", text="ReLoad", on_clicked_fn=lambda: reload_config())
        ui_elements["usd_path"] = ui_utils.str_builder(
            "USD Path",
            use_folder_picker=True,
            default_val=args.scene_folder,
            folder_dialog_title="Select Scene USD",
            bookmark_label="Scene Folder",
            bookmark_path=args.scene_folder,
        )
        ui_elements["scene_scale"] = ui_utils.combo_floatfield_slider_builder(
            "Scene Scale",
            default_val=1.0,
            min=0.01,
            max=100,
            step=0.01,
        )[0]
        ui_elements["collider"] = ui_utils.cb_builder("Collider", default_val=True)
        ui_elements["align_ground"] = ui_utils.cb_builder("Align Ground", default_val=True)
        ui_utils.btn_builder("Load Scene", text="Load", on_clicked_fn=lambda: load_scene())
    with ui_window.create_frame("Navmesh Settings", collapsed=True):
        ui_elements["navmesh_preset"] = ui_utils.dropdown_builder(
            "Navmesh Preset",
            items=task_generator.navmesh_preset_list,
            on_clicked_fn=lambda x: update_ui("navmesh_preset", x)
        )
        for key, value in navmesh_interface.settings.items():
            key = navmesh_interface._camel_to_snake(key)
            ui_elements[f"navmesh_settings_{key}"] = ui_utils.combo_floatfield_slider_builder(
                key,
                default_val=value,
                min=0.01*value,
                max=100*value,
                step=0.001,
            )[0]
        ui_utils.btn_builder("Navmesh Config", text="Save", on_clicked_fn=lambda: save_settings("navmesh_config"))
    with ui_window.create_frame("Navmesh Tools"):
        ui_utils.btn_builder("Build Navmesh", text="Build", on_clicked_fn=lambda: build_navmesh())
        ui_utils.btn_builder("Load Navmesh", text="Load", on_clicked_fn=lambda: load_navmesh())
        ui_utils.btn_builder("Save Navmesh", text="Save", on_clicked_fn=lambda: save_navmesh())
        ui_utils.btn_builder("Test Navmesh", text="Test", on_clicked_fn=lambda: test_navmesh())
        ui_utils.btn_builder("Teleport Robot", text="Teleport", on_clicked_fn=lambda: teleport_robot())
        ui_utils.btn_builder("Generate Cube", text="Generate", on_clicked_fn=lambda: generate_cube())
        ui_utils.btn_builder("Clear Visualization", text="Clear", on_clicked_fn=lambda: clear_visualization())
    with ui_window.create_frame("Episode Settings"):
        # ui_elements["scene_type"] = ui_utils.str_builder("Scene Type", default_val=current_episode["scene_type"])
        # episode number
        ui_elements["episode_number"] = ui_utils.int_builder("Episode Number", default_val=30)
        ui_elements["rule_pattern"] = ui_utils.dropdown_builder(
            "Rule Pattern",
            items=task_generator.rule_pattern_list,
            on_clicked_fn=lambda x: save_settings("scene_runtime")
        )
        # ui_elements["goal_rules"] = ui_utils.str_builder("Episode Goals", default_val="mailbox, park_bench, hydrant")
        ui_utils.btn_builder("Generate Episode", text="Generate", on_clicked_fn=lambda: generate_episodes())

# ui_name: (config_name, [getter_func, setter_func])
ui_config_map = {
    "scene_id": ("scene_id", sim_ui.choice_func(task_generator.scene_id_list)),
    "usd_path": ("path", [
        lambda x: x.as_string.replace(args.scene_folder+"/", "").replace(args.scene_folder, ""),
        lambda x, y: x.set_value(str(scene_folder / y))
    ]),
    "navmesh_preset": ("navmesh_preset", sim_ui.choice_func(task_generator.navmesh_preset_list)),
    "scene_scale": ("scene_scale", sim_ui.float_func),
    "collider": ("collider", sim_ui.bool_func),
    "align_ground": ("align_ground", sim_ui.bool_func),
    "episode_number": ("episode_number", sim_ui.int_func),
    "rule_pattern": ("rule_pattern", sim_ui.choice_func(task_generator.rule_pattern_list)),
    # "goal_rules": ("goal_rules", [
    #     lambda x: json.loads(x.as_string),
    #     lambda x, y: x.set_value(json.dumps(y))
    # ]),
}
navmesh_settings_map = {x: (x.replace("navmesh_settings_", ""), sim_ui.float_func) for x in ui_elements.keys() if x.startswith("navmesh_settings_")}

def reload_config():
    global task_generator, task_config, scene_config, current_episode
    previous_scene_id = current_episode["scene_id"]
    task_generator, task_config, scene_config, current_episode = load_task_config(args, previous_scene_id)
    update_ui("scene_id", previous_scene_id)
    update_ui("navmesh_preset", scene_config["navmesh_preset"])
    print("[INFO] Task config reloaded")

def update_ui(settings_type, selected_value):
    global scene_config
    if settings_type == "navmesh_preset":
        preset_name = selected_value
        for key, (value, _) in navmesh_settings_map.items():
            ui_window.set_ui_value(navmesh_settings_map, key, task_config['navmesh'][preset_name][value])
    elif settings_type == "scene_id":
        scene_id = selected_value
        scene_config = task_generator.get_scene_config(scene_id)
        current_episode = VLNEpisode(scene_config)
        for key, (value, _) in ui_config_map.items():
            ui_window.set_ui_value(ui_config_map, key, scene_config[value])
        print("[INFO] Loaded goal rules:", scene_config["goal_rules"])
        print("[INFO] Loaded excluded paths:", scene_config["navmesh_exclude"])


def save_settings(settings_type):
    if settings_type == "navmesh_runtime":
        for key, (value, _) in navmesh_settings_map.items():
            value_camel = navmesh_interface._snake_to_camel(value).replace("agent", "")
            navmesh_interface.settings[value_camel] = ui_window.get_ui_value(navmesh_settings_map, key)
    elif settings_type == "navmesh_config":
        preset_name = ui_window.get_ui_value(ui_config_map, "navmesh_preset")
        for key, (value, _) in navmesh_settings_map.items():
            task_config['navmesh'][preset_name][value] = ui_window.get_ui_value(navmesh_settings_map, key)
        task_generator.save_config()
    elif settings_type == "scene":
        for key, (value, _) in ui_config_map.items():
            scene_config[value] = ui_window.get_ui_value(ui_config_map, key)
            task_generator.update_config(scene_config)
        task_generator.save_config()
    elif settings_type == "scene_runtime":
        for key, (value, _) in ui_config_map.items():
            current_episode[value] = ui_window.get_ui_value(ui_config_map, key)

def load_scene():
    save_settings("scene_runtime")
    env.reset(current_episode)

def build_navmesh():
    save_settings("navmesh_runtime")
    selected_paths = ["/World/ground/terrain"]
    start_time = time.time()
    navmesh_interface.setup_navmesh(selected_paths, scene_config.get("navmesh_exclude", []))
    navmesh_interface.build_navmesh()
    end_time = time.time()
    print(f"[INFO]: Navmesh build time: {end_time - start_time:.2f} seconds")
    test_navmesh()

def load_navmesh():
    navmesh_path = str(scene_folder / f"navmesh/{current_episode['scene_id']}_navmesh.bin")
    navmesh_interface.load_navmesh(navmesh_path)
    test_navmesh()

def test_navmesh():
    navmesh_interface.visualize_navmesh()
    points = navmesh_interface.sample_random_points(1000)
    if points is not None:
        visualize_points(points, prim_path="/World/RandomPoints", width=0.8)
        for i in range(50):
            path = navmesh_interface.find_paths(points[2*i], points[2*i+1])
            visualize_curve(path, prim_path=f"/World/Path_{i}", width=0.4)

def save_navmesh():
    os.makedirs(scene_folder / "navmesh", exist_ok=True)
    navmesh_path = str(scene_folder / f"navmesh/{current_episode['scene_id']}_navmesh.bin")
    navmesh_interface.save_navmesh(navmesh_path)

def teleport_robot():
    # current_episode["start_position"] = navmesh_interface.sample_random_points(1)[0]
    # env.reset(current_episode)
    robot_root_state = manager_env.scene["robot"].data.default_root_state.clone()
    random_pos = navmesh_interface.sample_random_points(robot_root_state.shape[0])
    random_pos[:, 2] += 0.6
    robot_root_state[:, 0:3] = torch.tensor(random_pos, device=args.device)
    manager_env.scene["robot"].write_root_state_to_sim(robot_root_state)
    manager_env.scene.reset()

def generate_cube():
    prim_selection = omni.usd.get_context().get_selection()
    selected_prim_paths = prim_selection.get_selected_prim_paths()
    # create a cube for first selected prim
    if len(selected_prim_paths) > 0:
        prim_path = selected_prim_paths[0]
        cube_path = "/World/Cube"
        # remove cube if it exists
        if prim_utils.get_prim_at_path(cube_path):
            prim_utils.delete_prim(cube_path)
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        bb_cache = bounds_utils.create_bbox_cache()
        min_x, min_y, min_z, max_x, max_y, max_z = bounds_utils.compute_combined_aabb(bb_cache, prim_paths=[prim_path])
        cfg_cube = sim_utils.CuboidCfg(size=[1.0, 1.0, 1.0])
        position = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]
        cfg_cube.func("/World/Cube", cfg_cube, translation=list(position))
    else:
        print("[ERROR]: No prim selected")

def remove_prim(rule):
    prim_list = prim_utils.find_matching_prim_paths(rule)
    for prim_path in prim_list:
        prim_utils.delete_prim(prim_path)

def clear_visualization():
    remove_prim("/World/RandomPoints")
    remove_prim("/World/Path_*")
    remove_prim("/World/ground/navmeshmesh")

def generate_episodes():
    save_settings("scene")
    save_settings("scene_runtime")
    save_settings("navmesh_config")
    save_settings("navmesh_runtime")
    episodes = task_generator.generate_episodes(env, current_episode["scene_id"], navmesh_interface)
    print(task_generator.scene_config)
    save_episodes(episodes, f"episodes/{current_episode['scene_id']}.json")

update_ui("scene_id", current_episode["scene_id"])

"""Main simulation loop"""
print("[INFO]: Starting simulation")
while simulation_app.is_running():
    with torch.inference_mode():
        vln_sim.step()

simulation_app.close()