import asyncio
from pathlib import Path
import random
import time
import json
import os
import numpy as np
import torch

import omni.kit.app
import omni.ui as ui
from isaacsim.gui.components import ui_utils
from isaacsim.gui.components.style import get_style
import isaacsim.core.utils.prims as prim_utils
import omni.usd
import isaacsim.core.utils.bounds as bounds_utils
import isaaclab.sim as sim_utils
from utils.episode import VLNEpisode
from utils.vis import visualize_points, visualize_curve

LABEL_WIDTH = 120
SPACING = 4

str_func = [lambda x: x.as_string, lambda x, y: x.set_value(y)]
int_func = [lambda x: x.as_int, lambda x, y: x.set_value(y)]
float_func = [lambda x: x.as_float, lambda x, y: x.set_value(y)]
bool_func = [lambda x: x.as_bool, lambda x, y: x.set_value(y)]
choice_func = lambda item_list: [
    lambda x: item_list[x.get_item_value_model().as_int],
    lambda x, y: x.get_item_value_model().set_value(item_list.index(y))
]
xyz_func = [lambda x: [x[i].as_float for i in range(3)], lambda x, y: [x[i].set_value(y[i]) for i in range(3)]]

def set_text(x, y):
    x.text = y
label_func = [lambda x: x.text, lambda x, y: set_text(x, y)]

class BaseUI:

    def __init__(self, vln_sim, window_name = "In-N-Out Settings"):
        self.env = vln_sim.manager_env

        # create window for UI
        self.self = ui.Window(
            window_name, width=300, height=500, visible=True, dock_preference=ui.DockPreference.RIGHT_TOP
        )
        # dock next to properties window
        asyncio.ensure_future(self._dock_window(window_title=self.self.title))

        # keep a dictionary of stacks so that child environments can add their own UI elements
        # this can be done by using the `with` context manager
        self.ui_elements = dict()
        # create main frame
        self.ui_elements["main_frame"] = self.self.frame
        with self.ui_elements["main_frame"]:
            self.ui_elements["main_stack"] = ui.VStack(style=get_style(), spacing=5, height=0)
    

    def get_ui_value(self, map, key):
        try:
            getter_func, setter_func = map[key][1]
            # print(f"[INFO]: Get {key} value: {getter_func(self.ui_elements[key])}")
            return getter_func(self.ui_elements[key])
        except:
            print(f"[ERROR]: Failed to get value for {key}")
            import traceback
            traceback.print_exc()

    def set_ui_value(self, map, key, value):
        try:
            getter_func,  setter_func = map[key][1]
            setter_func(self.ui_elements[key], value)
            # print(f"[INFO]: Set {key} value: {value}")
        except:
            print(f"[ERROR]: Failed to set value for {key}")
            import traceback
            traceback.print_exc()

    def create_frame(self, name, collapsed=False):
        with ui.CollapsableFrame(name, collapsed=collapsed):
            return ui.VStack(style=get_style(), spacing=5, height=0)
            

    def __del__(self):
        """Destructor for the window."""
        # destroy the window
        if self.self is not None:
            self.self.visible = False
            self.self.destroy()
            self.self = None

    async def _dock_window(self, window_title):
        """Docks the custom UI window to the property window."""
        # wait for the window to be created
        for _ in range(5):
            if ui.Workspace.get_window(window_title):
                break
            await self.env.sim.app.next_update_async()

        # dock next to properties window
        custom_window = ui.Workspace.get_window(window_title)
        property_window = ui.Workspace.get_window("Property")
        if custom_window and property_window:
            custom_window.dock_in(property_window, ui.DockPosition.SAME, 1.0)
            custom_window.focus()

class BenchmarkUI(BaseUI):
    def __init__(self, vln_sim):
        super().__init__(vln_sim)
        # set ref to vln_sim
        self.vln_sim = vln_sim
        self.episode_label_list = vln_sim.episode_label_list
        self.episode_list = vln_sim.episode_list
        self.manager_env = vln_sim.manager_env
        self.env = vln_sim.env
        self.ui_episode = None
        # initialize ui
        self.build_ui()
        # map ui_episode fields to ui_elements
        # format: {ui_name: (config_name, [getter_func, setter_func])}
        self.ui_map = {
            "episode_label": ("episode_label", choice_func(self.episode_label_list)),
            "instruction": ("instruction", str_func),
            "episode_info": ("episode_info", label_func),
            "start_position": ("start_position", xyz_func),
        }
        # add callback to sim
        self.vln_sim.add_callback('client_episode_changed', lambda episode_label: self.update_ui("episode_label", episode_label))

    def build_ui(self):
        with self.ui_elements["main_stack"]:
            with self.create_frame("Debug Tools"):
                self.ui_elements["episode_label"] = ui_utils.dropdown_builder(
                    "Episode Id",
                    items=self.episode_label_list,
                    on_clicked_fn=lambda x: self.update_ui("episode_label", x)
                )
                self.ui_elements["instruction"] = ui_utils.str_builder(
                    "Instruction",
                    default_val="instruction",
                )
                self.ui_elements["start_position"] = ui_utils.xyz_builder(
                    "Start Position",
                    default_val=[0.0, 0.0, 0.0]
                )
                ui_utils.btn_builder("Load Scene", text="Load", on_clicked_fn=lambda: self.save_settings("episode_runtime"))
                ui_utils.btn_builder("Follow Reference Path", text="Start", on_clicked_fn=lambda: self.vln_sim.set_ref_waypoints(self.ui_episode))
                ui_utils.btn_builder("Stop Following", text="Stop", on_clicked_fn=lambda: self.vln_sim.clear_waypoints())
                ui_utils.btn_builder("Switch StopCalled State", text="Switch", on_clicked_fn=lambda: self.env.set_stop_called(self.vln_sim.robot_index, not self.env.is_stop_called[self.vln_sim.robot_index]))
            with self.create_frame("Episode Info"):
                self.ui_elements["episode_info"] = ui_utils.ui.Label(
                    "Episode Info",
                    style_type_name_override="Label::label",
                    word_wrap=True,
                    alignment=ui_utils.ui.Alignment.LEFT_TOP,
                )

    def update_ui(self, settings_type, selected_value=None):
        if settings_type == "episode_label":
            if self.ui_episode is not None and selected_value==self.ui_episode["episode_label"]:
                return
            print(f"[UI] UI is updating to episode {selected_value}")
            self.ui_episode = self.vln_sim.episode_list[self.vln_sim.episode_label_list.index(selected_value)]
            for key, (value, _) in self.ui_map.items():
                self.set_ui_value(self.ui_map, key, self.ui_episode[value])
        # elif settings_type == "episode_info":
        #     self.set_ui_value(self.ui_map, "episode_info", self.ui_episode.episode_info)


    def save_settings(self, settings_type):
        episode_label = self.get_ui_value(self.ui_map, "episode_label")
        self.ui_episode = self.vln_sim.episode_list[self.episode_label_list.index(episode_label)].copy()
        if settings_type == "episode_runtime":
            for key, (value, _) in self.ui_map.items():
                if key in ["episode_info", "episode_label"]:
                    continue
                self.ui_episode[value] = self.get_ui_value(self.ui_map, key)
            self.vln_sim.reset(self.ui_episode)

class TaskGeneratorUI(BaseUI):
    def __init__(self, vln_sim, task_generator):
        super().__init__(vln_sim)
        self.vln_sim = vln_sim
        self.task_generator = task_generator
        self.args = vln_sim.args
        self.episode_label_list = vln_sim.episode_label_list
        self.episode_list = vln_sim.episode_list
        self.manager_env = vln_sim.manager_env
        self.env = vln_sim.env
        self.ui_episode = None
        self.scene_config = None
        self.scene_folder = Path(self.args.scene_folder)
        # initialize ui
        self.build_ui()
        # map ui_episode fields to ui_elements
        # format: {ui_name: (config_name, [getter_func, setter_func])}
        self.ui_config_map = {
            "scene_id": ("scene_id", choice_func(task_generator.scene_id_list)),
            "usd_path": ("path", [
                lambda x: x.as_string.replace(self.args.scene_folder+"/", "").replace(self.args.scene_folder, ""),
                lambda x, y: x.set_value(str(self.scene_folder / y))
            ]),
            "navmesh_preset": ("navmesh_preset", choice_func(task_generator.navmesh_preset_list)),
            "scene_scale": ("scene_scale", float_func),
            "ceiling_height": ("ceiling_height", float_func),
            "collider": ("collider", bool_func),
            "align_ground": ("align_ground", bool_func),
            "episode_number": ("episode_number", int_func),
            "rule_pattern": ("rule_pattern", choice_func(task_generator.rule_pattern_list)),
        }
        navmesh_settings_keys = [self._camel_to_snake(x) for x in self.task_generator.navmesh_interface.settings.keys()]
        self.navmesh_settings_map = {f"navmesh_settings_{x}": (x, float_func) for x in navmesh_settings_keys}
        # add callback to sim
        def episode_changed_callback(x):
            self.update_ui("scene_id", self.vln_sim.next_episode.scene_id)
            self.update_ui("navmesh_preset", self.vln_sim.next_episode["navmesh_preset"])
        self.vln_sim.add_callback('client_episode_changed', episode_changed_callback)

    def build_ui(self):
        with self.ui_elements["main_stack"]:
            with self.create_frame("Scene Settings"):
                self.ui_elements["scene_id"] = ui_utils.dropdown_builder(
                    "Scene Id",
                    items=self.task_generator.scene_id_list,
                    on_clicked_fn=lambda x: self.update_ui("scene_id", x)
                )
                ui_utils.btn_builder("Load Scene", text="Load", on_clicked_fn=lambda: self.load_scene())
                ui_utils.btn_builder("Save Scene Settings", text="Save", on_clicked_fn=lambda: self.save_settings("scene"))
                ui_utils.btn_builder("ReLoad Task Config", text="ReLoad", on_clicked_fn=lambda: self.reload_config())
                self.ui_elements["usd_path"] = ui_utils.str_builder(
                    "USD Path",
                    use_folder_picker=True,
                    default_val=self.args.scene_folder,
                    folder_dialog_title="Select Scene USD",
                    bookmark_label="Scene Folder",
                    bookmark_path=self.args.scene_folder,
                )
                self.ui_elements["scene_scale"] = ui_utils.combo_floatfield_slider_builder(
                    "Scene Scale",
                    default_val=1.0,
                    min=0.01,
                    max=100,
                    step=0.01,
                )[0]
                self.ui_elements["ceiling_height"] = ui_utils.combo_floatfield_slider_builder(
                    "Ceiling Clip Range",
                    default_val=1.0,
                    min=0,
                    max=100,
                    step=0.1,
                )[0]
                self.ui_elements["ceiling_height"].add_value_changed_fn(lambda x: self.task_generator.update_bev_camera_clip(self.ui_episode["scene_id"], "ceiling", x.as_float))
                self.ui_elements["collider"] = ui_utils.cb_builder("Collider", default_val=True)
                self.ui_elements["align_ground"] = ui_utils.cb_builder("Align Ground", default_val=True)
            with self.create_frame("Navmesh Settings", collapsed=True):
                self.ui_elements["navmesh_preset"] = ui_utils.dropdown_builder(
                    "Navmesh Preset",
                    items=self.task_generator.navmesh_preset_list,
                    on_clicked_fn=lambda x: self.update_ui("navmesh_preset", x)
                )
                for key, value in self.task_generator.navmesh_interface.settings.items():
                    key = self._camel_to_snake(key)
                    self.ui_elements[f"navmesh_settings_{key}"] = ui_utils.combo_floatfield_slider_builder(
                        key,
                        default_val=value,
                        min=0.01*value,
                        max=100*value,
                        step=0.001,
                    )[0]
                ui_utils.btn_builder("Navmesh Config", text="Save", on_clicked_fn=lambda: self.save_settings("navmesh_config"))
            with self.create_frame("Navmesh Tools"):
                ui_utils.btn_builder("Setup Geometry", text="Setup", on_clicked_fn=lambda: self.setup_navmesh())
                ui_utils.btn_builder("Build Navmesh", text="Build", on_clicked_fn=lambda: self.build_navmesh())
                ui_utils.btn_builder("Load Navmesh", text="Load", on_clicked_fn=lambda: self.load_navmesh())
                ui_utils.btn_builder("Save Navmesh", text="Save", on_clicked_fn=lambda: self.save_navmesh())
                ui_utils.btn_builder("Test Navmesh", text="Test", on_clicked_fn=lambda: self.test_navmesh())
                ui_utils.btn_builder("Teleport Robot", text="Teleport", on_clicked_fn=lambda: self.teleport_robot())
                ui_utils.btn_builder("Generate Cube", text="Generate", on_clicked_fn=lambda: self.generate_cube())
                ui_utils.btn_builder("Clear Visualization", text="Clear", on_clicked_fn=lambda: self.clear_visualization())
            with self.create_frame("Episode Settings"):
                # episode number
                self.ui_elements["episode_number"] = ui_utils.int_builder("Episode Number", default_val=30)
                self.ui_elements["rule_pattern"] = ui_utils.dropdown_builder(
                    "Rule Pattern",
                    items=self.task_generator.rule_pattern_list,
                    on_clicked_fn=lambda x: self.save_settings("scene_runtime")
                )
                ui_utils.btn_builder("Generate Episode", text="Generate", on_clicked_fn=lambda: self.generate_episodes())
                ui_utils.btn_builder("Stop Generation", text="Stop", on_clicked_fn=lambda: self.stop_generation())
                ui_utils.btn_builder("Get Information", text="Get", on_clicked_fn=lambda: self.get_information())
                ui_utils.btn_builder("Toggle Ceiling", text="Toggle", on_clicked_fn=lambda: self.task_generator.toggle_ceiling(self.ui_episode["scene_id"]))
                ui_utils.btn_builder("Create BEV Map", text="Create", on_clicked_fn=lambda: self.task_generator.create_bev_map(self.ui_episode["scene_id"], clip_range="ceiling"))
                ui_utils.btn_builder("Save Occupancy Map", text="Save", on_clicked_fn=lambda: self.task_generator.create_bev_map(self.ui_episode["scene_id"], clip_range="robot", file_name="height_map"))
            with self.create_frame("Info"):
                self.ui_elements["task_config"] = ui_utils.ui.Label(
                    "Task Config",
                    style_type_name_override="Label::label",
                    word_wrap=True,
                    alignment=ui_utils.ui.Alignment.LEFT_TOP,
                )
    def _snake_to_camel(self, s):
        parts = s.split('_')
        return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])
    
    def _camel_to_snake(self, s):
        return ''.join(['_' + char.lower() if char.isupper() else char for char in s]).lstrip('_')

    def reload_config(self):
        scene_id = self.ui_episode["scene_id"]
        self.episode_ui = self.vln_sim.episode_list[self.episode_label_list.index(scene_id)]
        self.update_ui("scene_id", scene_id)
        self.update_ui("navmesh_preset", self.ui_episode["navmesh_preset"])
        print("[UI] Task config reloaded")

    def update_ui(self, settings_type, selected_value):
        if settings_type == "navmesh_preset":
            preset_name = selected_value
            print(f"[UI] Update navmesh_preset to {preset_name}")
            for key, (value, _) in self.navmesh_settings_map.items():
                self.set_ui_value(self.navmesh_settings_map, key, self.task_generator.task_config['navmesh'][preset_name][value])
        elif settings_type == "scene_id":
            scene_id = selected_value
            self.scene_config = self.task_generator.get_scene_config(scene_id)
            self.ui_episode = VLNEpisode(self.scene_config)
            for key, (value, _) in self.ui_config_map.items():
                self.set_ui_value(self.ui_config_map, key, self.scene_config[value])
            print("[UI] Loaded goal rules:", self.scene_config.get("goal_rules", {}))
            print("[UI] Loaded excluded paths:", self.scene_config["navmesh_exclude"])


    def save_settings(self, settings_type):
        if settings_type == "navmesh_runtime":
            for key, (value, _) in self.navmesh_settings_map.items():
                value_camel = self._snake_to_camel(value)
                self.task_generator.navmesh_interface.settings[value_camel] = self.get_ui_value(self.navmesh_settings_map, key)
        elif settings_type == "navmesh_config":
            preset_name = self.get_ui_value(self.ui_config_map, "navmesh_preset")
            for key, (value, _) in self.navmesh_settings_map.items():
                self.task_generator.task_config['navmesh'][preset_name][value] = self.get_ui_value(self.navmesh_settings_map, key)
            self.task_generator.save_config()
        elif settings_type == "scene":
            for key, (value, _) in self.ui_config_map.items():
                self.scene_config[value] = self.get_ui_value(self.ui_config_map, key)
                self.task_generator.update_config(self.scene_config)
            self.task_generator.save_config()
        elif settings_type == "scene_runtime":
            for key, (value, _) in self.ui_config_map.items():
                self.ui_episode[value] = self.get_ui_value(self.ui_config_map, key)

    def load_scene(self):
        self.save_settings("scene_runtime")
        self.env.reset(self.ui_episode)
    
    def setup_navmesh(self):
        self.save_settings("scene_runtime")
        selected_paths = ["/World/ground/terrain"]
        start_time = time.time()
        self.task_generator.navmesh_interface.setup_navmesh(selected_paths, self.scene_config.get("navmesh_exclude", []), self.manager_env.scene.stage, scene_type=self.scene_config.get("scene_type"))
        print(f"[INFO]: Navmesh geometry setup time: {time.time() - start_time:.2f} seconds")

    def build_navmesh(self):
        self.save_settings("navmesh_runtime")
        start_time = time.time()
        self.task_generator.navmesh_interface.build_navmesh()
        print(f"[INFO]: Navmesh build time: {time.time() - start_time:.2f} seconds")

    def load_navmesh(self):
        navmesh_path = str(self.scene_folder / f"navmesh/{self.ui_episode['scene_id']}_navmesh.bin")
        self.task_generator.navmesh_interface.load_navmesh(navmesh_path)
        self.test_navmesh()
        self.teleport_robot()

    def test_navmesh(self):
        self.task_generator.navmesh_interface.visualize_navmesh()
        points = self.task_generator.navmesh_interface.sample_random_points(1000)
        if points is not None:
            visualize_points(points, prim_path="/World/RandomPoints", width=0.2)
            for i in range(50):
                path = self.task_generator.navmesh_interface.find_paths(points[2*i], points[2*i+1])
                visualize_curve(path, prim_path=f"/World/Path_{i}", width=0.1)
        self.teleport_robot()

    def save_navmesh(self):
        os.makedirs(self.scene_folder / "navmesh", exist_ok=True)
        navmesh_path = str(self.scene_folder / f"navmesh/{self.ui_episode['scene_id']}_navmesh.bin")
        self.task_generator.navmesh_interface.save_navmesh(navmesh_path)

    def teleport_robot(self):
        # self.ui_episode["start_position"] = navmesh_interface.sample_random_points(1)[0]
        # env.reset(self.ui_episode)
        with torch.inference_mode(): 
            robot_root_state = self.manager_env.scene["robot"].data.default_root_state.clone()
            random_pos = self.task_generator.navmesh_interface.sample_random_points(robot_root_state.shape[0])
            random_pos[:, 2] += 0.6
            robot_root_state[:, 0:3] = torch.tensor(random_pos, device=self.args.device)
            self.manager_env.scene["robot"].write_root_state_to_sim(robot_root_state)
            self.manager_env.scene.reset()

    def generate_cube(self):
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
            cfg_cube.func(cube_path, cfg_cube, translation=list(position))
        else:
            print("[ERROR]: No prim selected")

    def clear_visualization(self):
        self.vln_sim.env.remove_prim("/World/RandomPoints")
        self.vln_sim.env.remove_prim("/World/Path_*")
        self.vln_sim.env.remove_prim("/World/ground/navmeshmesh")

    def generate_episodes(self):
        self.save_settings("scene")
        self.save_settings("scene_runtime")
        self.save_settings("navmesh_config")
        self.save_settings("navmesh_runtime")
        self.task_generator.generate_episodes(self.ui_episode["scene_id"])
    
    def stop_generation(self):
        self.task_generator.stop_generation()

    def get_information(self):
        # Get current robot world pose
        pos = self.manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy()
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        print(f"[UI] Scene type: {self.scene_config['scene_type']}")

        # Build base directory by combining scene_folder and path (drop filename)
        json_path = str(self.scene_folder / os.path.dirname(self.scene_config["path"])) if isinstance(self.scene_config.get("path"), str) else str(self.scene_folder)
        print(f"[UI] JSON path: {json_path}")
        try:
            from utils.vc_location_info_utils import CityDataReader
            reader = CityDataReader(json_path)
        except Exception as e:
            print(f"[ERROR] Failed to init CityDataReader: {e}")
            return

        # Query nearest road and nearby interesting points
        nearest_road = reader.get_nearest_road(x, y)
        nearby_points = reader.get_points_in_radius(x, y, radius=200.0)

        print("[UI] Current Position:", {"x": x, "y": y, "z": z})
        if nearest_road:
            rn = nearest_road.get("name", "unknown")
            rt = nearest_road.get("type", "unknown")
            dist = nearest_road.get("distance_to_road", None)
            cp = nearest_road.get("closest_point_on_road", None)
            print(f"[UI] Nearest road: name={rn}, type={rt}, distance={dist:.2f}m" if dist is not None else f"[UI] Nearest road: name={rn}, type={rt}")
            if cp is not None:
                print(f"[UI] Closest point on road: ({cp[0]:.2f}, {cp[1]:.2f})")
        else:
            print("[UI] No nearby road found")

        print(f"[UI] Nearby points within 200m: {len(nearby_points)}")
        for p in nearby_points[:5]:
            name = p.get("name", "-")
            ptype = p.get("type", "-")
            cat = p.get("category", "-")
            d = p.get("distance_from_center", None)
            if d is not None:
                print(f"  - {name} ({ptype}/{cat}), {d:.1f}m")
            else:
                print(f"  - {name} ({ptype}/{cat})")
        
        # --- Sample one episode and aggregate road/POI names along its trajectory ---
        episodes_file = f"episodes/{self.ui_episode['scene_id']}.json"
        if not os.path.exists(episodes_file):
            print(f"[UI] Episodes file not found: {episodes_file}; skipping trajectory info aggregation")
            return
        
        try:
            with open(episodes_file, 'r') as f:
                episodes = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load episodes from {episodes_file}: {e}")
            return
        
        if not episodes:
            print("[UI] No episodes in file; skipping trajectory info aggregation")
            return

        # Pick a random episode
        sampled_episode = random.choice(episodes)
        # Collect all reference path points from all goals
        trajectory_points = []
        for g in sampled_episode.get("goals", []):
            ref_path = g.get("reference_path", [])
            if isinstance(ref_path, list):
                trajectory_points.extend(ref_path)

        if not trajectory_points:
            print("[UI] Sampled episode has no reference_path; skipping info aggregation")
            return

        road_names = set()
        poi_names = set()

        for p in trajectory_points:
            if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                continue
            px, py = float(p[0]), float(p[1])
            # nearest road
            nearest_road = reader.get_nearest_road(px, py)
            if nearest_road:
                rn = nearest_road.get("name")
                if rn:
                    road_names.add(rn)
            # nearby POIs within 200m
            nearby_points = reader.get_points_in_radius(px, py, radius=200.0)
            for poi in nearby_points or []:
                name = poi.get("name")
                if name:
                    poi_names.add(name)

        print(f"[UI] Sampled episode id: {sampled_episode.get('episode_id')}")
        print(f"[UI] Roads along trajectory ({len(road_names)}): {sorted(road_names)}")
        print(f"[UI] POIs along trajectory ({len(poi_names)}): {sorted(poi_names)}")