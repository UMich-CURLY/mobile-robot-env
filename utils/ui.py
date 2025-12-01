import asyncio
import numpy as np

import omni.kit.app
import omni.ui as ui
from isaacsim.gui.components import ui_utils
from isaacsim.gui.components.style import get_style
import isaacsim.core.utils.prims as prim_utils

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
            getter_func, setter_func = map[key][1]
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
    def __init__(self, vln_sim, default_episode_label="none"):
        super().__init__(vln_sim)
        # set ref to vln_sim
        self.vln_sim = vln_sim
        self.episode_label_list = vln_sim.episode_label_list
        self.episode_list = vln_sim.episode_list
        self.manager_env = vln_sim.manager_env
        self.env = vln_sim.env
        self.ui_episode = None
        # map ui_episode fields to ui_elements
        self.ui_map = {
            "episode_label": ("episode_label", choice_func(self.episode_label_list)),
            "instruction": ("instruction", str_func),
            "episode_info": ("episode_info", label_func),
            "start_position": ("start_position", xyz_func),
        }
        # initialize ui
        self.build_ui()
        if default_episode_label=="none":
            default_episode_label = self.episode_label_list[0]
        self.update_ui("episode_label", default_episode_label)

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
                ui_utils.btn_builder("Apply Settings", text="Update", on_clicked_fn=lambda: self.save_settings("episode_runtime"))
                ui_utils.btn_builder("Follow Reference Path", text="Start", on_clicked_fn=lambda: self.start_following_waypoints())
                ui_utils.btn_builder("Stop Following", text="Stop", on_clicked_fn=lambda: self.stop_following_waypoints())
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
            print(f"[INFO] UI is updating to episode {selected_value}")
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
            # self.update_ui("episode_info")
            self.vln_sim.reset(self.ui_episode)

    def start_following_waypoints(self):
        goal_positions = np.array([x["location"] for x in self.ui_episode["goals"]])
        dist_to_goals = np.linalg.norm(goal_positions - self.ui_episode["start_position"], axis=1)
        closest_goal = self.ui_episode["goals"][np.argmin(dist_to_goals)]
        ref_path = closest_goal["reference_path"]
        self.vln_sim.set_waypoints(ref_path[1:], fix_yaw=True)

    def stop_following_waypoints(self):
        self.vln_sim.clear_waypoints()
        self.remove_prim("/World/WaypointPath")
        self.remove_prim("/World/Arrow")

    def remove_prim(self, rule):
        prim_list = prim_utils.find_matching_prim_paths(rule)
        for prim_path in prim_list:
            prim_utils.delete_prim(prim_path)