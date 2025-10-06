import asyncio
import os
import weakref
from datetime import datetime
from typing import TYPE_CHECKING

import isaacsim
import omni.kit.app
import omni.ui as ui
from isaacsim.gui.components import ui_utils
from isaacsim.gui.components.style import get_style

LABEL_WIDTH = 120
SPACING = 4

class SimWindow:

    def __init__(self, env, window_name = "In-N-Out Settings"):
        self.env = env

        # Listeners for environment selection changes
        self.values = {}

        # create window for UI
        self.ui_window = ui.Window(
            window_name, width=300, height=500, visible=True, dock_preference=ui.DockPreference.RIGHT_TOP
        )
        # dock next to properties window
        asyncio.ensure_future(self._dock_window(window_title=self.ui_window.title))

        # keep a dictionary of stacks so that child environments can add their own UI elements
        # this can be done by using the `with` context manager
        self.ui_elements = dict()
        # create main frame
        self.ui_elements["main_frame"] = self.ui_window.frame
        with self.ui_elements["main_frame"]:
            self.ui_elements["main_stack"] = ui.VStack(style=get_style(), spacing=5, height=0)
    

    def get_ui_value(self, map, key):
        try:
            getter_func, setter_func = map[key][1]
            print(f"[INFO]: Get {key} value: {getter_func(self.ui_elements[key])}")
            return getter_func(self.ui_elements[key])
        except:
            print(f"[ERROR]: Failed to get value for {key}")
            import traceback
            traceback.print_exc()

    def set_ui_value(self, map, key, value):
        try:
            getter_func, setter_func = map[key][1]
            setter_func(self.ui_elements[key], value)
            print(f"[INFO]: Set {key} value: {value}")
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
        if self.ui_window is not None:
            self.ui_window.visible = False
            self.ui_window.destroy()
            self.ui_window = None

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