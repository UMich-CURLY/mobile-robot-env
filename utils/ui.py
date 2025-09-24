# check https://github.com/isaac-sim/IsaacSim/blob/main/source/extensions/isaacsim.gui.components/isaacsim/gui/components/ui_utils.py
import asyncio
import os
import weakref
from datetime import datetime
from typing import TYPE_CHECKING

import isaacsim
import omni.kit.app

import omni.ui as ui

LABEL_WIDTH = 120
SPACING = 4

class SimWindow:

    def __init__(self, env, window_name = "In-N-Out Settings"):
        self.env = env

        # Listeners for environment selection changes
        self.values = {}

        # create window for UI
        self.ui_window = ui.Window(
            window_name, width=400, height=500, visible=True, dock_preference=ui.DockPreference.RIGHT_TOP
        )
        # dock next to properties window
        asyncio.ensure_future(self._dock_window(window_title=self.ui_window.title))

        # keep a dictionary of stacks so that child environments can add their own UI elements
        # this can be done by using the `with` context manager
        self.ui_elements = dict()
        # create main frame
        self.ui_elements["main_frame"] = self.ui_window.frame
    
    def create_group(self, name):
        with ui.CollapsableFrame():
            

    def __del__(self):
        """Destructor for the window."""
        # destroy the window
        if self.ui_window is not None:
            self.ui_window.visible = False
            self.ui_window.destroy()
            self.ui_window = None

    def create_float_drag(self, name, min, max, step, default_value):
        with ui.HStack():
            ui.Label(name, width=LABEL_WIDTH)
            def on_value_changed(model):
                self.values[name] = model.as_float
                print(f"[{name}] value: {model.as_float}")
            self.ui_elements[name] = ui.FloatDrag(name=name, min=min, max=max, step=step)
            self.ui_elements[name].model.set_value(default_value)
            self.ui_elements[name].model.add_value_changed_fn(on_value_changed)
            self.values[name] = default_value
            isaacsim.gui.components.ui_utils.add_line_rect_flourish()
    
    def create_button(self, name, text, clicked_fn: callable):
        with ui.HStack():
            ui.Label(name, width=LABEL_WIDTH)
            self.ui_elements[name] = ui.Button(
                name=name,
                text=text,
                clicked_fn=clicked_fn,
            )
            isaacsim.gui.components.ui_utils.add_line_rect_flourish()

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
