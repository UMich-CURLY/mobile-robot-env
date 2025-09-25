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
