from utils.server import run_server, format_data
import torch
import numpy as np
import carb
from threading import Thread
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name

class InNOutSim:
    def __init__(self, args, env):
        self.args = args
        self.device = args.device
        self.env = env
        self.manager_env = env.unwrapped.unwrapped
        
        # Shared buffers for server callbacks
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_position = None
        self._latest_quat_wxyz = None

        # simulation
        self.commands = torch.tensor([[0.0, 0.0, 0.0] for _ in range(args.num_envs)], device=self.device)
        self.commands_source = 'server'
        self.robot_index = 0
        self.viewport = get_viewport_from_window_name("Viewport")
        if self.args.headless:
            print("[INFO] Headless mode, disabling viewport updates")
            self.viewport.updates_enabled = False
        self.viewport_third_person = True
        self.update_viewer()

        # initialize keyboard and server
        self.set_up_keyboard()
        self.start_server()

        # generate tasks
        self.traverse_objects()
    
    def traverse_objects(self):
        prim_list = [x for x in self.manager_env.scene.stage.Traverse()]
    
    def load_scene(self, usd_path):
        manager_env = self.manager_env
        manager_env.stage.RemovePrim(manager_env.scene.terrain.terrain_prim_paths[0])
        manager_env.scene.terrain.terrain_prim_paths = []
        manager_env.scene.terrain.import_usd("terrain", usd_path)

    # Socket server integration (control Spot via external commands)
    def action_callback(self, msg_type, message):
        # Expect messages of type 'VEL' with fields x, y, omega
        if msg_type == 'VEL':
            self.commands[self.robot_index] = torch.tensor([float(message.x), float(message.y), float(message.omega)], device=self.device)
            self.commands_source = 'server_vel'
        elif msg_type == 'WAYPOINTS':
            print("ERROR: Waypoints not supported yet")

    def data_callback(self):
        if self._latest_rgb is None or self._latest_depth is None or self._latest_position is None or self._latest_quat_wxyz is None:
            print("Data missing")
            return None
        return format_data(self._latest_rgb, self._latest_depth, self._latest_position, self._latest_quat_wxyz)

    def planner_callback(self):
        # No onboard planner state here
        return {}
    
    def update_viewer(self):
        # set viewport
        if self.viewport_third_person:
            camera_path = f"/World/envs/env_{self.robot_index}/Robot/body/ThirdPersonCamera"
        else:
            camera_path = f"/World/envs/env_{self.robot_index}/Robot/body/Camera"
        self.viewport.set_active_camera(camera_path)

    def start_server(self):
        # Start socket server in background
        server_thread = Thread(target=run_server, kwargs={
            "data_cb": self.data_callback, 
            "action_cb": self.action_callback, 
            "planner_cb": self.planner_callback
        })
        server_thread.daemon = True
        server_thread.start()
        print("[INFO] Socket server started")

    _obs_index = 0
    def update_obs(self, obs, manager_env):
        # np.save(f"results/obs_{self._obs_index}.npy", obs.cpu().numpy())
        # self._obs_index = self._obs_index%100+1
        # only publish the first robot's obs for now
        try:
            self._latest_rgb = obs[0, :, :, :3].cpu().numpy().astype(np.uint8)
            depth = obs[0, :, :, 3].cpu().numpy()
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
            depth = np.clip(depth, 0, 65535).astype(np.uint16)
            self._latest_depth = depth
            self._latest_position = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy().astype(np.float32)
            self._latest_quat_wxyz = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error updating obs: {e}")
            import traceback
            traceback.print_exc()

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 2
        R = 2
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([-T, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([0, 0.0, R], device=self.device),
            "RIGHT": torch.tensor([0, 0.0, -R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            if event.input.name in self._key_to_control:
                print("keyboard event: ", event.input.name)
                self.commands[self.robot_index] = self._key_to_control[event.input.name]
                self.commands_source = 'keyboard'
            if event.input.name in ['KEY_'+str(i) for i in range(min(self.env.unwrapped.num_envs, 10))]:
                self.robot_index = int(event.input.name[-1])
                self.update_viewer()
            if event.input.name == 'C':
                self.viewport_third_person = not self.viewport_third_person
                self.update_viewer()
        # On key release, the robot stops moving
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.commands[self.robot_index] = self._key_to_control["ZEROS"]
