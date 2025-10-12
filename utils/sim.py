from utils.socket_server import run_server, format_data
import torch
import numpy as np
import carb
from threading import Thread
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from utils.path_following_utils import visualize_path, follow_waypoints
from scipy.spatial.transform import Rotation as R

class VLNSim:
    def __init__(self, args, env):
        self.args = args
        self.device = args.device
        self.env = env
        self.manager_env = env.unwrapped.unwrapped
        
        # Shared buffers for server callbacks
        self._latest_data = {
            "rgb": None,
            "depth": None,
            "position": None,
            "quat_wxyz": None,
            "info": {}
        }

        # simulation
        self.waypoints = []
        self.waypoints_idx = 0
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
        if not args.disable_socket_server:
            self.start_server(host=args.socket_server_host, port=args.socket_server_port, server_name="BenchmarkServer")

    # Socket server integration (control Spot via external commands)
    def action_callback(self, msg_type, message):
        # Expect messages of type 'VEL' with fields x, y, omega
        if msg_type == 'VEL':
            vx, vy, vw = message["vx"], message["vy"], message["vw"]
            self.commands[self.robot_index] = torch.tensor([float(vx), float(vy), float(vw)], device=self.device)
            self.commands_source = 'server_vel'
        elif msg_type == 'WAYPOINT':
            x_list, y_list = message["x_list"], message["y_list"]
            waypoints = []
            for x, y in zip(x_list, y_list):
                waypoints.append([float(x), float(y), 0.0])
            self.set_waypoints(waypoints)
            self.commands_source = 'server_waypoint'
        elif msg_type == 'STOP':
            self.clear_waypoints()
            self.commands[self.robot_index] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            self.commands_source = 'server_stop'
            self.env.set_stop_called(True)

    def data_callback(self):
        for key in self._latest_data:
            if self._latest_data[key] is None:
                print(f"[Warning] socket server data missing for key: {key}")
                return None
        return format_data(
            self._latest_data["rgb"],
            self._latest_data["depth"],
            self._latest_data["position"],
            self._latest_data["quat_wxyz"],
            self._latest_data["info"]
        )

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

    def start_server(self, host="localhost", port=12300, server_name="IsaacLabServer"):
        # Start socket server in background
        server_thread = Thread(target=run_server, kwargs={
            "data_cb": self.data_callback, 
            "action_cb": self.action_callback, 
            "planner_cb": self.planner_callback,
            "host": host,
            "port": port,
            "server_name": server_name
        })
        server_thread.daemon = True
        server_thread.start()
        print("[INFO] Socket server started")
    
    def set_waypoints(self, waypoints, visualize=False):
        self.waypoints = waypoints
        self.waypoints_idx = 0
        if visualize:
            visualize_path(self.manager_env, waypoints, target_xyz=waypoints[-1])

    def clear_waypoints(self):
        self.waypoints = None
        self.commands[self.robot_index] = torch.tensor([0.0, 0.0, 0.0], device=self.device)

    def get_cam_pose(self):
        manager_env = self.manager_env
        # robot pose
        pos_robot = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy().astype(np.float32)
        quat_robot = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
        quat_robot = R.from_quat(np.concatenate([quat_robot[1:],quat_robot[:1]]))
        # transform from robot to camera
        pos_cam_body = manager_env.scene["pov_camera"].cfg.offset.pos
        quat_cam_body = manager_env.scene["pov_camera"].cfg.offset.rot
        quat_cam_body = R.from_quat(np.concatenate([quat_cam_body[1:],quat_cam_body[:1]]))
        # camera pose in world frame
        pos_cam_world = pos_robot + quat_robot.as_matrix() @ pos_cam_body
        rot_cam_world = quat_robot * quat_cam_body
        rot_cam_world = rot_cam_world.as_quat()
        rot_cam_world = np.concatenate([rot_cam_world[-1:],rot_cam_world[:-1]])
        return pos_cam_world, rot_cam_world


    def update_obs(self, obs, current_episode):
        # only publish the first robot's obs for now
        manager_env = self.manager_env
        try:
            cam_focal_length = manager_env.scene["pov_camera"].cfg.spawn.focal_length
            cam_horizontal_aperture = manager_env.scene["pov_camera"].cfg.spawn.horizontal_aperture
            hfov_deg = 2 * np.arctan(cam_horizontal_aperture / 2.0 / cam_focal_length) * 180.0 / np.pi
            self._latest_data["rgb"] = obs[0, :, :, :3].cpu().numpy().astype(np.uint8)
            depth = obs[0, :, :, 3].cpu().numpy()
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
            depth = np.clip(depth, 0, 65535).astype(np.uint16)
            self._latest_data["depth"] = depth
            self._latest_data["position"], self._latest_data["quat_wxyz"] = self.get_cam_pose()
            self._latest_data["info"] = {
                "scene_id": current_episode["scene_id"],
                "episode_id": current_episode["episode_id"],
                "instruction": current_episode["instruction"],
                "intrinsic": None,
                "robot_height": 0.61,
                "hfov_deg": hfov_deg,
            }
            if self.waypoints is not None and len(self.waypoints) > 0:
                self.commands[self.robot_index], self.waypoints_idx = follow_waypoints(self.manager_env, self.device, self.waypoints, self.waypoints_idx)
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
                self.clear_waypoints()
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