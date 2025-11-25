import torch
import numpy as np
import time
from threading import Thread
from scipy.spatial.transform import Rotation as R
import open3d as o3d
# Isaac Import
import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.utils.math import convert_camera_frame_orientation_convention

# Local imports
import utils.rsl_rl_cli_args as rsl_rl_cli_args
from utils.path_following_utils import WaypointFollower, set_yaw_to_forward
from utils.socket_server import run_server, format_data
from utils.vis import visualize_curve, visualize_points, visualize_arrow
from utils.episode import VLNEpisode, load_episode_set
from utils.vln_env_wrapper import VLNEnvWrapper, init_env_cfg
from utils.foxglove_utils import FoxgloveVisualizer
from robot.spot_flat_env_cfg import SpotFlatEnvCfg_PLAY
from threading import Lock

class VLNSim:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # env
        self.env = None
        self.manager_env = None
        self.info = None
        self.obs = None
        self.reward = None
        self.done = None
        self.load_env_lock = Lock()
        self.obs_index = 0
        self.cam_frame = -1

        # episode
        self.reset_flag = True
        self.sim_state = "init" # init, loading, running, terminated
        self.next_episode = None # this is updated when starting a new episode
        self.current_episode = None # this is updated after an episode is loaded
        self.episode_list = VLNEpisode.from_json_folder(self.args.episode_folder)
        self.episode_label_list = [x.episode_label for x in self.episode_list]

        # Shared buffers for server callbacks
        self._latest_data = None
        self.reset_server_cache()

        # simulation
        self.waypoints = []
        self.waypoint_follower = WaypointFollower(device=self.device)
        self.commands = torch.tensor([[0.0, 0.0, 0.0] for _ in range(args.num_envs)], device=self.device)
        self.commands_source = 'server'
        self.robot_index = 0
        self.viewport = get_viewport_from_window_name("Viewport")
        if self.args.headless:
            print("[INFO] Headless mode, disabling viewport updates")
            self.viewport.updates_enabled = False
        self.viewport_third_person = True

        # initialize server
        self.server_data_lock = Lock()
        if not args.disable_socket_server:
            self.start_server(host=args.host, port=args.port, server_name="BenchmarkServer")
        if args.foxglove_port>0:
            self.visualizer = FoxgloveVisualizer(host=args.host, port=args.foxglove_port)
        else:
            self.visualizer = None
    def init_env(self):
        if self.next_episode is None:
            raise ValueError("Current episode must be set before initializing the environment")
        # isaac lab manager
        args = self.args
        env_cfg = SpotFlatEnvCfg_PLAY()
        init_env_cfg(env_cfg, args, self.next_episode)
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.curriculum = None
        manager_env = ManagerBasedRLEnv(cfg=env_cfg)

        # policy
        TASK = "Isaac-Velocity-Flat-Spot-v0"
        RL_LIBRARY = "rsl_rl"
        agent_cfg = rsl_rl_cli_args.parse_rsl_rl_cfg(TASK, args)
        env = RslRlVecEnvWrapper(manager_env)
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
        ppo_runner.load(checkpoint)
        policy = ppo_runner.get_inference_policy(device=args.device)

        # vln env
        all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "SoftSPL", "OracleNavigationError", "OracleSuccess"]
        env = VLNEnvWrapper(args, env, policy, "spot", measure_names=all_measures)
        print("[INFO] Env setup complete")

        # viewpoint
        self.update_viewer()

        # keyboard
        self.set_up_keyboard()

        self.env = env
        self.manager_env = manager_env

    def reset(self, episode):
        with self.server_data_lock:
            self.reset_server_cache()
        self.next_episode = episode
        self.reset_flag = True
        if self.env is None:
            with self.load_env_lock:
                    self.init_env()
        self.clear_waypoints()

    def load_episode(self, episode_label):
        print(f"[INFO] Loading episode: {episode_label}")
        self.reset(self.episode_list[self.episode_label_list.index(episode_label)])

    def reset_server_cache(self):
        self._latest_data = {
            "rgb": None,
            "depth": None,
            "position": None,
            "quat_xyzw": None,
            "timestamp": None,
            "info": {}
        }

    def action_callback(self, msg_type, message):
        print(f"[INFO] Received command: {msg_type}")
        if msg_type == 'VEL':
            vx, vy, vw = message["vx"], message["vy"], message["vw"]
            self.commands[self.robot_index] = torch.tensor([float(vx), float(vy), float(vw)], device=self.device)
            self.commands_source = 'server_vel'
        elif msg_type == 'WAYPOINT':
            waypoints = message["waypoint"]
            self.set_waypoints(waypoints)
            self.commands_source = 'server_waypoint'
        elif msg_type == 'STOP':
            self.clear_waypoints()
            self.commands[self.robot_index] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            self.commands_source = 'server_stop'
            self.env.set_stop_called(self.robot_index, True)
        elif msg_type == 'EPISODE':
            # this will trigger a reset of the vln sim
            self.load_episode(message["episode_label"])

    def data_callback(self, request_type):
        if request_type == "GET_SENSOR_DATA":
            error_message = {
                "success": False,
                "message": "Unknown error",
                "episode_id": self.current_episode.get("episode_id", ""),
                "scene_id": self.current_episode.get("scene_id", ""),
            }
            if self.sim_state == "init":
                error_message["message"] = "Simulator is initializing"
                return error_message
            elif self.sim_state == "loading":
                error_message["message"] = "Episode is loading"
                return error_message
            elif self.sim_state == "terminated":
                error_message["message"] = "Episode has been terminated"
                error_message.update(self._latest_data["info"])
                return error_message
            # return the observation data
            with self.server_data_lock:
                for key in self._latest_data:
                    if self._latest_data[key] is None:
                        print(f"[Warning] socket server data missing for key: {key}")
                        error_message["message"] = "Episode is not ready yet"
                        return error_message
                return format_data(
                    self._latest_data["rgb"],
                    self._latest_data["depth"],
                    self._latest_data["position"],
                    self._latest_data["quat_xyzw"],
                    self._latest_data["info"],
                    self._latest_data["timestamp"]
                )
        elif request_type == "GET_EPISODE_LIST":
            episode_set_list = load_episode_set(self.args.episode_folder)
            episode_set_list["all"] = self.episode_label_list
            return episode_set_list

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
            "host": host,
            "port": port,
            "server_name": server_name
        })
        server_thread.daemon = True
        server_thread.start()
        print("[INFO] Socket server started")
    
    def set_waypoints(self, waypoints, fix_yaw=False):
        if fix_yaw:
            waypoints = set_yaw_to_forward(self.get_cam_pose()[0], waypoints)
        self.waypoints = waypoints
        self.waypoint_follower.reset()

    def clear_waypoints(self):
        self.waypoints = None
        self.commands[self.robot_index] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
    
    def follow_waypoints(self, visualize_waypoints=False):
        cam_pos, cam_quat = self.get_cam_pose()
        self.commands[self.robot_index] = self.waypoint_follower.update(cam_pos, cam_quat, self.waypoints)
        if self.waypoint_follower.arrived_at_goal:
            self.clear_waypoints()
        if visualize_waypoints:
            cam_height = cam_pos[2]
            points = [cam_pos]+[[wp[0], wp[1], cam_height] for wp in self.waypoints[self.waypoint_follower.current_wp_idx:]]
            visualize_curve(points, prim_path="/World/WaypointPath", width=0.02)
            visualize_arrow(self.waypoints[self.waypoint_follower.current_wp_idx:], cam_height, prim_path="/World/Arrow", color=(0.0, 1.0, 0.0), scale=(0.2, 0.2, 0.2))

    def get_cam_pose(self):
        manager_env = self.manager_env
        # robot pose
        pos_robot = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy().astype(np.float32)
        quat_robot = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32) # wxyz
        quat_robot = R.from_quat(np.concatenate([quat_robot[1:],quat_robot[:1]]))
        # transform from robot to camera
        pos_cam_body = manager_env.scene["pov_camera"].cfg.offset.pos
        quat_cam_body = manager_env.scene["pov_camera"].cfg.offset.rot # wxyz
        quat_cam_body = R.from_quat(np.concatenate([quat_cam_body[1:],quat_cam_body[:1]]))
        # camera pose in world frame
        pos_cam_world = pos_robot + quat_robot.as_matrix() @ pos_cam_body
        quat_cam_world = quat_robot * quat_cam_body
        quat_cam_world = quat_cam_world.as_quat() # xyzw
        pose = manager_env.scene["pov_camera"]._view.get_world_poses()
        pos = pose[0][0].detach().cpu().numpy()
        quat = convert_camera_frame_orientation_convention(pose[1][0], origin="opengl", target="world").detach().cpu().numpy()
        quat = np.concatenate([quat[1:],quat[:1]])
        return pos, quat
        # return pos_cam_world, quat_cam_world

    def step(self):
        # reset if needed
        if self.reset_flag:
            print(f"[INFO]: Resetting env state..")
            self.sim_state = "loading"
            self.env.reset_idx([self.robot_index])
            obs, _ = self.env.reset(self.next_episode)
            self.env.step(torch.tensor([0.0, 0.0, 0.0], device=self.device))
            self.reset_flag = False
            self.current_episode = self.next_episode
            self.next_episode = None
        # update waypoints
        if self.waypoints is not None and len(self.waypoints) > 0:
            self.follow_waypoints(visualize_waypoints=False)
        # Policy forward pass
        obs, reward, done, info = self.env.step(self.commands)
        self.info = info
        self.obs = obs
        self.reward = reward
        self.done = done
        # update obs
        with self.server_data_lock:
            # check if episode is terminated
            if "terminations" in info:
                self.sim_state = "terminated"
                self._latest_data["info"] = {
                    "terminations": info["terminations"]
                }
            else:
                self.sim_state = "running"
                self.update_obs(obs, info)
    
    def update_obs(self, obs, info):
        manager_env = self.manager_env
        current_episode = self.current_episode
        # do not update obs if camera frame is not updated
        if self.cam_frame == self.manager_env.scene.sensors['pov_camera'].frame:
            return
        self.cam_frame = self.manager_env.scene.sensors['pov_camera'].frame.clone()
        # update server data (only publish the first robot's obs for now)
        if not self.args.disable_socket_server:
            try:
                cam_focal_length = manager_env.scene["pov_camera"].cfg.spawn.focal_length
                cam_horizontal_aperture = manager_env.scene["pov_camera"].cfg.spawn.horizontal_aperture
                hfov_deg = 2 * np.arctan(cam_horizontal_aperture / 2.0 / cam_focal_length) * 180.0 / np.pi
                self._latest_data["rgb"] = obs['pov_rgb'][0, :, :, :3].cpu().numpy().astype(np.uint8)
                depth = obs['pov_depth'][0, :, :, 0].cpu().numpy()
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
                depth = np.clip(depth, 0, 65535).astype(np.uint16)
                self._latest_data["depth"] = depth
                # self._latest_data["position"], self._latest_data["quat_xyzw"] = self.get_cam_pose()
                obs_pose = obs['pov_pose'][self.robot_index].cpu().numpy()
                self._latest_data["position"] = obs_pose[:3]
                self._latest_data["quat_xyzw"] = obs_pose[3:]
                # print(self._latest_data["position"]-obs_pose[:3], self._latest_data["quat_xyzw"]-obs_pose[3:])
                self._latest_data["timestamp"] = time.time_ns()
                self._latest_data["info"] = {
                    "scene_id": current_episode["scene_id"],
                    "episode_id": current_episode["episode_id"],
                    "instruction": current_episode["instruction"],
                    "intrinsic": None,
                    "robot_height": 0.61,
                    "hfov_deg": hfov_deg,
                    "metrics": info["measurements"],
                }
            except Exception as e:
                print(f"Error updating obs: {e}")
                import traceback
                traceback.print_exc()

        self.obs_index += 1
        # save pcd
        # self.save_debug_data(obs)
        if self.visualizer is not None:
            self.visualize_obs(obs)
    
    def save_debug_data(self, obs):
        if self.obs_index%10==0:
            data = {
                "rgb": obs['pov_rgb'][0, :, :, :3].cpu().numpy().astype(np.uint8),
                "depth": obs['pov_depth'][0, :, :, 0].cpu().numpy(),
                "pose": obs['pov_pose'].cpu().numpy(),
            }
            np.savez(f"data/obs_{self.obs_index//10}.npz", **data)
    
    def visualize_obs(self, obs):
        if not self.visualizer.listener.has_subscribers() and self.obs_index>1:
            return
        time_start = time.time()
        rgb = obs['pov_rgb'][0, :, :, :3].cpu().numpy().astype(np.uint8)
        depth = obs['pov_depth'][0, :, :, 0].cpu().numpy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        pose = obs['pov_pose'][self.robot_index].cpu().numpy()
        pose_matrix = np.eye(4)
        pose_matrix[:3, 3] = pose[:3]
        pose_matrix[:3, :3] = R.from_quat(pose[3:]).as_matrix()
        # convert pose: x (forward), y (left), z (up) -> x (right), y (down), z (forward)
        S = np.eye(4)
        S[:3, :3] = np.array(
            [[0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]]
        )
        pose_matrix = S @ pose_matrix @ S.T
        # log tf, rgb, depth
        self.visualizer.log({
            'type': 'tf',
            'channel_name': 'robot_tf',
            'pose': pose_matrix,
            'parent_frame_id': 'world',
            'child_frame_id': 'robot',
        })
        self.visualizer.log({
            'type': 'image',
            'channel_name': 'obs_rgb',
            'image': rgb,
            'frame_id': 'robot',
        })
        self.visualizer.log({
            'type': 'image',
            'channel_name': 'obs_depth',
            'image': depth,
            'frame_id': 'robot',
        })
        # log point cloud
        if self.obs_index%5==0:
            rgb_image = o3d.geometry.Image(rgb)
            depth_image = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, convert_rgb_to_intensity=False, depth_scale=1., depth_trunc=10000.)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=rgb.shape[1],
                height=rgb.shape[0],
                fx=733,
                fy=733,
                cx=rgb.shape[1]//2,
                cy=rgb.shape[0]//2,
            )
            obs_pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            obs_pc = o3d.geometry.PointCloud.voxel_down_sample(obs_pc, voxel_size=0.03)
            points = np.asarray(obs_pc.points)
            colors = np.asarray(obs_pc.colors)
            self.visualizer.log({
                'type': 'point_cloud',
                'channel_name': 'obs_pc',
                'points': points,
                'color': colors,
                'pose': pose_matrix,
                'frame_id': 'world',
            })
            print(f"Visualize obs time: {time.time() - time_start}")

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 2
        R = 2
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0], device=self.device), # forward
            "DOWN": torch.tensor([-T, 0.0, 0.0], device=self.device), # backward
            "LEFT": torch.tensor([0, 0.0, R], device=self.device), # turn left
            "RIGHT": torch.tensor([0, 0.0, -R], device=self.device), # turn right
            "W": torch.tensor([T, 0.0, 0.0], device=self.device), # forward
            "S": torch.tensor([-T, 0.0, 0.0], device=self.device), # backward
            "A": torch.tensor([0, 0.5*T, 0.0], device=self.device), # left
            "D": torch.tensor([0, -0.5*T, 0.0], device=self.device), # right
            "Q": torch.tensor([0, 0.0, R], device=self.device), # turn left
            "E": torch.tensor([0, 0.0, -R], device=self.device), # turn right
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