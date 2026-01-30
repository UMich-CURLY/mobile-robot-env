from pathlib import Path
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporter
from utils.measures import add_measurement
import isaaclab.sim as sim_utils
import isaacsim.core.utils.bounds as bounds_utils
from pxr import Gf
import torch
from utils.termination_cfg import VLNTerminationsCfg
from isaaclab.managers import TerminationManager
import isaacsim.core.utils.prims as prim_utils
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaacsim.core.utils.stage import add_reference_to_stage
import os

def init_env_cfg(env_cfg, args, episode):
    load_scene(env_cfg, args, episode)
    set_robot_pose(env_cfg, episode)
    
def load_scene(env_cfg, args, episode):
    scene_path = episode["path"]
    print(f"[INFO] Loading scene {scene_path}...")
    if scene_path == "generator":
        env_cfg.load_generator()
    else:
        print(f"[DEBUG] Loading USD from {str(Path(args.scene_folder) / scene_path)}")
        env_cfg.load_usd(str(Path(args.scene_folder) / scene_path))

def set_robot_pose(env_cfg, episode, robot=None):
    pos = list(episode["start_position"])
    pos[2] += 0.8
    rot = list(episode["start_rotation"]) # wxyz
    env_cfg.scene.robot.init_state.pos = pos
    env_cfg.scene.robot.init_state.rot = rot
    if robot is not None:
        robot_root_state = robot.data.default_root_state.clone()
        robot_root_state[:, 0:3] = torch.tensor(pos, device=robot.device)
        robot_root_state[:, 3:7] = torch.tensor(rot, device=robot.device)
        robot_root_state[:, 7:] = 0.
        robot.write_root_state_to_sim(robot_root_state)
        # robot.reset()
        robot.write_data_to_sim()


class VLNEnvWrapper:
    """Wrapper to configure an :class:`RslRlVecEnvWrapper` instance to VLN environment."""

    def __init__(self, args, env, low_level_policy, robot_name, measure_names=None):
        self.env = env
        self.manager_env = env.unwrapped
        self.sim = self.manager_env.sim
        self.device = self.manager_env.device
        self.scene = self.manager_env.scene
        self.robot_name = robot_name
        self.first_init = True
        self.usd_path = None
        self.scene_scale = None
        self.scene_setting = None
        self.args = args
        self.num_envs = self.manager_env.num_envs

        if measure_names is None:
            measure_names = [
                "PathLength",
                "DistanceToGoal",
                "ClosestGoal",
                "Success",
                "SPL",
                "OracleNavigationError",
                "OracleSuccess",
                "SimDuration",
            ]
        self.measure_names = measure_names

        self.env_step = 0
        self.step_dt = self.manager_env.step_dt

        self.high_level_obs_key = "camera"
        if not self.args.disable_camera:
             # check if camera observation is available
            assert self.high_level_obs_key in self.env.observation_space.spaces.keys()

        self.low_level_policy = low_level_policy
        self.low_level_action = None

        self.is_stop_called = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.measure_manager = None
        self.termination_states = {
            "stuck_prev_pos": torch.zeros(self.num_envs, 3, device=self.args.device),
            "stuck_same_pos_count": torch.zeros(self.num_envs, dtype=torch.int32, device=self.args.device),
            "stuck_dist_history": [[] for _ in range(self.num_envs)],
            "back_n_forth_prev_pos": torch.zeros(self.num_envs, 3, device=self.args.device),
            "back_n_forth_same_pos_count": torch.zeros(self.num_envs, dtype=torch.int32, device=self.args.device),
        }
        self.terminations_cfg = VLNTerminationsCfg()
        print(f"original max_time:", self.terminations_cfg.time_out.params["max_time"])
        print(f"new max time:", self.args.timeout)
        self.terminations_cfg.time_out.params["max_time"] = self.args.timeout
        self.termination_manager = TerminationManager(self.terminations_cfg, self)

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.
        """
        return self.manager_env
    
    def get_prim_bounding_box(self, prim_path):
        """Return min_x, min_y, min_z, max_x, max_y, max_z of the prim."""
        prim = self.scene.stage.GetPrimAtPath(prim_path)
        if prim is None:
            raise ValueError(f"Prim at path {prim_path} not found")
        bb_cache = bounds_utils.create_bbox_cache()
        return bounds_utils.compute_combined_aabb(bb_cache, prim_paths=[prim_path])

    def remove_prim(self, rule):
        prim_list = prim_utils.find_matching_prim_paths(rule)
        for prim_path in prim_list:
            prim_utils.delete_prim(prim_path)

    def get_prim_position(self, prim_path):
        """Return position of the prim (center of the bounding box)."""
        min_x, min_y, min_z, max_x, max_y, max_z = self.get_prim_bounding_box(prim_path)
        return (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2

    def get_prim_radius(self, prim_path):
        """Return radius of the prim."""
        min_x, min_y, min_z, max_x, max_y, max_z = self.get_prim_bounding_box(prim_path)
        return max(max_x - min_x, max_y - min_y, max_z - min_z) / 2

    def get_prim_orientation(self, prim_path):
        """Return orientation of the prim."""
        prim = self.scene.stage.GetPrimAtPath(prim_path)
        if prim is None:
            raise ValueError(f"Prim at path {prim_path} not found")
        return prim.GetAttribute('xformOp:orient').Get()
    
    def get_body_pose(self):
        manager_env = self.manager_env
        pos_robot = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy().astype(np.float32)
        quat_robot = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32) # wxyz
        quat_robot = np.concatenate([quat_robot[1:],quat_robot[:1]])
        return pos_robot, quat_robot

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
        # pose = manager_env.scene["pov_camera"]._view.get_world_poses()
        # pos = pose[0][0].detach().cpu().numpy()
        # quat = convert_camera_frame_orientation_convention(pose[1][0], origin="opengl", target="world").detach().cpu().numpy()
        # quat = np.concatenate([quat[1:],quat[:1]])
        # return pos, quat
        return pos_cam_world, quat_cam_world
    
    def create_camera(self, prim_path="/World/camera", perspective=True, pos=None, quat_opengl=None, focal_length=24.0, horizontal_aperture=20.955, clipping_range_min=0.1, clipping_range_max=20.0, width=640, height=480):
        if prim_utils.get_prim_at_path(prim_path):
            prim_utils.delete_prim(prim_path)
        camera_cfg = TiledCameraCfg(
            prim_path=prim_path,
            update_period=0.1,
            update_latest_camera_pose=True,
            offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=quat_opengl, convention="opengl"),
            width=width,
            height=height,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, horizontal_aperture=horizontal_aperture,
                clipping_range=(clipping_range_min, clipping_range_max)
            ),
        )
        tiled_camera = TiledCamera(camera_cfg)
        if not perspective:
            camera_prim = self.manager_env.scene.stage.GetPrimAtPath(prim_path)
            camera_prim.GetAttribute('projection').Set("orthographic")
        tiled_camera._initialize_callback(None)
        return tiled_camera

    def reset(self, episode=None, warmup_steps=0) -> tuple[torch.Tensor, dict]:
        """Reset the environment."""
        zero_cmd = torch.tensor([0., 0., 0.], device=self.args.device)
        if episode is not None:
            self.episode = episode
        else:
            if self.episode is None:
                raise ValueError("Episode is not set")
        # access episode settings
        collider = self.episode.get("collider", True)
        scene_scale = self.episode.get("scene_scale", 1.0)
        align_ground = self.episode.get("align_ground", True)
        scene_settings = {
            "scale": scene_scale,
            "align_ground": align_ground,
            "collider": collider,
        }

        # load measures
        self.measure_manager = add_measurement(self, self.episode, self.measure_names)

        # load scene
        scene_changed = self.usd_path != self.episode["path"]
        settings_changed = (self.scene_setting is None) or (self.scene_setting != scene_settings)

        # do not update scene if it is the first init
        if scene_changed or settings_changed:
            # remove previous scene
            while len(self.scene.terrain.terrain_prim_paths) > 0:
                self.scene.stage.RemovePrim(self.scene.terrain.terrain_prim_paths[0])
                while self.scene.stage.GetPrimAtPath(self.scene.terrain.terrain_prim_paths[0]).IsValid():
                    self.update_command(zero_cmd)
                    actions = self.low_level_policy(self.low_level_obs)
                    low_level_obs, _, _, infos = self.env.step(actions)
                self.scene.terrain.terrain_prim_paths.pop(0)
            print(f"[ENV] Waiting for previous scene to be removed...")
            # load new scene
            load_scene(self.manager_env.cfg, self.args, self.episode)
            self.manager_env.scene._terrain = TerrainImporter(self.manager_env.cfg.scene.terrain)
            if self.episode["path"] == "generator":
                self.scene.terrain.terrain_prim_paths.append("/World/ground/terrain")
            # disable all lights in scene
            prim_list = [x for x in self.manager_env.scene.stage.Traverse()]
            for prim in prim_list:
                from_usd = str(prim.GetPath()).startswith("/World/ground/terrain")
                is_domelight = prim.GetTypeName() == "DomeLight"
                if from_usd and is_domelight:
                    prim.SetActive(False)
            default_dome_light = self.scene.stage.GetPrimAtPath('/World/skyLight')
            default_dome_light.SetActive(False)
            # add ref to usd file
            add_reference_to_stage(prim_path="/World/sky", usd_path=os.path.join(self.args.scene_folder, "nvidia", "sky", "CloudySky.usd"))
            # apply scale and translation
            terrain_prim = self.scene.stage.GetPrimAtPath('/World/ground/terrain')
            if collider:
                collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
                sim_utils.define_collision_properties(terrain_prim.GetPrimPath(), collider_cfg)
            terrain_prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(scene_scale, scene_scale, scene_scale))
            if align_ground:
                min_x, min_y, min_z, max_x, max_y, max_z = self.get_prim_bounding_box(terrain_prim.GetPrimPath())
                print(f"Bounding box: min_x: {min_x}, min_y: {min_y}, min_z: {min_z}, max_x: {max_x}, max_y: {max_y}, max_z: {max_z}")
                terrain_prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(0, 0, -min_z-20))
        self.scene_setting = scene_settings
        self.usd_path = self.episode["path"]

        # reset robot position
        robot = self.scene["robot"]
        set_robot_pose(self.manager_env.cfg, episode, robot)

        # reset termination states
        self.reset_termination_states([*range(self.num_envs)])
        self.termination_manager.reset()

        # reset low-level environment
        print(f"[INFO] Updating environment, this may take a while...")
        low_level_obs, infos = self.env.reset()
        self.low_level_obs = low_level_obs
        for i in range(self.num_envs):
            self.set_stop_called(i, False)

        for i in range(warmup_steps):
            if i==0:
                print("Resetting environment...")
            elif i%10==0:
                print(f"Warmup {i} / {warmup_steps}...")
            self.update_command(zero_cmd)
            actions = self.low_level_policy(self.low_level_obs)
            low_level_obs, _, _, infos = self.env.step(actions)
            self.low_level_obs = low_level_obs
            self.low_level_action = actions

        self.env_step = 0
        self.same_pos_count = 0

        self.measure_manager.reset_measures()
        measurements = self.measure_manager.get_measurements()
        infos["measurements"] = measurements

        self.first_init = False

        # log
        if not self.args.disable_camera:
            obs = infos["observations"][self.high_level_obs_key]
        else:
            obs = []
        return obs, infos
    
    def reset_termination_states(self, reset_env_ids):
        for env_id in reset_env_ids:
            for key in self.termination_states.keys():
                if "dist_history" in key:
                    self.termination_states[key][env_id] = []
                else:
                    self.termination_states[key][env_id] = 0
    
    def update_command(self, command) -> None:
        """Update the command for the low-level policy."""

        # make sure command is a tensor on the same device as low_level_obs
        if not torch.is_tensor(command):
            command = torch.tensor(command, device=self.manager_env.device)
        self.low_level_obs = self.low_level_obs.clone()
        self.low_level_obs[:, 9:12] = command
        self.scene["robot"].velocity_command = command

    def step(self, action) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take a step in the environment.

        Args:
            action: The action of high-level planner, which should be velocity command to the low-level policy.

        Returns:
            obs: The observation of the high-level planner.
            reward: The reward of the environment.
            done: Whether the episode is done.
            info: Additional information of the environment.
        
        """

        self.update_command(action)

        low_level_action = self.low_level_policy(self.low_level_obs)
        self.low_level_action = low_level_action

        low_level_obs, reward, done, info = self.env.step(low_level_action)
        self.low_level_obs = low_level_obs
        if not self.args.disable_camera:
            obs = info["observations"][self.high_level_obs_key]
        else:
            obs = []
        self.env_step += 1

        # update measures
        self.measure_manager.update_measures()
        measurements = self.measure_manager.get_measurements()
        info["measurements"] = measurements
        # update terminations
        if not self.args.disable_termination:
            self.reset_buf = self.termination_manager.compute()
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            termination_terms = self.termination_manager.get_active_iterable_terms(0)
            done = done.any() or self.reset_buf.any()
            if self.reset_buf.any():
                termination_reason = max(termination_terms, key=lambda x:x[1])[0]
                info["terminations"] = {"termination_reason": termination_reason}
                # print termination reason if it is not the same as the last time
                print_termination_reason = (self.episode['episode_label'], termination_reason)
                if not hasattr(self, 'print_termination_reason') or self.print_termination_reason != print_termination_reason:
                    self.print_termination_reason = print_termination_reason
                    print(f"Episode {self.episode['episode_label']} terminated due to {termination_reason}")
                    print("Measures: ", ", ".join([f"{k}={v:.2f}" for k, v in info["measurements"].items()]))
            else:
                if "terminations" in info:
                    del info["terminations"]
        else:
            # termination disabled, only respond to stop_called
            done = done.any() or self.is_stop_called.any()
            if self.is_stop_called.any():
                info["terminations"] = {"termination_reason": "stop_called"}
                print(f"Episode {self.episode['episode_label']} terminated due to stop_called")
            else:
                if "terminations" in info:
                    del info["terminations"]


        return obs, reward, done, info
    
    def reset_idx(self, reset_env_ids):
        reset_env_ids = torch.tensor(reset_env_ids, device=self.device, dtype=torch.int32)
        self.reset_termination_states(reset_env_ids)
        self.reset_buf = self.termination_manager.compute()
        for env_id in reset_env_ids:
            self.set_stop_called(env_id, False)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.manager_env.recorder_manager.record_pre_reset(reset_env_ids)
            self.manager_env._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.manager_env.sim.has_rtx_sensors():
                num_rerenders_on_reset = 1
                for _ in range(num_rerenders_on_reset):
                    self.manager_env.sim.render()
            # trigger recorder terms for post-reset calls
            self.manager_env.recorder_manager.record_post_reset(reset_env_ids)

    def set_stop_called(self, robot_index: int, is_stop_called: bool) -> None:
        """Set the stop called flag."""
        self.is_stop_called[robot_index] = is_stop_called
    
    def close(self) -> None:
        self.env.close()
