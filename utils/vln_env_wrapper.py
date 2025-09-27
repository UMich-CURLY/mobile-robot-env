from pathlib import Path
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporter
from utils.measures import add_measurement
from utils.vis import visualize_path
import isaaclab.sim as sim_utils
import isaacsim.core.utils.bounds as bounds_utils
from pxr import Gf
import torch

class VLNEnvWrapper:
    """Wrapper to configure an :class:`RslRlVecEnvWrapper` instance to VLN environment."""

    def __init__(self, args, env, 
                 low_level_policy, robot_name, max_length=10000,
                 measure_names=["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
        ):
        self.env = env
        self.manager_env = env.unwrapped
        self.scene = self.manager_env.scene
        self.robot_name = robot_name
        self.measure_names = measure_names
        self.usd_path = None
        self.args = args

        self.env_step = 0
        self.max_length = max_length

        self.high_level_obs_key = "camera"
        if not self.args.disable_camera:
             # check if camera observation is available
            assert self.high_level_obs_key in self.env.observation_space.spaces.keys()

        self.low_level_policy = low_level_policy
        self.low_level_action = None

        self.curr_pos, self.prev_pos = None, None
        self.is_stop_called = False

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

    def reset(self, episode=None) -> tuple[torch.Tensor, dict]:
        """Reset the environment."""
        if episode is not None:
            self.episode = episode
        else:
            if self.episode is None:
                raise ValueError("Episode is not set")

        # load scene
        if self.usd_path != self.episode["path"]:
            while len(self.scene.terrain.terrain_prim_paths) > 0:
                self.scene.stage.RemovePrim(self.scene.terrain.terrain_prim_paths[0])
                while self.scene.stage.GetPrimAtPath(self.scene.terrain.terrain_prim_paths[0]).IsValid():
                    zero_cmd = torch.tensor([0., 0., 0.], device=self.args.device)
                    self.update_command(zero_cmd)
                    actions = self.low_level_policy(self.low_level_obs)
                    low_level_obs, _, _, infos = self.env.step(actions)
                self.scene.terrain.terrain_prim_paths.pop(0)
            print(f"Waiting for scene {self.episode['path']} to be loaded...")
            if self.episode["path"] == "generator":
                self.manager_env.cfg.load_generator()
                self.manager_env.scene._terrain = TerrainImporter(self.manager_env.cfg.scene.terrain)
                self.scene.terrain.terrain_prim_paths.append("/World/ground/terrain")
            else:
                self.manager_env.cfg.load_usd(str(Path(self.args.scene_folder) / self.episode["path"]))
                self.manager_env.scene._terrain = TerrainImporter(self.manager_env.cfg.scene.terrain)

        robot_root_state = self.scene["robot"].data.default_root_state.clone()
        robot_root_state[:, 0:3] = torch.tensor(episode["start_position"], device=self.args.device)
        robot_root_state[:, 3:7] = torch.tensor(episode["start_rotation"], device=self.args.device)
        self.scene["robot"].write_root_state_to_sim(robot_root_state)

        # reset low-level environment
        low_level_obs, infos = self.env.reset()
        self.low_level_obs = low_level_obs

        # set collider and scene scale
        terrain_prim = self.scene.stage.GetPrimAtPath('/World/ground/terrain')
        if self.episode.get("collider", True):
            collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
            sim_utils.define_collision_properties(terrain_prim.GetPrimPath(), collider_cfg)
        scene_scale = self.episode.get("scene_scale", 1.0)
        # if scene_scale != 1.0:
        terrain_prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(scene_scale, scene_scale, scene_scale))
        if self.episode.get("align_ground", True):
            bb_cache = bounds_utils.create_bbox_cache()
            min_x, min_y, min_z, max_x, max_y, max_z = bounds_utils.compute_combined_aabb(bb_cache, prim_paths=[terrain_prim.GetPrimPath()])
            print(f"Bounding box: min_x: {min_x}, min_y: {min_y}, min_z: {min_z}, max_x: {max_x}, max_y: {max_y}, max_z: {max_z}")
            terrain_prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(0, 0, -min_z-20))
        # warmup_steps = 50

        # for i in range(warmup_steps):
        #     if i % 10 == 0 or i == warmup_steps - 1:
        #         print(f"Warmup step {i}/{warmup_steps}...")

        #     self.update_command(zero_cmd)
        #     actions = self.low_level_policy(self.low_level_obs)
        #     low_level_obs, _, _, infos = self.env.step(actions)
        #     self.low_level_obs = low_level_obs
        #     self.low_level_action = actions

        self.env_step, self.same_pos_count = 0, 0

        self.measure_manager = add_measurement(self.env, self.episode, self.measure_names)

        self.measure_manager.reset_measures()
        measurements = self.measure_manager.get_measurements()
        infos["measurements"] = measurements

        self.prev_pos = self.scene["robot"].data.root_pos_w[0].detach()

        # log
        if not self.args.disable_camera:
            obs = infos["observations"][self.high_level_obs_key]
        else:
            obs = []
        return obs, infos
    
    def update_command(self, command) -> None:
        """Update the command for the low-level policy."""

        # make sure command is a tensor on the same device as low_level_obs
        if not torch.is_tensor(command):
            command = torch.tensor(command, device=self.manager_env.device)
        self.low_level_obs = self.low_level_obs.clone()
        self.low_level_obs[:, 9:12] = command

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

        self.measure_manager.update_measures()
        measurements = self.measure_manager.get_measurements()
        info["measurements"] = measurements

        # Check if the robot has stayed in the same location for 1000 steps or env has reached max length
        same_pos = self.check_same_pos()
        done = done[0] or same_pos or self.env_step >= self.max_length

        return obs, reward, done, info
    
    def check_same_pos(self) -> bool:
        curr_pos = self.scene["robot"].data.root_pos_w[0].detach()
        robot_vel = torch.norm(self.scene["robot"].data.root_vel_w[0].detach())
        if torch.norm(curr_pos - self.prev_pos) < 0.01 and robot_vel < 0.1:
            self.same_pos_count += 1
        else:
            self.same_pos_count = 0
        self.prev_pos = curr_pos

        # Break out of the loop if the robot has stayed in the same location for 1000 steps
        if self.same_pos_count >= 1000:
            print("Robot has stayed in the same location for 1000 steps. Breaking out of the loop.")
            return True
        
        return False

    def set_stop_called(self, is_stop_called: bool) -> None:
        """Set the stop called flag."""
        self.env.is_stop_called = is_stop_called
    
    def close(self) -> None:
        self.env.close()

    