from isaaclab.envs import ManagerBasedRLEnv
from utils.measures import add_measurement
from utils.vis import visualize_path
from omni.kit.viewport.utility import get_viewport_from_window_name
import isaaclab.sim as sim_utils
from pxr import Gf
import torch

class VLNEnvWrapper:
    """Wrapper to configure an :class:`ManagerBasedRLEnv` instance to VLN environment."""

    def __init__(self, env: ManagerBasedRLEnv, 
                 low_level_policy, robot_name, 
                 episode, max_length=10000,
                 measure_names=["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
        ):
        self.env = env
        self.robot_name = robot_name
        self.episode = episode
        self.measure_names = measure_names
        self.viewport = get_viewport_from_window_name("Viewport")

        self.env_step = 0
        self.max_length = max_length

        self.high_level_obs_key = "camera"
        assert self.high_level_obs_key in self.env.observation_space.spaces.keys() # CHECK this

        self.low_level_policy = low_level_policy
        self.low_level_action = None

        self.curr_pos, self.prev_pos = None, None
        self.is_stop_called = False

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environment."""
        # set viewport
        self.viewport.set_active_camera("/World/envs/env_0/Robot/body/ThirdPersonCamera")
        terrain_prim = self.env.unwrapped.scene.stage.GetPrimAtPath('/World/ground/terrain')

        # set collider
        if self.episode.get("collider", False):
            collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
            sim_utils.define_collision_properties(terrain_prim.GetPrimPath(), collider_cfg)

        # set scene scale
        scene_scale = self.episode.get("scene_scale", 1.0)
        if scene_scale != 1.0:
            terrain_prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(scene_scale, scene_scale, scene_scale))

        low_level_obs, infos = self.env.reset()
        self.low_level_obs = low_level_obs
        zero_cmd = torch.tensor([0., 0., 0.], device=low_level_obs.device)

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

        self.prev_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach()

        # set viewer camera
        # if self.episode["scene_type"] == "nvidia":
        #     self.env.unwrapped.viewer.set_camera_position(self.episode["start_position"])
        #     self.env.unwrapped.viewer.set_camera_rotation(self.episode["start_rotation"])
        #     self.env.unwrapped.viewer.set_camera_fov(60)
        #     self.env.unwrapped.viewer.set_camera_near_clip(0.1)

        # visualize ref path
        for goal in self.episode["goals"]:
            ref_path = goal["reference_path"]
            visualize_path(
                self.env.unwrapped.unwrapped,
                ref_path,
                target_xyz=goal["location"]
            )
        # log
        obs = infos["observations"][self.high_level_obs_key]
        return obs, infos
    
    def update_command(self, command) -> None:
        """Update the command for the low-level policy."""

        # make sure command is a tensor on the same device as low_level_obs
        if not torch.is_tensor(command):
            command = torch.tensor(command, device=self.env.unwrapped.device)

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
        obs = info["observations"][self.high_level_obs_key]
        self.env_step += 1

        self.measure_manager.update_measures()
        measurements = self.measure_manager.get_measurements()
        info["measurements"] = measurements

        # Check if the robot has stayed in the same location for 1000 steps or env has reached max length
        same_pos = self.check_same_pos()
        done = done[0] or same_pos or self.env_step >= self.max_length

        return obs, reward, done, info
    
    def check_same_pos(self) -> bool:
        curr_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach()
        robot_vel = torch.norm(self.env.unwrapped.scene["robot"].data.root_vel_w[0].detach())
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

    