import numpy as np
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
import isaacsim.core.utils.bounds as bounds_utils

def terrain_out_of_bounds(
    env, asset_cfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
):
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    source: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/terminations.py
    """
    terrain_type = env.scene.cfg.terrain.terrain_type
    if terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    elif terrain_type == "usd":
        asset: RigidObject = env.scene[asset_cfg.name]
        min_x, min_y, min_z, max_x, max_y, max_z = env.get_prim_bounding_box("/World/ground/terrain")
        return (asset.data.root_pos_w[:, 0] > max_x - distance_buffer) | \
            (asset.data.root_pos_w[:, 0] < min_x + distance_buffer) | \
            (asset.data.root_pos_w[:, 1] > max_y - distance_buffer) | \
            (asset.data.root_pos_w[:, 1] < min_y + distance_buffer) | \
            (asset.data.root_pos_w[:, 2] < min_z)
    else:
        raise ValueError(f"Received unsupported terrain type: {terrain_type}, must be either 'plane' or 'generator'.")

def time_out(env, max_time: float = 600.0) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    if env.measure_manager:
        return env.measure_manager.get_measure("sim_duration").get_metric() >= max_time
    else:
        return False
    
def robot_stuck(env, asset_cfg = SceneEntityCfg("robot"), label: str = "stuck", max_time: float = 30.0, dist_threshold: float = 0.01, vel_threshold: float = 0.1):
    """Terminate the episode when the robot is not moving even though there is a command input for a certain time."""
    robot = env.scene[asset_cfg.name]
    curr_pos = robot.data.root_pos_w

    prev_pos = env.termination_states[f"{label}_prev_pos"]
    same_pos_count = env.termination_states[f"{label}_same_pos_count"]
    if hasattr(robot, "velocity_command"):
        robot_vel = torch.norm(robot.velocity_command, dim=-1)
    else:
        robot_vel = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    same_pos = torch.logical_and(torch.norm(curr_pos - prev_pos) < dist_threshold, robot_vel > vel_threshold)
    same_pos_count[same_pos] += 1
    same_pos_count[~same_pos] = 0
    env.termination_states[f"{label}_prev_pos"] = curr_pos

    # if the robot has stayed in the same location for 1000 steps
    return same_pos_count*env.manager_env.step_dt >= max_time

def stop_called(env):
    """Terminate the episode when the stop is called."""
    return env.is_stop_called

@configclass
class VLNTerminationsCfg:
    """Termination terms for the MDP.
    The env should have a termination_states dictionary to store the termination states.
    Once a termination is activated, the env will put termination reason into info.
    VLNSim will receive the info in update_obs() and pause the socket server until episode resets.
    """
    time_out = DoneTerm(func=time_out, time_out=True, params={"max_time": 100.0})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": float(np.deg2rad(45.0))},
    )
    terrain_out_of_bounds = DoneTerm(
        func=terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    stuck = DoneTerm(
        func=robot_stuck,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_time": 10.0,
            "label": "stuck",
            "dist_threshold": 0.05,
            "vel_threshold": 0.01,
        },
    )
    # back_n_forth = DoneTerm(
    #     func=robot_stuck,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "max_time": 30.0,
    #         "label": "back_n_forth",
    #         "dist_threshold": 2.0,
    #         "vel_threshold": 0,
    #     },
    # )
    stop_called = DoneTerm(
        func=stop_called,
    )