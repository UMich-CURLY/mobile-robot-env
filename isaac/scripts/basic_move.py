import argparse
import os
import cv2
import time
import math
import gzip, json
import numpy as np

# omni-isaaclab
from omni.isaac.lab.app import AppLauncher

import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to collect data from the matterport dataset.")
parser.add_argument("--episode_index", default=0, type=int, help="Episode index.")

parser.add_argument("--task", type=str, default="go2_matterport", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
# parser.add_argument("--draw_pointcloud", action="store_true", default=False, help="DRaw pointlcoud.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
# import ipdb; ipdb.set_trace()
simulation_app = app_launcher.app

import omni.isaac.core.utils.prims as prim_utils
import torch
from omni.isaac.core.objects import VisualCuboid

import gymnasium as gym
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.markers.config import CUBOID_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers
import omni.isaac.lab.utils.math as math_utils

from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper



def quat2eulers(q0, q1, q2, q3):
    """
    Calculates the roll, pitch, and yaw angles from a quaternion.

    Args:
        q0: The scalar component of the quaternion.
        q1: The x-component of the quaternion.
        q2: The y-component of the quaternion.
        q3: The z-component of the quaternion.

    Returns:
        A tuple containing the roll, pitch, and yaw angles in radians.
    """

    roll = math.atan2(2 * (q2 * q3 + q0 * q1), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
    pitch = math.asin(2 * (q1 * q3 - q0 * q2))
    yaw = math.atan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2)

    return roll, pitch, yaw

#uses task = go2_matterport_vision. where defined?
env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)

episode_idx = args_cli.episode_index
dataset_file_name = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")


with gzip.open(dataset_file_name, "rt") as f:
    deserialized = json.loads(f.read())
    episode = deserialized["episodes"][episode_idx]
    env_cfg.scene_id = episode["scene_id"].split('/')[1]


udf_file = os.path.join(ASSETS_DIR, f"matterport_usd/{env_cfg.scene_id}/{env_cfg.scene_id}.usd")
if os.path.exists(udf_file):
    env_cfg.scene.terrain.obj_filepath = udf_file #this loads the filepath to the mesh and textures for the rooms.
else:
    raise ValueError(f"No USD file found in scene directory: {udf_file}")  
if "go2" in args_cli.task:
    env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+0.4)

print("scene_id: ", env_cfg.scene_id)
print("robot_init_pos: ", env_cfg.scene.robot.init_state.pos)
print(env_cfg)
# initialize environment and low-level policy
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

if args_cli.history_length > 0:
    env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
else:
    env = RslRlVecEnvWrapper(env)


agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

log_root_path = os.path.join(os.path.dirname(__file__),"../logs", "rsl_rl", agent_cfg.experiment_name)
log_root_path = os.path.abspath(log_root_path)
# import pdb; pdb.set_trace()
resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)

ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)  # Adjust device as needed
ppo_runner.load(resume_path)

low_level_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
env = VLNEnvWrapper(env, low_level_policy, args_cli.task, episode, high_level_obs_key="camera_obs",
                    measure_names=all_measures)

robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
roll, pitch, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

obs, infos = env.reset()

from server import run_server














from planner import Planner
controller = Planner()
points = np.array([
    [0, 0], [1, 0],[2, 1],[3, 2.8], 
    [4, 2], [5, 1], [6, 1.2], [7, 2.2], 
    [7.5, 4], [7, 6], [6, 7], [4.5, 7.2], 
    [3, 6], [2, 4.5], [1.5, 3.5]
])

points = np.array([
    [0, 0.1], [1, 0],[1, 1.5],[1, 2],[0.5,2.2],[0,2.3],[-1,2.3]
])
controller.update_waypoints(points)
vel_command = torch.tensor([0, 0, 0.3])
from scipy.spatial.transform import Rotation
init_pos = env_cfg.scene.robot.init_state.pos

flag = True
while True:
    obs, _, done, infos = env.step(vel_command)

    rgb_image = infos['observations']['camera_obs'][0,:,:,:3].clone().detach()
    rgb_image_np = rgb_image.cpu().numpy()
    rgb_image_np = cv2.rotate(rgb_image_np, cv2.ROTATE_90_CLOCKWISE)

    depth_image = infos['observations']['camera_obs'][0,:,:,3].clone().detach()
    depth_image_np = depth_image.cpu().numpy()
    
    depth_image_np = cv2.rotate(depth_image_np,cv2.ROTATE_90_CLOCKWISE)

    # save_path_rgb = os.path.join(os.getcwd(), "rgb_image"+str(it-start_it)+".png")

    # proprio_go2 = infos['observations']['policy'].clone().detach().cpu().numpy()

    robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    
    robot_pos = robot_pos_w[:3]-init_pos 
    robot_ori_full_quat = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
    robot_yaw_quat = math_utils.yaw_quat(env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu()).unsqueeze(0)
    robot_yaw_angle = math_utils.euler_xyz_from_quat(robot_yaw_quat)[2].numpy()[0]
    if robot_yaw_angle>np.pi:
        robot_yaw_angle-=2*np.pi
    if(flag):
        init_rot = robot_yaw_angle*1
        flag = False

    vel_command = controller.step(robot_pos[0],robot_pos[1],robot_yaw_angle-init_rot)
    print(vel_command)
    print(controller._step(robot_pos[0],robot_pos[1],robot_yaw_angle-init_rot,0))
    print("robot_pos %s robot ori %s" % (str(robot_pos),str(robot_yaw_angle)))
    # robot_ori_full_rpy = math_utils.euler_xyz_from_quat(robot_ori_full_quat)
    
    print(rgb_image.shape)


    print(infos['observations'].keys())
    print(infos['measurements'].keys())

    print(obs.shape)
simulation_app.close()
print("closed!!!")