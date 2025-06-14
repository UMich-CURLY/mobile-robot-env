import argparse
import os
import cv2
import time
import math
import gzip, json
import numpy as np
from scipy.spatial.transform import Rotation
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
from omni.isaac.lab.markers import VisualizationMarkers,VisualizationMarkersCfg
import omni.isaac.lab.utils.math as math_utils


import omni.isaac.lab.sim as sim_utils

from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper

from protocol import *
from planner import *

def quat_wxyz_to_xyzw(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])
def quat_xyzw_to_wxyz(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])

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
env_cfg.viewer.resolution=(640,480)
# env_cfg.sim.dt=0.05
# ender.rendering_mode='performance'
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
    env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+0.2)
  

print("scene_id: ", env_cfg.scene_id)
print("robot_init_pos: ", env_cfg.scene.robot.init_state.pos)
print(env_cfg)
# initialize environment and low-level policy
env = gym.make(args_cli.task, cfg=env_cfg, render_mode='performance')

# env.env.physics_dt = 0.01 #0.005
print(env.env.step_dt)
print(env.env.physics_dt)
# simulation_app.close()

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

marker_cfg = CUBOID_MARKER_CFG.copy()
marker_cfg.prim_path = "/Visuals/Command/pos_goal_command"
marker_cfg.markers["cuboid"].scale = (0.5, 0.5, 0.5)

cfg = VisualizationMarkersCfg(
    prim_path="/World/Visuals/testMarkers",
    markers={
        "marker1": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
    }
)

cfg2 = VisualizationMarkersCfg(
    prim_path="/World/Visuals/testMarkers",
    markers={
        "marker1": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        )
    }
)
     


from server import run_server,format_data
from planner import Planner
vel_command = np.array([0,0,0.0])
use_planner = False
planner = Planner(cruise_vel=1.2,max_vw=2)
rgb,depth,position,quat = None,None,None,None
identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device).repeat(1, 1)
point_visualizer = VisualizationMarkers(cfg)

bot_visualizer = VisualizationMarkers(cfg2)
# point_visualizer.set_visibility(True)


def action_callback(message):
    global vel_command
    global use_planner
    global position,quat
    global planner

    if message.type == 'VEL':
        print(message.omega)
        vel_command[0] = message.x
        vel_command[1] = message.y
        vel_command[2] = message.omega
        print(vel_command)

        # print(f"position: {position} quat: {quat}")
        use_planner = False
    if message.type == 'WAYPOINT':
        wps = np.vstack((message.x,message.y)).T
        use_planner = True
        # for visualizer in visualizers:
        #     visualizer.set_visibility(False)
        # visualizers.clear()
        translations = np.hstack((wps,np.ones((len(wps),1))*0.2))
        print("first waypoint raw: %s" % str(translations[0]))

        # translations = translations @ init_mat.T+np.array(init_pos)

        planner.update_waypoints(translations[:,:2])

        point_visualizer.visualize(translations)
        bot_visualizer.visualize((np.array(position)+np.array([0,0,0.2])).reshape((1,3)))
        print("first waypoint: %s" % str(translations[0]))
        # for i in range(len(wps)):
        #     pos = wps[i]
  

        #     point = np.array([pos[0]+init_pos[0],pos[1]+init_pos[1],position[2]]).reshape(1, 3)
        #     print(point)
        #     print(position)
        #     default_scale = point_visualizer.cfg.markers["cuboid"].scale
        #     # import ipdb; ipdb.set_trace()
        #     larger_scale = 2.0*torch.tensor(default_scale, device=env.unwrapped.device).repeat(1, 1)
        #     point_visualizer.visualize(point, identity_quat,larger_scale)


init_pos = None
init_quat = None
init_mat = None

def data_callback():
    global rgb,depth,position,quat,init_pos,init_quat,init_mat
    if init_pos is None:
        init_pos = position
        init_quat = quat
        init_mat = Rotation.from_quat(quat_wxyz_to_xyzw(quat)).as_matrix()
    return format_data(rgb,depth,position,quat)

def planner_callback():

    global planner,use_planner
    try:
        ex,ey,_ = planner.get_tracking_error()
        return {'err_x':ex,'err_y':ey,"cmd_x":planner.cmd_x,"cmd_y":planner.cmd_y,"cmd_w":planner.cmd_w}
    except:
        return {}
from threading import Thread

server_thread = Thread(target=run_server,kwargs={"data_cb":data_callback,"action_cb":action_callback,"planner_cb":planner_callback})
started = False
i=1
try:
    while True:
        print("velocity: %s" % str(vel_command))
        obs, _, done, infos = env.step(vel_command)
        rgb_image = infos['observations']['camera_obs'][0,:,:,:3].clone().detach()
        rgb = rgb_image.cpu().numpy()
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        depth_image = infos['observations']['depth_obs'][0,0,:,:].clone().detach()*1000
        depth = depth_image.cpu().numpy().astype(np.uint16)

        # save_path_rgb = os.path.join(os.getcwd(), "rgb_image"+str(it-start_it)+".png")

        # proprio_go2 = infos['observations']['policy'].clone().detach().cpu().numpy()

        robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        
        position = robot_pos_w[:3]
        quat = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()



        
        if use_planner:
            current_rotation = Rotation.from_quat(quat_wxyz_to_xyzw(quat))
            relative_rotation = current_rotation * Rotation.from_quat(quat_wxyz_to_xyzw(init_quat)).inv()
            
            robot_yaw_angle = current_rotation.as_euler('zyx')[0]
            if robot_yaw_angle>np.pi:
                robot_yaw_angle-=2*np.pi
            robot_pos = robot_pos_w[:3]#-init_pos 
            print("trying")
            vel_command = np.array(planner.step(robot_pos[0],robot_pos[1],robot_yaw_angle))
            print("done")



        if not started:
            server_thread.start()
            print("server started")
            started=True
        # if(i%10==0):
        #     i=0
        #     fps = omni.kit.viewport.utility.get_active_viewport().fps

        #     print(fps)
        i+=1
        # print("robot_pos %s robot ori %s" % (str(robot_pos),str(robot_ori_full_quat)))
        # robot_ori_full_rpy = math_utils.euler_xyz_from_quat(robot_ori_full_quat)
        
        # print(rgb_image.shape)


        # print(infos['observations'].keys())
        # print(infos['measurements'].keys())

        # print(obs.shape)
except Exception as e:
    print(e)
finally:
    simulation_app.close()
    print("closed!!!")