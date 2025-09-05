# cd pohsun/SG-VLN
# ../IsaacLab/isaaclab.sh -p robot_env/isaac_lab_server_spot.py --enable_cameras
# Isaac Lab version of Spot robot server with Matterport scene

import argparse
import os
import sys
import numpy as np
import torch
import omni
import io
import time
import math
from threading import Thread

# start simulation
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Isaac Lab Server for Spot robot with USD scene")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--scene_path", type=str, default=None, help="Path to USD scene file")
parser.add_argument("--robot_pos", type=str, default="-52.8,11.5,0.8", help="Robot initial position (x,y,z)")
parser.add_argument("--checkpoint", type=str, default="/home/junzhewu/pohsun/SG-VLN/robot_env/spot_policy.pt", help="Path to model checkpoint exported as jit.")


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Lab app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import carb
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.sim import SimulationContext, PhysicsMaterialCfg
from isaaclab.utils.math import quat_from_euler_xyz

# Isaac Lab pretrained spot policy 
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import SpotFlatEnvCfg_PLAY


TASK = "Isaac-Velocity-Flat-Spot-v0"
RL_LIBRARY = "rsl_rl"

# Local imports
from utils.server import run_server, format_data

# Parse robot position
robot_pos = [float(x) for x in args_cli.robot_pos.split(',')]

# Global variables
first_step = True
reset_needed = False
base_command = np.zeros(3)

# Shared buffers for server callbacks
_latest_rgb = None
_latest_depth = None
_latest_position = None
_latest_quat_wxyz = None

# Spot robot configuration for Isaac Lab
@configclass
class SpotRobotCfg(ArticulationCfg):
    """Configuration for Spot robot in Isaac Lab"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prim_path = "/World/Robot"
        self.spawn = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/BostonDynamics/spot/spot.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, 
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
        )
        self.init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(robot_pos),
            joint_pos={
                "fl_kn": -0.8,
                "fr_kn": -0.8,
                "hl_kn": -0.8,
                "hr_kn": -0.8,
            },  
        )
        self.actuators = {}



@configclass
class EnvCfg(SpotFlatEnvCfg_PLAY):
    def __init__(self, num_envs=1):
        super().__init__()
        

# Socket server integration (control Spot via external commands)
def action_callback(msg_type, message):
    global base_command
    # Expect messages of type 'VEL' with fields x, y, omega
    if msg_type == 'VEL':
        base_command = np.array([float(message.x), float(message.y), float(message.omega)])
        print("base_command: ", base_command)

def data_callback():
    global _latest_rgb, _latest_depth, _latest_position, _latest_quat_wxyz
    if _latest_rgb is None or _latest_depth is None or _latest_position is None or _latest_quat_wxyz is None:
        print("Data missing")
        return None
    return format_data(_latest_rgb, _latest_depth, _latest_position, _latest_quat_wxyz)

def planner_callback():
    # No onboard planner state here
    return {}

# Start socket server in background
server_thread = Thread(target=run_server, kwargs={
    "data_cb": data_callback, 
    "action_cb": action_callback, 
    "planner_cb": planner_callback
})
server_thread.daemon = True
server_thread.start()

# Main simulation loop

# Initialize the simulation context
# sim_cfg = sim_utils.SimulationCfg(device="cuda:0")  # Default to CUDA device
# sim = sim_utils.SimulationContext(sim_cfg)

# Initialize scene
# scene_cfg = SpotUSDSceneCfg(num_envs=args_cli.num_envs)
# scene = InteractiveScene(scene_cfg)

# Play the simulator
# sim.reset()
# Now we are ready!
print("[INFO]: Setup complete...")

# sim_dt = sim.get_physics_dt()

""" new """
# load the trained jit policy
policy_path = os.path.abspath(args_cli.checkpoint)
file_content = omni.client.read_file(policy_path)[2]
file = io.BytesIO(memoryview(file_content).tobytes())
policy = torch.jit.load(file, map_location=args_cli.device)

# setup environment
env_cfg = EnvCfg(num_envs=args_cli.num_envs)
env_cfg.curriculum = None
env_cfg.sim.device = args_cli.device
# env_cfg.scene.terrain = TerrainImporterCfg(
#             prim_path="/World/ground",
#             terrain_type="usd",
#             usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
#         )
# print("env_cfg")
print("[INFO]: Using Device ", args_cli.device)
env = ManagerBasedRLEnv(cfg=env_cfg)
print("env.num_envs:", env.num_envs)
obs, _ = env.reset()
print("[INFO]: Env setup complete...")
""" end of new """

"""Main simulation loop"""
while simulation_app.is_running():

    if first_step or reset_needed:
        # Initialize robot on first step
        # root_spot_state = env.scene["robot"].data.default_root_state.clone()
        # print("Robot root state: ", root_spot_state)
        # root_spot_state[:, :3] += env.scene.env_origins

        # env.scene["robot"].write_root_pose_to_sim(root_spot_state[:, :7])
        # env.scene["robot"].write_root_velocity_to_sim(root_spot_state[:, 7:])

        # joint_pos, joint_vel = (
        #     env.scene["robot"].data.default_joint_pos.clone(),
        #     env.scene["robot"].data.default_joint_vel.clone(),
        # )
        # env.scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
        env.reset()
        first_step = False
        reset_needed = False
        print(f"[INFO]: Resetting robot state..")
    try:

        """ new """
        # Use policy instead of root velocity hack
        # Construct command input (normalized if needed)
        command = torch.tensor([base_command[0], base_command[1], 0, base_command[2]], device="cuda:0", dtype=torch.float32)

        # Policy forward pass
        with torch.inference_mode():
            print("in update")
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)
            obs["policy"][0, 9:13] = command

            print("aaa")

        

        # Update simulation + scene (already inside env.step)
        # sim.step()
        # scene.update(sim_dt)
        """ end of new """
        # # Convert base_command [x, y, omega] to linear and angular velocity
        # linear_vel = np.array([base_command[0], base_command[1], 0.0])  # x, y, z
        # angular_vel = np.array([0.0, 0.0, base_command[2]])  # roll, pitch, yaw
        
        # # Create a batched velocity tensor
        # velocity_tensor = torch.tensor([
        #     linear_vel[0], linear_vel[1], linear_vel[2],
        #     angular_vel[0], angular_vel[1], angular_vel[2]
        # ], device=scene["spot"].device, dtype=torch.float32).unsqueeze(0)

        # scene["spot"].write_root_velocity_to_sim(velocity_tensor)
        # print(f"Applied command: linear_vel={linear_vel[:2]}, angular_vel={angular_vel[2]}")
            
        # # Update simulation
        # scene.write_data_to_sim() 
        # sim.step()
        # scene.update(sim_dt) 

        # --- Capture camera data and robot pose ---
        # Get camera data
        camera = env.scene["camera"] 
        rgb_t = camera.data.output["rgb"]
        # print("-------------------------------")
        # print(env.scene["camera"])
        # print("Received shape of rgb image: ", rgb_t.shape)
        rgb_np = rgb_t[0].cpu().numpy()[..., :3].astype(np.uint8)
        
        # Convert depth to millimeters
        depth_m_t = camera.data.output["distance_to_image_plane"]
        depth_m_np = depth_m_t[0].cpu().numpy()[..., :3]
        depth_mm = np.nan_to_num(depth_m_np, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
        depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        
        # Get robot position and orientation from Isaac Lab articulation
        # This data is already batched, so we select the first environment's data
        position = env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy().astype(np.float32)
        quat_wxyz = env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
        print("-------------------------------")
        print("Received spot position: ", position)
        print("Received spot rot: ", quat_wxyz)
        
        # Update shared buffers
        if(_latest_rgb is None or _latest_depth is None or _latest_position is None or _latest_quat_wxyz is None):
            print("Data missing")
        _latest_rgb = rgb_np
        _latest_depth = depth_uint16
        _latest_position = position
        _latest_quat_wxyz = quat_wxyz

    except Exception as e:
        print(f"Error in main loop: {e}")

simulation_app.close()