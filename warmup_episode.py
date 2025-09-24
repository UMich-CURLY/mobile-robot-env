"""This script is used to warm-up the simulation.
NOTE:
    If the local disk space or the preset local mesh cache size is not enough,
    the warm-up process is likely to fail.

python warmup.py --dirs /data/isaac_scenes_v1 --enable_cameras
"""
import argparse
import os
import time
import torch
import tqdm

parser = argparse.ArgumentParser(description='Warmup the simulation.')
parser.add_argument(
    '-r',
    '--reset',
    required=False,
    action='store_true',
    help='If specified, it will release the old local mesh cache first.',
)
parser.add_argument('-f', '--files', required=False, nargs='*', help='the usd file(s) to warmup')
parser.add_argument('-d', '--dirs', required=False, nargs='*', help='the folders including usd file(s) to warmup')

import utils.rsl_rl_cli_args as rsl_rl_cli_args  # isort: skip
import utils.vln_args as vln_cli_args
from isaaclab.app import AppLauncher


rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Check cli args
if not args.files and not args.dirs:
    print('No path specified!')
    exit(1)


log_file = f"warmup_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
if os.path.exists(log_file):
    os.remove(log_file)

# SimulationApp should be initialized first before importing omni plugins
app_launcher = AppLauncher(args)
kit = app_launcher.app


from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
print(f"ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")

import omni.physx
import omni.physx.bindings._physx as physx_bindings
from isaaclab.sim import SimulationContext
from pxr import UsdUtils
import isaaclab.sim as sim_utils


kit.set_setting(physx_bindings.SETTING_UJITSO_COLLISION_COOKING, False)
kit.set_setting(physx_bindings.SETTING_USE_LOCAL_MESH_CACHE, True)
kit.set_setting(physx_bindings.SETTING_LOCAL_MESH_CACHE_SIZE_MB, 1024*100)

import carb, os
settings = carb.settings.get_settings()
settings.set("/renderer/multiGPU/enabled", True)
settings.set("/renderer/activeGpu", 0)
settings.set("/rtx/post/dlss/execMode", 1) # 0: Performance, 1: Balanced, 2: Quality, 3: Auto


ujitso_cooking_enabled = kit._carb_settings.get_as_bool(physx_bindings.SETTING_UJITSO_COLLISION_COOKING)
use_local_cache = kit._carb_settings.get_as_bool(physx_bindings.SETTING_USE_LOCAL_MESH_CACHE)
local_cache_MB = kit._carb_settings.get(physx_bindings.SETTING_LOCAL_MESH_CACHE_SIZE_MB)

print(f'========== ujitso_cooking_enabled: {ujitso_cooking_enabled} ==========')
print(f'========== use_local_cache: {use_local_cache} ==========')
print(f'========== local_cache_MB: {local_cache_MB} ==========')
# with open(log_file, "a+") as f:
#     f.write(f"========== ujitso_cooking_enabled: {ujitso_cooking_enabled} ==========\n")
#     f.write(f"========== use_local_cache: {use_local_cache} ==========\n")
#     f.write(f"========== local_cache_MB: {local_cache_MB} ==========\n")


from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from omni.kit.viewport.utility import get_viewport_from_window_name
from pathlib import Path
import isaacsim.core.utils.prims as prims_utils

# Local imports
from utils.episode import VLNEpisodes
from utils.vln_env_wrapper import VLNEnvWrapper

from robot.spot_flat_env_cfg import SpotFlatEnvCfg_PLAY

from utils.server import run_server, format_data
from utils.innout_sim import InNOutSim

TASK = "Isaac-Velocity-Flat-Spot-v0"
RL_LIBRARY = "rsl_rl"

# Main simulation loop

# load episodes
episode_list = VLNEpisodes()
current_episode = episode_list[0]

# setup environment
env_cfg = SpotFlatEnvCfg_PLAY()
# env_cfg.load_usd(scene_abs_path)

env_cfg.scene.robot.init_state.pos = current_episode["start_position"]
env_cfg.scene.robot.init_state.rot = current_episode["start_rotation"]      
env_cfg.scene.num_envs = args.num_envs
# env_cfg.viewer.cam_prim_path = '/World/pov_camera'

env_cfg.sim.device = args.device
env_cfg.curriculum = None
manager_env = ManagerBasedRLEnv(cfg=env_cfg)

agent_cfg: RslRlOnPolicyRunnerCfg = rsl_rl_cli_args.parse_rsl_rl_cfg(TASK, args)
env = RslRlVecEnvWrapper(manager_env)
ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
ppo_runner.load(checkpoint)
policy = ppo_runner.get_inference_policy(device=args.device)

all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "SoftSPL", "OracleNavigationError", "OracleSuccess"]
env = VLNEnvWrapper(args, env, policy, "spot", current_episode, measure_names=all_measures)
print("[INFO] Env setup complete")

in_n_out_sim = InNOutSim(args, env)

obs, _ = env.reset()
for frame_count in range(50):
    _ = env.step(in_n_out_sim.commands)

def test_single(scene_abs_path):
    """Main simulation loop"""
    # remove the old terrain
    # terrain_prim = manager_env.scene.stage.GetPrimAtPath('/World/ground/terrain')
    manager_env.scene.stage.RemovePrim(manager_env.scene.terrain.terrain_prim_paths[0])
    manager_env.scene.terrain.terrain_prim_paths = []
    with open(log_file, "a+") as f:
        f.write(f"Removing old terrain\n")
    # with torch.inference_mode():
    #     env.step(in_n_out_sim.commands)
    #     env.step(in_n_out_sim.commands)
    
    start_time = 0
    total_start_time = time.time()
    print("[INFO]: Starting simulation")

    # prims_utils.delete_prim(terrain_prim.GetPath())
    with open(log_file, "a+") as f:
        f.write(f"Loading new terrain\n")
    manager_env.scene.terrain.import_usd("terrain", scene_abs_path)
    sim_init = False
    for frame_count in range(100):
        with torch.inference_mode():
            if not sim_init:
                obs, _ = env.reset()
                sim_init = True
                print(f"[INFO]: Resetting robot state..")
            obs, reward, done, info = env.step(in_n_out_sim.commands)
        if frame_count==50:
            start_time = time.time()
    with open(log_file, "a+") as f:
        frame_count = 50
        f.write(f"  Frame count: {frame_count}")
        f.write(f"  Simulation time: {time.time() - start_time:.2f}s\n ")
        f.write(f"  FPS: {frame_count / (time.time() - start_time):.2f}\n")
        f.write(f"  Total time: {time.time() - total_start_time:.2f}s\n")
    # kit.close()

def test_all(scene_abs_folder):
    # scene_dirs = [_ for _ in os.listdir(scene_abs_folder) if os.path.isdir(os.path.join(scene_abs_folder, _))]
    # for scene_dir in tqdm.tqdm(scene_dirs):
    #     navigation_path = os.path.join(scene_abs_folder, scene_dir, 'start_result_navigation.usd')
    #     interaction_path = os.path.join(scene_abs_folder, scene_dir, 'start_result_interaction.usd')
    #     test_single(navigation_path)
    #     test_single(interaction_path)
    vc_folders = os.path.join(scene_abs_folder, 'virtual_community')
    vc_scenes = [_ for _ in os.listdir(vc_folders) if os.path.isdir(os.path.join(vc_folders, _))]
    vc_scenes = [os.path.join(vc_folders, f"{_}/{_}.usd") for _ in vc_scenes]
    print("========== vc_scenes: ==========\n"+'\n'.join(vc_scenes))
    nvidia_scenes = [
        "nvidia/AECDemo_NVD@10012/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Morning(20Gb).usd",
        "nvidia/AECDemo_NVD@10012/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Night(20Gb).usd",
        "nvidia/AECDemo_NVD@10012/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Brownstone(8Gb).usd",
        "nvidia/AECO_CityDemoPack_NVD@10011/Demos/AEC/TowerDemo/CityDemopack/World_CityDemopack.usd",
        "nvidia/AECO_TowerDemoPack_NVD@10012/Demos/AEC/TowerDemo/TowerDemopack/World_TowerDemopack.usd",
    ]
    nvidia_folder = [os.path.join(scene_abs_folder, _) for _ in nvidia_scenes]
    print("========== nvidia_folder: ==========\n"+'\n'.join(nvidia_folder))
    gr_commercial_folder = os.path.join(scene_abs_folder, 'grscenes_commercial/scenes/')
    gr_commercial_scenes = [_ for _ in os.listdir(gr_commercial_folder) if os.path.isdir(os.path.join(gr_commercial_folder, _))]
    gr_commercial_scenes = [os.path.join(gr_commercial_folder, _+'/start_result_navigation.usd') for _ in gr_commercial_scenes]
    print("========== gr_commercial_scenes: ==========\n"+'\n'.join(gr_commercial_scenes))
    gr_home_folder = os.path.join(scene_abs_folder, 'grscenes_home/scenes/')
    gr_home_scenes = [_ for _ in os.listdir(gr_home_folder) if os.path.isdir(os.path.join(gr_home_folder, _))]
    gr_home_scenes = [os.path.join(gr_home_folder, _+'/start_result_navigation.usd') for _ in gr_home_scenes]
    print("========== gr_home_scenes: ==========\n"+'\n'.join(gr_home_scenes))
    all_scenes = vc_scenes + nvidia_folder + gr_commercial_scenes + gr_home_scenes
    for scene in tqdm.tqdm(all_scenes):
        with open(log_file, "a+") as f:
            f.write(f"Processing {scene} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  File exists: {os.path.exists(scene)}\n")
            f.flush()
            if os.path.exists(scene):
                try:
                    test_single(scene)
                except Exception as e:
                    f.write(f"  Error: {e}\n")
                    import traceback
                    traceback.print_exc()
                    f.write(f"  Traceback: {traceback.format_exc()}\n")
            f.write(f"\n")
if __name__ == '__main__':
    # Release the old local mesh cache if specified
    if args.reset:
        omni.physx.get_physx_cooking_interface().release_local_mesh_cache()

    if args.files:
        for f in args.files:
            f_abs_path = os.path.abspath(f)
            if not os.path.exists(f_abs_path):
                print(f'Error! {f} not found!')
            else:
                test_single(f_abs_path)

    if args.dirs:
        for d in args.dirs:
            d_abs_path = os.path.abspath(d)
            if not os.path.exists(d_abs_path):
                print(f'Error! {d} not found!')
            else:
                test_all(d_abs_path)
    app_launcher.close()