
import argparse
import numpy as np
import torch
import time

# start simulation
from isaaclab.app import AppLauncher
import utils.rsl_rl_cli_args as rsl_rl_cli_args
import utils.vln_args as vln_cli_args

# Add command line arguments
parser = argparse.ArgumentParser(description="Benchmark")
rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = vln_cli_args.parse_args(parser)

# Launch Isaac Lab app
sim_start_time = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Enable Extension and setup settings
import carb
import omni.kit.app
# from isaacsim.core.utils.extensions import enable_extension
# enable_extension("omni.anim.navigation.bundle")
settings = carb.settings.get_settings()
settings.set("/renderer/multiGPU/enabled", False)
settings.set("/renderer/activeGpu", 0)
settings.set("/renderer/shadercache/driverDiskCache/enabled", True)
# no RTX or rendering
settings.set("/rtx/enabled", False)
settings.set("/rtx/post/aa/enabled", False)
settings.set("/rendering/enabled", False)
# Disable texture streaming entirely
settings.set("/rtx/texturestreaming/gpuBudgetPriority", 0.0)

# physics cooking OFF
settings.set("/physics/cooking/ujitsoCollisionCooking", False)

# reduce USD updates
settings.set("/physics/updateToUsd", False)
settings.set("/physics/updateVelocitiesToUsd", False)
settings.set("/physics/updateParticlesToUsd", False)
settings.set("/physics/updateForceSensorsToUsd", False)
settings.set("/physics/updateResidualsToUsd", False)

omni.kit.app.get_app().update()
import isaaclab.sim as sim_utils

# Local imports
from utils.sim import VLNSim

# setup simulation
default_episode = "test_generator_0"
vln_sim = VLNSim(args)
vln_sim.load_episode(default_episode)

# Setup UI
from utils.ui import BenchmarkUI
benchmark_ui = BenchmarkUI(vln_sim)
benchmark_ui.update_ui("episode_label", default_episode)
vln_sim.on_client_episode_changed(lambda episode_label: benchmark_ui.update_ui("episode_label", episode_label))

"""Main simulation loop"""
print(f"[INFO] Starting simulation took {time.time()-sim_start_time:.2f}s")
print("[INFO] Starting benchmark loop")
start_time = time.time()
start_episode_step = 0
frame_count = 0
end_time = 0
while simulation_app.is_running():
    vln_sim.step()
    # print infos
    frame_count += 1
    if frame_count == 1:
        print(f"[INFO] Loading default scene took {time.time() - start_time:.2f}s")
        pass
    log_fps_interval = 100
    if frame_count % log_fps_interval == 0:
        duration = time.time() - start_time
        sim_time = int(vln_sim.manager_env.common_step_counter - start_episode_step) * vln_sim.manager_env.step_dt
        print(f"[INFO] Frame count: {frame_count}, Time: {duration:.2f}s, FPS: {log_fps_interval / duration:.2f}, Sim Time: {sim_time:.2f}s ({sim_time/duration:.2f}x), step_dt={vln_sim.manager_env.step_dt}, count={vln_sim.manager_env.common_step_counter}")
        start_time = time.time()
        start_episode_step = int(vln_sim.manager_env.common_step_counter)

simulation_app.close()