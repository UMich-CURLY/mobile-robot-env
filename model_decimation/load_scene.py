from isaaclab.app import AppLauncher
# Launch Isaac Sim
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import omni.usd
import omni.kit.app
import carb.settings
import time
import os
from pxr import Usd, PhysxSchema

kit = omni.kit.app.get_app()

# Disable automatic physics / collision during load
settings = carb.settings.get_settings()
root = settings.get_settings_dictionary("/hydra")
with open("hydra_settings.txt", "w") as f:
    f.write(str(root))
settings.set("/renderer/multiGPU/enabled", False)
settings.set("/renderer/activeGpu", 0)
settings.set("/renderer/shadercache/driverDiskCache/enabled", True)

# no RTX or rendering
settings.set("/rtx/enabled", False)
settings.set("/rtx/post/aa/enabled", False)
settings.set("/rendering/enabled", False)

# physics cooking OFF
settings.set("/physics/cooking/ujitsoCollisionCooking", False)

# reduce USD updates
settings.set("/physics/updateToUsd", False)
settings.set("/physics/updateVelocitiesToUsd", False)
settings.set("/physics/updateParticlesToUsd", False)
settings.set("/physics/updateForceSensorsToUsd", False)
settings.set("/physics/updateResidualsToUsd", False)

# Non-flattened USD path
usd_path = os.path.expanduser("~/scratch/isaac_scenes_v1/grscenes_commercial/scenes/MV4AFHQKTKJZ2AABAAAAADQ8_usd/start_result_navigation.usd")

ctx = omni.usd.get_context()

start_time = time.time()

# Open USD (blocking for initial parse)
ctx.open_stage(usd_path)


load_duration = time.time() - start_time
print(f"Scene fully loaded in {load_duration:.2f} seconds")


# Keep the app running
while simulation_app.is_running():
    simulation_app.update()
