from isaaclab.app import AppLauncher
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import omni.usd

usd_path = "~/scratch/isaac_scenes_v1/grscenes_commercial/scenes/MV4AFHQKTKJZ2AABAAAAADQ8_usd/start_result_navigation.usd"
ctx = omni.usd.get_context()
ctx.open_stage(usd_path)  # original terrain file

stage = ctx.get_stage()

# output path
output = "~/scratch/isaac_scenes_v1/grscenes_commercial/scenes/MV4AFHQKTKJZ2AABAAAAADQ8_usd/start_result_navigation_flat.usd"

omni.usd.flatten(stage, output)
print("Flattened USD saved to", output)
