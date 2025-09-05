# /isaac-sim/python.sh robot_env/isaac_server_spot.py
# /isaac-sim/kit/python/bin/python3 -m pip install jsonpickle

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard
# import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera

from threading import Thread
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import omni.replicator.core as rep

from robot_env.utils.server import run_server, format_data

first_step = True
reset_needed = False

# initialize robot on first step, run robot advance
def on_physics_step(step_size) -> None:
    global first_step
    global reset_needed
    if first_step:
        spot.initialize()
        first_step = False
    elif reset_needed:
        my_world.reset(True)
        reset_needed = False
        first_step = True
    else:
        spot.forward(step_size, base_command)


# spawn world
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# spawn scene
prim = define_prim("/World/Ground", "Xform")
asset_path = "/data/isaac_scenes_v1/nvidia_flatten/park_morning/park_morning_edit.usd"
prim.GetReferences().AddReference(asset_path)
prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(0.01, 0.01, 0.01))
# collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
# sim_utils.define_collision_properties(prim.GetPrimPath(), collider_cfg)

# spawn robot
spot = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=np.array([-52.8, 11.5, 0.8]),
)
my_world.reset()
my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

# robot command
base_command = np.zeros(3)

# --- Attach a third person camera to the robot ---
camera_path = "/World/Spot/body/ThirdPersonCamera"
define_prim(camera_path, "Camera")
cam_prim = get_prim_at_path(camera_path)
cam_xform_api = UsdGeom.XformCommonAPI(cam_prim)
cam_xform_api.SetTranslate(Gf.Vec3d(-5.0, 0.0, 0.0))
cam_xform_api.SetRotate(Gf.Vec3f(90.0, 0.0, -90.0))

# --- Attach a camera to the robot and set up capture ---
camera_path = "/World/Spot/body/Camera"
define_prim(camera_path, "Camera")
cam_prim = get_prim_at_path(camera_path)
cam_xform_api = UsdGeom.XformCommonAPI(cam_prim)
cam_xform_api.SetTranslate(Gf.Vec3d(-1.0, 0.0, 0.5))
cam_xform_api.SetRotate(Gf.Vec3f(90.0, 0.0, -90.0))



# Create a camera
camera_resolution = (640, 480)
cam = Camera(
    camera_path,
    name="RobotCamera",
    frequency=10,
    resolution=camera_resolution,
)
cam.initialize()
cam.add_distance_to_camera_to_frame()

# Shared buffers for server callbacks
_latest_rgb = None
_latest_depth = None
_latest_position = None
_latest_quat_wxyz = None

# socket server integration (control Spot via external commands)
def action_callback(msg_type, message):
    global base_command
    # Expect messages of type 'VEL' with fields x, y, omega
    if msg_type == 'VEL':
        base_command = np.array([float(message['x']), float(message['y']), float(message['omega'])])
        print("base_command: ", base_command)

def data_callback():
    global _latest_rgb, _latest_depth, _latest_position, _latest_quat_wxyz
    if _latest_rgb is None or _latest_depth is None or _latest_position is None or _latest_quat_wxyz is None:
        return None
    return format_data(_latest_rgb, _latest_depth, _latest_position, _latest_quat_wxyz)

def planner_callback():
    # No onboard planner state here
    return {}

# start socket server in background
server_thread = Thread(target=run_server, kwargs={"data_cb": data_callback, "action_cb": action_callback, "planner_cb": planner_callback})
server_thread.daemon = True
server_thread.start()

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped():
        reset_needed = True
    # Capture camera data and robot pose after each rendered step
    try:
        rgb = cam.get_rgba()[:, :, :3]
        depth_m = cam.get_current_frame()["distance_to_camera"]
        if rgb is not None and depth_m is not None:
            depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
            depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)

            spot_prim = get_prim_at_path("/World/Spot")
            xform = UsdGeom.Xformable(spot_prim)
            m = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = m.ExtractTranslation()
            r = m.ExtractRotation()
            q = r.GetQuat()
            quat_wxyz = np.array([q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]], dtype=np.float32)
            position = np.array([t[0], t[1], t[2]], dtype=np.float32)
            print("-------------------------------")
            print("Received spot position: ", position)
            print("Received spot rot: ", quat_wxyz)
            _latest_rgb = rgb[..., :3].astype(np.uint8)
            _latest_depth = depth_uint16
            _latest_position = position
            _latest_quat_wxyz = quat_wxyz
    except Exception as e:
        print(e)
    # if my_world.is_playing():
    #     if i >= 0 and i < 80:
    #         # forward
    #         base_command = np.array([2, 0, 0])
    #     elif i >= 80 and i < 130:
    #         # rotate
    #         base_command = np.array([1, 0, 2])
    #     elif i >= 130 and i < 200:
    #         # side ways
    #         base_command = np.array([0, 1, 0])
    #     elif i == 200:
    #         i = 0
    #     i += 1

simulation_app.close()
