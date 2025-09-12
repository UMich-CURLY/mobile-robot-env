import argparse
import os
import cv2
import time
import math
import gzip, json
import numpy as np
from scipy.spatial.transform import Rotation
# omni-isaaclab
from isaaclab.app import AppLauncher

import isaac.scripts.isaac_cli_args as isaac_cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Server for reading data from Isaac Lab agent and issuing action commands")
parser.add_argument("--episode_index", default=0, type=int, help="Episode index.")

parser.add_argument("--task", type=str, default="go2_matterport_vision", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--history_length", default=9, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")

# usd editing
parser.add_argument("--usd_path", type=str, required=True, help="Absolute path to the USD scene file to edit.")
parser.add_argument("--load_timeout", type=float, default=600.0, help="Max seconds to wait for /Looks or stage to finish loading.")

isaac_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
# parser.add_argument("--draw_pointcloud", action="store_true", default=False, help="DRaw pointlcoud.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
# import ipdb; ipdb.set_trace()
simulation_app = app_launcher.app

# pxr / USD
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import omni.usd


def _ensure_world_xform(stage: Usd.Stage, world_path: Sdf.Path) -> Usd.Prim:
    """Ensure that a `World` Xform exists and return its prim."""
    world_prim = stage.GetPrimAtPath(world_path)
    if not world_prim or not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, world_path).GetPrim()
    return world_prim


def _copy_looks_under_world(stage: Usd.Stage, src_looks_path: Sdf.Path, dst_looks_path: Sdf.Path) -> bool:
    """Copy the entire `Looks` scope subtree from stage root to under `/World`.

    This uses Sdf.CopySpec to duplicate the specs within the same root layer.
    Returns True if copied, False if source scope is missing.
    """
    src_prim = stage.GetPrimAtPath(src_looks_path)
    if not src_prim or not src_prim.IsValid():
        print(f"Source Looks scope not found at '{str(src_looks_path)}'; skipping copy.")
        return False

    root_layer = stage.GetRootLayer()
    # Copy the full subtree spec into the destination location
    Sdf.CopySpec(root_layer, src_looks_path, root_layer, dst_looks_path)
    return True


def _rebind_world_materials_to_world_looks(stage: Usd.Stage, world_path: Sdf.Path, src_looks_path: Sdf.Path, dst_looks_path: Sdf.Path) -> int:
    """Rebind direct material bindings under `/World` from `/Looks/...` to `/World/Looks/...`.

    Returns the number of prims whose bindings were updated.
    """
    updated_count = 0
    world_prim = stage.GetPrimAtPath(world_path)
    if not world_prim or not world_prim.IsValid():
        return 0

    for prim in Usd.PrimRange(world_prim):
        binding_api = UsdShade.MaterialBindingAPI(prim)
        direct_binding = binding_api.GetDirectBinding()
        bound_material = direct_binding.GetMaterial() if direct_binding else None
        if not bound_material:
            continue

        old_mat_prim = bound_material.GetPrim()
        if not old_mat_prim or not old_mat_prim.IsValid():
            continue

        old_mat_path = old_mat_prim.GetPath()
        if not old_mat_path.HasPrefix(src_looks_path):
            continue

        new_mat_path = old_mat_path.ReplacePrefix(src_looks_path, dst_looks_path)
        new_mat_prim = stage.GetPrimAtPath(new_mat_path)
        if not new_mat_prim or not new_mat_prim.IsValid():
            continue

        new_material = UsdShade.Material(new_mat_prim)
        # Re-bind with default strength/purpose; DirectBinding doesn't expose binding strength reliably
        binding_api.Bind(new_material)

        updated_count += 1

    return updated_count


def _export_with_suffix(stage: Usd.Stage, source_path: str, suffix: str = "_edit") -> str:
    base, ext = os.path.splitext(source_path)
    out_path = f"{base}{suffix}{ext or '.usd'}"
    stage.GetRootLayer().Export(out_path)
    return out_path


def remove_paint_tools(stage: Usd.Stage) -> int:
    """Remove all prims that look like paint tools.

    Heuristics:
    - Prim name or type name contains the substring "paint" (case-insensitive)
    - Exclude `UsdShade.Material` and `UsdShade.Shader` to avoid deleting materials/shaders
    """
    to_remove_paths = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim or not prim.IsValid():
            continue
        name_lower = prim.GetName().lower()
        type_lower = (prim.GetTypeName() or "").lower()
        if "painttool" in name_lower or "painttool" in type_lower:
            to_remove_paths.append(prim.GetPath())

    # Remove deepest paths first to avoid removing parents before children
    to_remove_paths.sort(key=lambda p: len(str(p).split('/')), reverse=True)
    for path in to_remove_paths:
        stage.RemovePrim(path)

    return len(to_remove_paths)


def scale_material_uvs(stage: Usd.Stage, factor: float = 100.0) -> int:
    """Multiply UV texture scale inputs across material shader networks by `factor`.

    Targets common patterns:
    - `UsdTransform2d.inputs:scale`
    - Fallback inputs often used for UV tiling: `uv_scale`, `uvScale`, `texture_scale`, `textureScale`,
      `UVScale`, `uvTiling`, `tiling`, `repeat`, `tile`, `scale_uv`, `scaleUV`.
    Returns the number of inputs updated.
    """
    updated_inputs = 0

    def _scale_value(val):
        if isinstance(val, Gf.Vec2f):
            return Gf.Vec2f(val[0] * factor, val[1] * factor)
        if isinstance(val, Gf.Vec3f):
            return Gf.Vec3f(val[0] * factor, val[1] * factor, val[2] * factor)
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            return (val[0] * factor, val[1] * factor)
        if isinstance(val, (int, float)):
            return val * factor
        return None

    uv_input_names = (
        "uv_scale",
        "uvScale",
        "texture_scale",
        "textureScale",
        "UVScale",
        "uvTiling",
        "tiling",
        "repeat",
        "tile",
        "scale_uv",
        "scaleUV",
    )

    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim or not prim.IsValid():
            continue
        if prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(prim)
        if not shader:
            continue

        shader_id = shader.GetIdAttr().Get()
        # Prefer canonical Transform2d nodes
        if shader_id and ("Transform2d" in str(shader_id) or str(shader_id).lower() == "usdtransform2d"):
            input_scale = shader.GetInput("scale")
            if input_scale:
                val = input_scale.Get()
                new_val = _scale_value(val)
                if new_val is not None:
                    input_scale.Set(new_val)
                    updated_inputs += 1

        # Fallback: common UV tiling/scaling parameters on various shader nodes
        for name in uv_input_names:
            inp = shader.GetInput(name)
            if not inp:
                continue
            val = inp.Get()
            new_val = _scale_value(val)
            if new_val is not None:
                inp.Set(new_val)
                updated_inputs += 1

    return updated_inputs


def wait_until_stage_is_fully_loaded(max_frames=10, frametime_threshold=0.1, time_ratio_treshold=5):
    prev_frametime = 0
    for i in range(max_frames):
        start_time = time.time()
        omni.kit.app.get_app().update()
        elapsed_time = time.time() - start_time
        print(f"Frame {i} frametime: {elapsed_time}")
        if elapsed_time < frametime_threshold or elapsed_time * time_ratio_treshold < prev_frametime:
            print(f"Stage fully loaded at frame {i}, last frametime: {elapsed_time}")
            break
        prev_frametime = elapsed_time

from isaacsim.core.utils.stage import is_stage_loading

def scale_scene(stage: Usd.Stage, world_path: Sdf.Path, scale: float = 0.01) -> bool:
    """Apply a uniform scale to the `/World` Xform so the entire scene scales.

    Creates or updates an `xformOp:scale` on `/World` to the given uniform value.
    Returns True if the scale was applied.
    """
    world_prim = stage.GetPrimAtPath(world_path)
    if not world_prim or not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, world_path).GetPrim()
    xform = UsdGeom.Xform(world_prim)
    scale_op = None
    for op in xform.GetOrderedXformOps():
        try:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        except Exception:
            continue
    if scale_op is None:
        scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(scale, scale, scale))
    return True

def run_update():
    usd_path = args_cli.usd_path
    if not usd_path:
        raise RuntimeError("--usd_path must be provided")

    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)

    # simulation_app.update()
    # simulation_app.update()

    # while is_stage_loading():
    #     print("asset still loading, waiting to finish")
    #     simulation_app.update()

    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError(f"Failed to open stage: {usd_path}")
    else:
        print("Stage loaded successfully")

    # for i in range(100):
    #     simulation_app.update()
    #     time.sleep(0.1)

    src_looks_path = Sdf.Path("/Looks")
    world_path = Sdf.Path("/World")
    dst_looks_path = Sdf.Path("/World/Looks2")

    # Remove paint tools before making structural/material changes
    removed_tools = remove_paint_tools(stage)
    if removed_tools>0:
        print(f"Removed {removed_tools} paint tool prim(s).")
    else:
        print("No paint tool prims found.")

    _ensure_world_xform(stage, world_path)
    if scale_scene(stage, world_path, scale=0.01):
        print("Applied world scale 0.01 to /World.")
    did_copy_looks = _copy_looks_under_world(stage, src_looks_path, dst_looks_path)
    updated = 0
    if did_copy_looks:
        updated = _rebind_world_materials_to_world_looks(stage, world_path, src_looks_path, dst_looks_path)
        print(f"Updated material bindings on {updated} prim(s) under /World.")
    else:
        print("Skipped rebind: no /Looks to copy.")

    # Scale UV texture transforms across all materials
    uv_updates = scale_material_uvs(stage, factor=100.0)
    print(f"Scaled UV texture inputs on {uv_updates} shader input(s).")

    out_path = _export_with_suffix(stage, usd_path, suffix="_edit")
    print(f"Saved edited stage to: {out_path}")


if __name__ == "__main__":
    try:
        run_update()
    except Exception as e:
        print(e)
        simulation_app.close()
    finally:
        simulation_app.close()