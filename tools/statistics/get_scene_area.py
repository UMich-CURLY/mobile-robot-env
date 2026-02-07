# python get_scene_area.py
import os
import numpy as np
import logging
# -----------------------------
# HELPER FUNCTION
# -----------------------------
from pxr import Usd, UsdGeom, Gf
log_file = "output_prims.log"

# minimal logger setup
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def get_prim_bbox(usd_file, prim_path):
    stage = Usd.Stage.Open(usd_file)
    prim = stage.GetPrimAtPath(prim_path)

    # Iterate over all prims recursively
    # for prim in stage.Traverse():
    #     logger.info(f"{prim.GetPath()} {prim.GetTypeName()}")


    if prim is None:
        raise ValueError(f"Prim {prim_path} not found")

    if not prim.IsA(UsdGeom.Imageable):
        raise ValueError(f"Prim {prim_path} is not geometry")

    imageable = UsdGeom.Imageable(prim)
    # Compute bounding box in world coordinates
    bbox = imageable.ComputeWorldBound(
        time=Usd.TimeCode.Default(),
        purpose1=UsdGeom.Tokens.default_,
        purpose2=UsdGeom.Tokens.render,
        purpose3=UsdGeom.Tokens.proxy
    )
    
    bbox_range = bbox.GetRange()
    min_pt = bbox_range.GetMin()  # Gf.Vec3d
    max_pt = bbox_range.GetMax()  # Gf.Vec3d
    return tuple(min_pt), tuple(max_pt)

# Usage
usd_file = "/home/junzhe_lighthouse/lighthouse/scratch/isaac_scenes_v1/grscenes_commercial/scenes/scene1/start_result_navigation.usd"
prim_path = "/Root/Meshes/Base/ground/model_92d1d4723ae0b159825921ffd7e39531_0/Instance"


min_bound, max_bound = get_prim_bbox(usd_file, prim_path)
print("Min:", min_bound)
print("Max:", max_bound)

