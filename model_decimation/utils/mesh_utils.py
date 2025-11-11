import os
from pxr import Usd, UsdGeom, UsdShade, Sdf, Tf

def count_total_faces_in_mesh_prim(prim):
    total_faces = 0
    mesh = UsdGeom.Mesh(prim)
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    if face_counts is not None:
        total_faces += len(face_counts)
    return total_faces

def get_parent_prim_name(prim):
    """Get the parent prim name."""
    parent_path = prim.GetPath().GetParentPath()
    if parent_path == Sdf.Path.absoluteRootPath:
        return None  # No parent (root prim)
    
    parent_prim = prim.GetStage().GetPrimAtPath(parent_path)
    return parent_prim.GetName() if parent_prim else None