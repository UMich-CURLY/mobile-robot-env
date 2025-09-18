import os
from pxr import Usd, UsdGeom, UsdShade, Sdf, Tf

def swap_mesh_geometry(original_usd_path, decimated_usd_path, output_usd_path):

    stage_orig = Usd.Stage.Open(original_usd_path)
    stage_dec = Usd.Stage.Open(decimated_usd_path)

    # Build a name-to-prim map for decimated meshes
    dec_mesh_map = {}
    for prim in stage_dec.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            dec_mesh_map[prim.GetName()] = prim

    # For each mesh in the original, if a decimated mesh with the same name exists, swap geometry
    for prim in stage_orig.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_name = prim.GetName()
            if mesh_name in dec_mesh_map:
                dec_prim = dec_mesh_map[mesh_name]
                # Copy geometry attributes
                for attr_name in [
                    "points", "normals", "faceVertexIndices", "faceVertexCounts",
                    "primvars:normals", "primvars:st", "primvars:uv"
                ]:
                    orig_attr = prim.GetAttribute(attr_name)
                    dec_attr = dec_prim.GetAttribute(attr_name)
                    if orig_attr and dec_attr and dec_attr.HasAuthoredValue():
                        orig_attr.Set(dec_attr.Get())
    # Save the modified original stage as the output
    stage_orig.GetRootLayer().Export(output_usd_path)
    print(f"✅ Exported USD with original hierarchy/materials and swapped mesh geometry: {output_usd_path}")


def count_total_faces_in_instance(stage):
    total_faces = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).startswith("/Root/Instance"):
            mesh = UsdGeom.Mesh(prim)
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            if face_counts is not None:
                total_faces += len(face_counts)
    return total_faces


# --- Example usage ---
if __name__ == "__main__":

    original_usd = "/home/junzhewu/data/isaac_scenes_v1/grscenes_commercial/models/object/others/backpack/abd4eebb509a10a1c4c6719a8cca399d/instance.usd"
    decimated_usd = "/home/junzhewu/data/isaac_scenes_v1/grscenes_commercial/models/object/others/backpack/abd4eebb509a10a1c4c6719a8cca399d/test2.usd"
    output_usd = "/home/junzhewu/data/isaac_scenes_v1/grscenes_commercial/models/object/others/backpack/abd4eebb509a10a1c4c6719a8cca399d/test2_material_new.usd"

    # Print face count in original
    stage_orig = Usd.Stage.Open(original_usd)
    orig_faces = count_total_faces_in_instance(stage_orig)
    print(f"Original USD total face count under /Root/Instance: {orig_faces}")

    swap_mesh_geometry(original_usd, decimated_usd, output_usd)

    # Print face count in swapped output
    stage_out = Usd.Stage.Open(output_usd)
    out_faces = count_total_faces_in_instance(stage_out)
    print(f"Output USD total face count under /Root/Instance: {out_faces}")