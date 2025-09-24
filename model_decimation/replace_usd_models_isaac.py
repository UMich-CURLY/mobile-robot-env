# python /home/junzhewu/pohsun/SG-VLN/robot_env/model_decimation/replace_usd_models_isaac.py -- --input_folder /path/to/folder

## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu

import os
import sys
import argparse
from pxr import Usd, UsdGeom, UsdShade, Sdf, Tf
import time

def count_total_faces_in_mesh_prim(prim):
    total_faces = 0
    mesh = UsdGeom.Mesh(prim)
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    if face_counts is not None:
        total_faces += len(face_counts)
    return total_faces

def swap_mesh_geometry(original_usd_path, decimated_usd_path, output_usd_path):

    stage_orig = Usd.Stage.Open(original_usd_path)
    try:
        stage_dec = Usd.Stage.Open(decimated_usd_path)
    except Exception as e:
        print(f"[WARNING] Failed to open decimated USD: {decimated_usd_path}\n  {e}\n  Skipping this instance.")
        return

    # Build a name-to-prim map for decimated meshes
    dec_mesh_map = {}
    for prim in stage_dec.Traverse():
        dec_mesh_map[prim.GetName()] = prim

    # For each mesh in the original, if a decimated mesh with the same name exists, swap geometry
    total_faces_original = 0
    total_faces_swapped = 0
    for prim in stage_orig.Traverse():
        
        total_faces_original += count_total_faces_in_mesh_prim(prim)

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
        total_faces_swapped += count_total_faces_in_mesh_prim(prim)

    print(f"[INFO] Total faces in original meshes: {total_faces_original}")
    print(f"[INFO] Total faces in swapped meshes: {total_faces_swapped}")

    # Save the modified original stage as the output
    stage_orig.GetRootLayer().Export(output_usd_path)
    print(f"✅ Exported USD with original hierarchy/materials and swapped mesh geometry: {output_usd_path}")


def swap_all_decimated_meshes_in_folder(models_folder):
    """
    Iterate through all subfolders in models_folder, find instance.usd and
    corresponding instance_decimated.usd, and swap mesh geometry.
    """
    processed_count = 0
    
    for root, dirs, files in os.walk(models_folder):
        # Only process folders containing 'instance_renamed.usd' and 'instance_renamed_decimated.usd'
        if 'instance_renamed_decimated.usd' in files and 'instance_renamed.usd' in files:
            original_usd = os.path.join(root, 'instance_renamed.usd')
            decimated_usd = os.path.join(root, 'instance_renamed_decimated.usd')
            output_usd = os.path.join(root, 'instance.usd')

            if os.path.exists(decimated_usd):
                print(f"[INFO] Swapping meshes: {decimated_usd} -> {original_usd}")
                swap_mesh_geometry(original_usd, decimated_usd, output_usd)
                processed_count += 1
            else:
                print(f"[WARNING] No decimated USD found for {original_usd}, skipping.")
    
    print(f"[INFO] Processing complete. Total USD files processed: {processed_count}")

# --- SCRIPT USAGE EXAMPLE ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Replace mesh geometry in USD files with decimated versions")
    parser.add_argument("--input_folder", 
                        type=str, 
                        required=True,
                        help="Path to the input folder containing USD files to process")
    
    args = parser.parse_args()
    
    # Use parsed arguments
    input_folder = args.input_folder
    
    print(f"[INFO] Input folder: {input_folder}")
    print(f"[INFO] Starting mesh replacement process...")
    
    # Validate input folder exists
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    # Main function call
    swap_all_decimated_meshes_in_folder(input_folder)