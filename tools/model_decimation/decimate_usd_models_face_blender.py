# blender --background --python /home/junzhewu/pohsun/SG-VLN/robot_env/model_decimation/decimate_usd_models_face_blender.py -- --input_folder "/home/junzhewu/pohsun/data_decimated/grscenes_commercial/models/" --ratio 0.1

## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu
import shutil
import sys
import os
print(sys.executable)
print(sys.version)
import site
sys.path.append(site.getusersitepackages())
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import json
import random
import bpy
import os
import argparse
import numpy as np
from tqdm import tqdm
from pxr import Usd, UsdGeom, UsdShade, Sdf, Tf, Vt, Gf
from utils.mesh_utils import count_total_faces_in_mesh_prim, get_parent_prim_name
from utils.uv_utils import apply_decimate_modifiers, apply_dissolve_modifiers, apply_remesh_modifiers, protect_uv_seams
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)  # also flush errors immediately


class SuppressBlenderOutput:
    def __enter__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

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
        if prim.IsA(UsdGeom.Mesh):
            parent_prim_name = get_parent_prim_name(prim)
            if parent_prim_name is not None:
                if str(parent_prim_name) in prim.GetName():
                    dec_mesh_map[parent_prim_name] = prim
                else:
                    dec_mesh_map[prim.GetName()] = prim

    # For each mesh in the original, if a decimated mesh with the same name exists, swap geometry
    total_faces_original = 0
    total_faces_swapped = 0
    to_remove_prims = []
    for prim in stage_orig.Traverse():
        total_faces_original += count_total_faces_in_mesh_prim(prim)

        mesh_name = prim.GetName()
        if mesh_name in dec_mesh_map:
            dec_prim = dec_mesh_map[mesh_name]
            for attr_name in [
                "points", "normals", "faceVertexIndices", "faceVertexCounts",
                "primvars:normals", "primvars:st", "primvars:uv"
            ]:
                orig_attr = prim.GetAttribute(attr_name)
                dec_attr = dec_prim.GetAttribute(attr_name)
                if orig_attr and dec_attr and dec_attr.HasAuthoredValue():
                    orig_attr.Set(dec_attr.Get())
            total_faces_swapped += count_total_faces_in_mesh_prim(prim)
        elif prim.IsA(UsdGeom.Mesh):
            # remove the prim
            to_remove_prims.append(prim.GetPath())
    
    for path in to_remove_prims:
        stage_orig.RemovePrim(path)

    # print(f"[INFO] Total faces in original meshes: {total_faces_original}")
    # print(f"[INFO] Total faces in swapped meshes: {total_faces_swapped}")

    # Save the modified original stage as the output
    stage_orig.GetRootLayer().Export(output_usd_path)
    # print(f"Exported USD with original hierarchy/materials and swapped mesh geometry: {output_usd_path}")

def export_usd(output_usd_path):
    """Export current Blender scene to USD."""
    try:
        with SuppressBlenderOutput():
            bpy.ops.wm.usd_export(
                filepath=output_usd_path,
                relative_paths=True,
                export_materials=False,
                export_animation=False,
                export_uvmaps=True,
                use_instancing=True,
                triangulate_meshes=True,
            )
        print(f"[INFO] Decimation and export complete: {output_usd_path}")
    except Exception as e:
        print(f"[ERROR] Export USD failed: {e}")

def decimate_usd_meshes(input_usd_path, output_usd_path, ratio=0.1):
    """
    Imports a USD file, decimates all meshes, and exports to a new USD file.
    
    Args:
        input_usd_path (str): The full path to the USD file to import.
        output_usd_path (str): The full path for the decimated USD file.
        ratio (float): The decimation ratio. A value of 0.1 means 10% of the original faces.
    """
    print(f"================================================")
    
    # Check if the input file exists
    if not os.path.exists(input_usd_path):
        print(f"Error: Input file not found at {input_usd_path}")
        return
    
    # Check if the output file exists
    if os.path.exists(output_usd_path):
        print(f"Output file found at {output_usd_path}, skipping...")
        return

    # Clear the scene of existing objects before import
    bpy.ops.scene.new()

    print(f"Importing USD file: {input_usd_path}")
    # Import the USD file
    try:
        with SuppressBlenderOutput():
            bpy.ops.wm.usd_import(filepath=input_usd_path)
    except Exception as e:
        print(f"Error importing USD file: {e}")
        return

    original_face_count = 0
    decimated_face_count = 0
    num_mesh = len([x.type == 'MESH' for x in bpy.context.scene.objects])
    success = True
    print(f"Processing {num_mesh} meshes")
    
    # Iterate through all objects in the scene
    processed_meshes = set()

    # thresholds for decimation
    threshold1 = 300 # first decimation
    threshold2 = 1000 # second decimation
    threshold3 = 5000 # remesh
    threshold4 = 10000 # problem mesh
    threshold5 = 30000 # problem object
    angle_limit = 15

    # ratio for decimation
    if "person" in input_usd_path:
        print("Person mesh detected, using custom thresholds and ratio")
        threshold1 = 1000
        threshold2 = 30000//min(num_mesh, 3)
        threshold3 = 50000//min(num_mesh, 5)
        threshold4 = 30000
        threshold5 = 50000
        ratio = 0.2
        angle_limit = 15
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            original_face_count += len(obj.data.polygons)
    
    if original_face_count < 100:
        print(f"Original face count is too low: {original_face_count}, skipping...")
        shutil.copy(input_usd_path, os.path.join(os.path.dirname(output_usd_path), "instance.usd"))
        with open(f"{os.path.dirname(output_usd_path)}/skipped.txt", "w") as f:
            f.write(f"Face count is too low: {original_face_count}, skipping...")
        return

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.name not in processed_meshes:
            if "component1_SRC_UV" in obj.name:
                continue
            if len(obj.data.polygons) > threshold1:
                print(f"Processing mesh object: {obj.name}, face count: {len(obj.data.polygons)}")
            processed_meshes.add(obj.name)
            # Count number of faces
            original_mesh_face_count = len(obj.data.polygons)
            # Select the object and make it active
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.make_single_user(type='ALL', object=True, obdata=True)

            if len(obj.data.polygons) > threshold5 and not "person" in input_usd_path:
                ratio0 = threshold5/len(obj.data.polygons)
                apply_decimate_modifiers(obj, ratio0, modifier_name="Decimate_Faces0")
                print(f"  > Applied decimation: face count = {original_mesh_face_count} → {len(obj.data.polygons)}, ratio = {ratio0}")

            if len(obj.data.polygons) > threshold1:
                print(f"  > Protecting UV seams for {obj.name}")
                protect_uv_seams(obj)

            if len(obj.data.polygons) > threshold1:
                num_polygons = len(obj.data.polygons)
                apply_decimate_modifiers(obj, ratio, modifier_name="Decimate_Faces1")
                apply_dissolve_modifiers(obj, angle_limit, modifier_name="Dissolve_Faces1")
                print(f"  > Applied first decimation: face count = {num_polygons} → {len(obj.data.polygons)}, ratio = {ratio}")

            if len(obj.data.polygons)>threshold2:
                num_polygons = len(obj.data.polygons)
                ratio2 = threshold2/len(obj.data.polygons)
                apply_decimate_modifiers(obj, ratio2, modifier_name="Decimate_Faces2")
                apply_dissolve_modifiers(obj, angle_limit, modifier_name="Dissolve_Faces2")
                print(f"  > Applied second decimation: face count = {num_polygons} → {len(obj.data.polygons)}, ratio = {ratio2}")

            if len(obj.data.polygons)>threshold3:
                # try remesh
                voxel_size = 3.0
                adaptivity = 3.0
                if len(obj.data.polygons)> 5*threshold3:
                    voxel_size = 5.0
                    adaptivity = 5.0
                print(f"  > Applying remesh, original face count: {len(obj.data.polygons)}, voxel_size: {voxel_size}")
                bpy.ops.ed.undo_push()
                apply_remesh_modifiers(obj, voxel_size, adaptivity, modifier_name="Remesh_Faces3")
                if len(obj.data.polygons)==0:
                    print(f"  > Remesh failed, original face count: {len(obj.data.polygons)}, skipping...")
                    bpy.ops.ed.undo()
                    success = False
                    continue
                ratio3 = min(threshold3/len(obj.data.polygons), 0.5)
                print(f"  > Applying third decimation, original face count: {len(obj.data.polygons)}, ratio: {ratio3}")
                apply_decimate_modifiers(obj, ratio3, modifier_name="Decimate_Faces3")
                apply_dissolve_modifiers(obj, 10, modifier_name="Dissolve_Faces3")
                print("  > Decimated faces after remesh:", len(obj.data.polygons))

            if len(obj.data.polygons)>threshold4:
                with open("problem_mesh.log", "a") as f:
                    f.write(f"{obj.name}, {input_usd_path}, {len(obj.data.polygons)}\n")
                print(f"  > Problem mesh: {obj.name}, {input_usd_path}, {len(obj.data.polygons)}")

            if len(obj.data.polygons) > threshold1:
                print(f"  > Processed mesh object {obj.name}, face count: {original_mesh_face_count} → {len(obj.data.polygons)}")

            decimated_face_count += len(obj.data.polygons)

            # Deselect the object
            obj.select_set(False)
    print(f"Object total faces: {original_face_count} → {decimated_face_count}")
    if decimated_face_count > threshold5 or not success:
        success = False
        with open("problem_objects.log", "a") as f:
            f.write(f"{input_usd_path}, {decimated_face_count}\n")
            print(f"Problem object: {input_usd_path}, {decimated_face_count}")
    
        # save blend file
        output_blend_path = os.path.splitext(output_usd_path)[0] + ".blend"
        with SuppressBlenderOutput():
            bpy.ops.wm.save_mainfile(filepath=output_blend_path)
        print(f"Saved blend file: {output_blend_path}")

    # Export decimated geometry
    export_usd(output_usd_path)
    original_usd_path = output_usd_path.replace("instance_renamed_decimated.usd", "instance.usd")
    
    # merge decimated geometry into original hierarchy
    print(f"Merging decimated geometry into original hierarchy...")
    swap_mesh_geometry(input_usd_path, output_usd_path, original_usd_path)

    # Verify face count
    bpy.ops.scene.new()
    faces = count_total_faces_in_usd(original_usd_path)
    if faces == 0:
        success = False
        print(f"Error: Resulting USD has 0 triangles")
        os.remove(output_usd_path)
    else:
        print(f"Resulting USD has {faces} triangles.")
        # save a "optimized.txt" file in that folder
        with open(f"{os.path.dirname(output_usd_path)}/optimized.txt", "w") as f:
            f.write(f"Total triangles in optimized meshes: {faces}\n")

        info = {
            "original_face_count": original_face_count,
            "decimated_face_count": decimated_face_count,
            "decimation_success": success,
        }

        # save to info.json
        with open(os.path.join(os.path.dirname(output_usd_path), "info.json"), "w") as f:
            json.dump(info, f)

def count_total_faces_in_usd(usd_path):
    """Count total faces in a USD file."""
    bpy.ops.scene.new()
    if not os.path.exists(usd_path):
        return None
    with SuppressBlenderOutput():
        bpy.ops.wm.usd_import(filepath=usd_path)
    total_faces = 0
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            total_faces += len(obj.data.polygons)
    return total_faces

def decimate_all_usd_in_folder(input_folder, ratio=0.1, worker_id=0, total_workers=1):
    """Recursively decimate all 'instance_renamed.usd' files in a folder."""
    all_files = []
    for root, dirs, files in os.walk(input_folder):
        if "instance_renamed.usd" in files:
            all_files.append((root, files))
    usd_files = []
    random.shuffle(all_files)
    pbar = tqdm(all_files, desc="Processing info.json files", maxinterval=0.5)
    for root, files in pbar:
        decimation_success = None
        if "info.json" in files:
            try:
                with open(os.path.join(root, "info.json"), "r") as f:
                    info = json.load(f)
                    decimation_success = info["decimated_face_count"]>0 and ("instance_renamed_decimated.usd" in files) or ("skipped.txt" in files)
                    # decimation_success = info["decimation_success"] and info["decimated_face_count"]>0 and ("instance_renamed_decimated.usd" in files) or ("skipped.txt" in files)
            except Exception as e:
                print(f"Error loading {os.path.join(root, 'info.json')}: {e}")
        if decimation_success is None:
            decimated_face_count = count_total_faces_in_usd(os.path.join(root, "instance_renamed_decimated.usd"))
            decimation_success = (decimated_face_count is not None) and (decimated_face_count < 30000) or ("skipped.txt" in files)
            decimated_face_count = 0 if decimated_face_count is None else decimated_face_count
            info = {
                "decimated_face_count": decimated_face_count,
                "decimation_success": decimation_success,
            }
        with open(os.path.join(root, "info.json"), "w") as f:
            json.dump(info, f)
        if not decimation_success:
            usd_files.append(os.path.join(root, "instance_renamed.usd"))
    usd_files.sort()
    # Partition files among workers
    if len(usd_files) < 500:
        my_files = usd_files
    else:
        my_files = [f for i, f in enumerate(usd_files) if i % total_workers == worker_id]
    # shuffle the files
    random.shuffle(my_files)

    print(f"TERMINAL: Worker {worker_id}/{total_workers-1} processing {len(my_files)}/{len(usd_files)} USD files in folder: {input_folder}")

    for idx, usd_path in tqdm(enumerate(my_files)):
        output_path = os.path.splitext(usd_path)[0] + "_decimated.usd"
        print(f"[INFO] Worker {worker_id} ({idx}/{len(my_files)}) Decimating: {usd_path} → {output_path}")
        try:
            decimate_usd_meshes(usd_path, output_path, ratio)
        except Exception as e:
            print(f"[ERROR] Worker {worker_id} Failed to process {usd_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    # Parse command line arguments
    # Note: When running with Blender, arguments after '--' are passed to the script
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # Get arguments after '--'
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Recursively decimate all USD files in a folder using Blender")
    parser.add_argument("--input_folder", 
                        type=str, 
                        required=True,
                        help="Path to the input folder containing USD files to decimate")
    parser.add_argument("--ratio", 
                        type=float, 
                        default=0.1,
                        help="Decimation ratio (default: 0.1)")
    parser.add_argument("--worker_id",
                        type=int,
                        default=0,
                        help="ID of the current worker (0-indexed)")
    parser.add_argument("--total_workers",
                        type=int,
                        default=1,
                        help="Total number of workers")

    args = parser.parse_args(argv)
    
    # Use parsed arguments
    input_folder = args.input_folder
    decimation_ratio = args.ratio
    worker_id = args.worker_id
    total_workers = args.total_workers
    
    print(f"[INFO] Input folder: {input_folder}")
    print(f"[INFO] Decimation ratio: {decimation_ratio}")
    print(f"[INFO] Worker ID: {worker_id}, Total Workers: {total_workers}")
    
    # Main function call
    decimate_all_usd_in_folder(input_folder, decimation_ratio, worker_id, total_workers)