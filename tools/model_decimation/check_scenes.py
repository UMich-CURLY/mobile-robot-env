
import sys
# print(sys.executable)
# print(sys.version)
import site
sys.path.append(site.getusersitepackages())

import os
import argparse
import re
import glob
import subprocess
import shutil
import time
from tqdm import tqdm

try:
    import bpy
    IS_IN_BLENDER = True
except ImportError:
    IS_IN_BLENDER = False

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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

def get_obj_path(obj):
    """Construct a path string based on parent hierarchy."""
    path = obj.name
    curr = obj
    while curr.parent:
        curr = curr.parent
        path = curr.name + "/" + path
    return "/" + path

def resolve_model_path(model_path, model_name, dataset_root):
    """
    Resolves the source directory for a model based on its scene path and name.
    
    Args:
        model_path (str): Full path in the scene (e.g., /Root/.../decoration/model_xxx_0)
        model_name (str): Name of the object (e.g., model_xxx_0)
        dataset_root (str): Root of the dataset (e.g., /home/.../grscenes_commercial)
        
    Returns:
        str: Resolved filesystem path or empty string if not found.
    """
    # Expecting path format: .../<category>/model_<id>_<instance>
    # 1. Extract category from path
    path_parts = model_path.strip("/").split("/")
    if len(path_parts) < 2:
        return ""
    
    category = path_parts[-2]
    
    # 2. Extract ID from model name
    # Format: model_<id>_<instance> or model_<id>
    name_parts = model_name.split('_')
    if len(name_parts) < 2 or name_parts[0] != "model":
        return ""
    
    model_id = name_parts[1]
    
    # 3. Check potential locations
    # {dataset_root}/models/object/others/{category}/{model_id}
    # {dataset_root}/models/object/articulated/{category}/{model_id}
    
    subdirs = ["others", "articulated"]
    
    for subdir in subdirs:
        candidate = os.path.join(dataset_root, "models", "object", subdir, category, model_id)
        if os.path.exists(candidate):
            return candidate
            
    return ""

def analyze_scene(usd_path, dataset_root):
    """
    Imports a USD file, counts total faces, and identifies objects with >10k faces.
    Aggregates face counts for objects where a parent starts with "model_".
    Returns: (total_faces, list_of_heavy_objects)
    """
    if not IS_IN_BLENDER:
        print("Error: analyze_scene must be run inside Blender")
        return 0, []

    print(f"Analyzing scene: {usd_path}")
    # Clear the scene of existing objects before import
    bpy.ops.scene.new()

    # Import the USD file
    try:
        with SuppressBlenderOutput():
            bpy.ops.wm.usd_import(filepath=usd_path)
    except Exception as e:
        print(f"Error importing USD file: {e}")
        return 0, []

    total_faces = 0
    # Store aggregated counts: key = (path, name), value = face_count
    aggregated_counts = {}

    # Iterate through all objects in the scene
    print(f"Number of objects in the scene: {len(bpy.context.scene.objects)}")
    for obj in tqdm(bpy.context.scene.objects, desc="Analyzing objects"):
        if obj.type == 'MESH':
            # Count number of faces
            face_count = len(obj.data.polygons)
            total_faces += face_count

            full_path = get_obj_path(obj)
            
            # Find the "model" parent in the path (nearest ancestor / last occurrence)
            matches = list(re.finditer(r"/(model_[0-9a-f]{32}(?:_\d+)?)", full_path))
            
            if matches:
                match = matches[-1]
                model_name = match.group(1)
                model_path = full_path[:match.end()]
                key = (model_path, model_name)
            else:
                key = (full_path, obj.name)

            if key not in aggregated_counts:
                aggregated_counts[key] = 0
            aggregated_counts[key] += face_count

    print("Analyzing heavy objects...")
    heavy_objects = []
    for (path, name), count in aggregated_counts.items():
        if count > 10000:
            source_path = resolve_model_path(path, name, dataset_root)
            heavy_objects.append({
                "name": name,
                "path": path,
                "faces": count,
                "source_path": source_path
            })
    print(f"Found {len(heavy_objects)} heavy objects")
    # Sort heavy objects by faces descending
    heavy_objects.sort(key=lambda x: x["faces"], reverse=True)

    return total_faces, heavy_objects

def process_folder_worker(input_folder, scene_log_path, objects_log_path, dataset_root, worker_id, total_workers):
    """
    Recursively find and process start_result_navigation.usd files.
    This runs inside Blender as a worker.
    """
    
    # Load processed scenes to skip from the MAIN log file (if exists, optional)
    processed_scenes = set()
    # We might want to check the main log file to skip already processed scenes from previous runs
    main_scene_log = scene_log_path.replace(f"_{worker_id}.csv", ".csv")
    if os.path.exists(main_scene_log):
         with open(main_scene_log, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                if "," in line:
                    processed_scenes.add(line.split(",")[0].strip())

    # Scan for files
    target_files = []
    print(f"Scanning {input_folder} for usd files...")
    for root, dirs, files in os.walk(input_folder):
        if "start_result_navigation.usd" in files:
            target_files.append(os.path.join(root, "start_result_navigation.usd"))
            continue
        for file in files:
            if file.endswith(".usd") and not "_renamed.usd" in file and not "_decimated.usd" in file:
                target_files.append(os.path.join(root, file))
    target_files.sort()
    
    # Filter files for this worker
    my_files = [f for i, f in enumerate(target_files) if i % total_workers == worker_id]
    
    # Further filter already processed
    my_files_filtered = [f for f in my_files if f not in processed_scenes]
    
    print(f"Worker {worker_id}/{total_workers}: Processing {len(my_files_filtered)} scenes (Total found: {len(target_files)}, Assigned: {len(my_files)}, Skipped: {len(my_files) - len(my_files_filtered)})")

    # Initialize partial logs with headers if they don't exist
    if not os.path.exists(scene_log_path):
        with open(scene_log_path, "w") as f:
            f.write("scene_path,total_faces\n")
        
    if not os.path.exists(objects_log_path):
        with open(objects_log_path, "w") as f:
            f.write("scene_path,prim_path,prim_name,face_count,source_usd_path\n")
    
    for usd_path in tqdm(my_files_filtered, desc=f"Worker {worker_id} Processing"):
        total_faces, heavy_objs = analyze_scene(usd_path, dataset_root)
        
        # Log scene info
        with open(scene_log_path, "a") as f:
            f.write(f"{usd_path},{total_faces}\n")
            
        # Log heavy objects
        if heavy_objs:
            with open(objects_log_path, "a") as f:
                for obj in heavy_objs:
                    f.write(f"{usd_path},{obj['path']},{obj['name']},{obj['faces']},{obj['source_path']}\n")

def launch_workers(input_folder, scene_log, object_log, dataset_root, num_workers):
    """
    Launch multiple Blender processes to process the folder in parallel.
    """
    print(f"Launching {num_workers} workers...")
    processes = []
    
    # Determine blender executable
    blender_exe = "blender" # Assume in path or use specific path
    # If shutil.which("blender") is None, you might need a fallback or error
    if shutil.which("blender") is None:
        print("Warning: 'blender' not found in PATH. Trying default locations...")
        # Add common locations if needed, but assuming env is set up like other scripts
    
    script_path = os.path.abspath(__file__)
    
    for i in range(num_workers):
        # Define partial log paths
        base_scene, ext_scene = os.path.splitext(scene_log)
        base_obj, ext_obj = os.path.splitext(object_log)
        
        partial_scene_log = f"{base_scene}_{i}{ext_scene}"
        partial_object_log = f"{base_obj}_{i}{ext_obj}"
        
        cmd = [
            blender_exe,
            "--background",
            "--python", script_path,
            "--",
            "--input_folder", input_folder,
            "--scene_log", partial_scene_log,
            "--object_log", partial_object_log,
            "--dataset_root", dataset_root,
            "--worker_id", str(i),
            "--total_workers", str(num_workers),
        ]
        
        log_file = open(f"check_scenes_worker_{i}.log", "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file))
        print(f"Started worker {i} (PID {p.pid})")
        
    # Wait for completion
    for p, f in processes:
        p.wait()
        f.close()
        if p.returncode != 0:
            print(f"Worker process failed with return code {p.returncode}")
            
    print("All workers finished. Merging logs...")
    
    # Merge logs
    # Scene Log
    with open(scene_log, "a") as outfile: # Append to existing or create new
        if os.path.getsize(scene_log) == 0:
             outfile.write("scene_path,total_faces\n")
             
        for i in range(num_workers):
            base_scene, ext_scene = os.path.splitext(scene_log)
            partial_log = f"{base_scene}_{i}{ext_scene}"
            if os.path.exists(partial_log):
                with open(partial_log, "r") as infile:
                    # Skip header
                    lines = infile.readlines()
                    if lines:
                        outfile.writelines(lines[1:])
                # Clean up
                os.remove(partial_log)

    # Object Log
    with open(object_log, "a") as outfile:
        if os.path.getsize(object_log) == 0:
             outfile.write("scene_path,prim_path,prim_name,face_count,source_usd_path\n")
             
        for i in range(num_workers):
            base_obj, ext_obj = os.path.splitext(object_log)
            partial_log = f"{base_obj}_{i}{ext_obj}"
            if os.path.exists(partial_log):
                with open(partial_log, "r") as infile:
                     # Skip header
                    lines = infile.readlines()
                    if lines:
                        outfile.writelines(lines[1:])
                # Clean up
                os.remove(partial_log)
                
    print("Logs merged successfully.")


if __name__ == "__main__":
    # Parse command line arguments
    # Note: When running with Blender, arguments after '--' are passed to the script
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # Get arguments after '--'
    parser = argparse.ArgumentParser(description="Analyze face counts in USD scenes using Blender")
    parser.add_argument("--input_folder", 
                        type=str, 
                        required=True,
                        help="Path to the input folder containing USD files")
    parser.add_argument("--scene_log", 
                        type=str, 
                        default="scene_faces.csv", 
                        help="Output log file for scene face totals")
    parser.add_argument("--object_log", 
                        type=str, 
                        default="heavy_objects.csv", 
                        help="Output log file for objects with >10000 faces")
    parser.add_argument("--dataset_root",
                        type=str,
                        default="/home/junzhewu/data/isaac_scenes_v1/grscenes_commercial",
                        help="Root path of the dataset for resolving model paths")
    
    # Worker arguments
    parser.add_argument("--worker_id", type=int, default=0, help="Worker ID")
    parser.add_argument("--total_workers", type=int, default=1, help="Total workers")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to launch (Manager mode only)")

    args = parser.parse_args(argv)
    
    print(f"[INFO] Input folder: {args.input_folder}")
    
    if IS_IN_BLENDER:
        # Worker mode running inside Blender
        process_folder_worker(
            args.input_folder, 
            args.scene_log, 
            args.object_log, 
            args.dataset_root,
            args.worker_id,
            args.total_workers
        )
    elif not IS_IN_BLENDER:
        # Manager mode running in Python
        launch_workers(
            args.input_folder,
            args.scene_log,
            args.object_log,
            args.dataset_root,
            args.num_workers
        )
