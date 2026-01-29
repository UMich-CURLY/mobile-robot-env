import argparse
import os
import sys
import glob
import subprocess
import json
import tempfile
import numpy as np
import yaml
from tqdm import tqdm

# Add repo root to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)

from utils.navmesh_utils import NavmeshInterface

def parse_args():
    parser = argparse.ArgumentParser(description="Generate navmesh from USD files")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing USD files")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for navmesh files")
    parser.add_argument("--config_path", type=str, default="episodes/task_config.yaml", help="Path to task config yaml")
    return parser.parse_args()

def extract_mesh_with_blender(usd_path, output_dir):
    # Blender script content
    blender_script = f"""
import bpy
import json
import numpy as np
import os

def convert_to_mesh(obj):
    # Apply all modifiers and convert to mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    object_eval = obj.evaluated_get(depsgraph)
    mesh = object_eval.to_mesh()
    return mesh

def main():
    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import USD
    bpy.ops.wm.usd_import(filepath="{usd_path}")
    
    verts_path = os.path.join("{output_dir}", "vertices.bin")
    faces_path = os.path.join("{output_dir}", "faces.bin")
    
    f_verts = open(verts_path, 'wb')
    f_faces = open(faces_path, 'wb')
    
    vert_offset = 0
    
    # Collect mesh objects first
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    for obj in mesh_objects:
        # Get world matrix
        matrix_world = obj.matrix_world
        
        # Get mesh data
        mesh = convert_to_mesh(obj)
        
        # Transform vertices to world space
        # mesh.vertices coordinates are local
        # Doing this list comprehension is cleaner than low-level numpy for now to ensure transform correctness
        verts = [matrix_world @ v.co for v in mesh.vertices]
        
        n_verts = len(verts)
        if n_verts == 0:
            obj.to_mesh_clear()
            continue
            
        # Convert to numpy and write directly
        # Flattening (x, y, z)
        verts_np = np.empty((n_verts, 3), dtype=np.float32)
        for i, v in enumerate(verts):
            verts_np[i] = (v.x, v.y, v.z)
            
        verts_np.tofile(f_verts)
        
        # Get faces (loop triangles are triangulated faces)
        mesh.calc_loop_triangles()
        
        # Collect faces
        # We need to flatten the list of indices
        faces_list = []
        for tri in mesh.loop_triangles:
            # Add offset to indices
            faces_list.extend([v_index + vert_offset for v_index in tri.vertices])
            
        if faces_list:
            np.array(faces_list, dtype=np.int32).tofile(f_faces)
        
        vert_offset += n_verts
        
        # Clean up
        obj.to_mesh_clear()
            
    f_verts.close()
    f_faces.close()

if __name__ == "__main__":
    main()
"""
    # Write blender script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(blender_script)
        script_path = f.name
        
    # Run Blender
    cmd = ["blender", "--background", "--python", script_path]
    
    try:
        print(f"Running Blender extraction for {usd_path}...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Blender failed: {e}")
        # print stderr for debugging
        print(e.stderr.decode())
        raise
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

def find_scene_config(file_path, task_config):
    # Normalize file path to compare with config paths
    # The config paths seem to be relative to 'data' folder typically, but let's check
    # e.g., vc_plus/AMSTERDAM/AMSTERDAM_gr_merged.usd
    
    abs_file_path = os.path.abspath(file_path)
    
    for scene_type, scene_data in task_config.get('scene', {}).items():
        episodes = scene_data.get('episodes', {})
        
        for scene_id, episode_data in episodes.items():
            config_path = episode_data.get('path')
            if not config_path:
                continue
                
            # We check if the config_path is a suffix of the file_path
            # This handles cases where file_path is absolute and config_path is relative
            if abs_file_path.endswith(config_path):
                return {
                    'scene_type': scene_type,
                    'scene_id': scene_id,
                    'navmesh_preset': episode_data.get('navmesh_preset', 'default'),
                }
    return None

def process_file(usd_path, output_folder, task_config):
    filename = os.path.basename(usd_path)
    # Construct output filename: innout_CITYNAME_navmesh.bin
    
    parts = filename.split('_gr_merged.usd')
    if len(parts) > 0:
        city_name = parts[0].lower()
    else:
        city_name = os.path.splitext(filename)[0].lower()
        
    output_filename = f"innout_{city_name}_navmesh.bin"
    output_path = os.path.join(output_folder, output_filename)
    
    # Find config
    scene_config = find_scene_config(usd_path, task_config)
    
    navmesh_settings = {}
    
    if scene_config:
        print(f"Found config for {usd_path}: {scene_config}")
        preset_name = scene_config['navmesh_preset']
        
        # Get navmesh settings from config
        if 'navmesh' in task_config and preset_name in task_config['navmesh']:
            navmesh_settings = task_config['navmesh'][preset_name]
            print(f"Using navmesh preset '{preset_name}': {navmesh_settings}")
        else:
             print(f"Warning: Navmesh preset '{preset_name}' not found in config, using defaults.")
    else:
        print(f"Warning: No configuration found for {usd_path} in task_config.yaml. Using defaults.")

    print(f"Processing {usd_path} -> {output_path}")
    
    # Use ./tmp for caching
    tmp_base = os.path.abspath("./tmp")
    cache_dir = os.path.join(tmp_base, os.path.splitext(os.path.basename(usd_path))[0])
    os.makedirs(cache_dir, exist_ok=True)
    
    v_path = os.path.join(cache_dir, "vertices.bin")
    f_path = os.path.join(cache_dir, "faces.bin")
    
    try:
        # Check if we need to extract
        if not os.path.exists(v_path) or not os.path.exists(f_path):
            print(f"Extracting mesh to {cache_dir}...")
            extract_mesh_with_blender(usd_path, cache_dir)
        else:
            print(f"Using cached mesh from {cache_dir}")
        
        if not os.path.exists(v_path) or not os.path.exists(f_path):
             print(f"Warning: No geometry extracted from {usd_path}")
             return
             
        vertices = np.fromfile(v_path, dtype=np.float32).reshape(-1, 3)
        if os.path.getsize(f_path) > 0:
            faces = np.fromfile(f_path, dtype=np.int32)
        else:
            faces = np.array([], dtype=np.int32)
        
        print(f"Loaded {len(vertices)} vertices and {len(faces)//3} triangles from Blender export")

        if len(vertices) == 0:
             print(f"Warning: No geometry found in {usd_path}")
             return
        
        # Prepare for NavmeshInterface
        if len(faces) > 0:
            faces_reshaped = faces.reshape(-1, 3)
            threes = np.full((faces_reshaped.shape[0], 1), 3, dtype=np.int32)
            formatted_faces_np = np.hstack((threes, faces_reshaped))
            formatted_faces = formatted_faces_np.flatten().tolist()
        else:
            formatted_faces = []
            
        # Initialize NavmeshInterface
        nm_interface = NavmeshInterface(up_axis='Z')
        
        # Update settings if provided
        if navmesh_settings:
            if 'tileSize' not in navmesh_settings:
                print("Enabling tiling (tileSize=256) for large scene support")
                navmesh_settings['tileSize'] = 256
            nm_interface.update_settings(navmesh_settings)
        
        # Convert coordinate system
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        print(f"Geometry Bounds: Min {min_bounds}, Max {max_bounds}")
        print(f"Geometry Size: {max_bounds - min_bounds}")
        
        verts_y_up = nm_interface._convert_up_axis(vertices)
        verts_flat = verts_y_up.flatten().tolist()
        
        print("Building navmesh...")
        nm_interface.nm.init_by_raw(verts_flat, formatted_faces)
        nm_interface.build_navmesh()
        nm_interface.save_navmesh(output_path)
        print("Done.")
        
    except Exception as e:
        print(f"Error processing {usd_path}: {e}")
        raise

def main():
    args = parse_args()
    
    # Load config
    config_path = os.path.join(repo_root, args.config_path)
    if not os.path.exists(config_path):
         print(f"Error: Config file not found at {config_path}")
         return
         
    with open(config_path, 'r') as f:
        task_config = yaml.safe_load(f)
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    # Find files
    search_pattern = os.path.join(args.input_folder, "**", "*_gr_merged.usd")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No files found matching {search_pattern}")
        # Try checking if input_folder is the directory containing the file directly
        files = glob.glob(os.path.join(args.input_folder, "*_gr_merged.usd"))
        if not files:
             print(f"No files found.")
             return

    print(f"Found {len(files)} files to process.")
    for file_path in files:
        process_file(file_path, args.output_folder, task_config)

if __name__ == "__main__":
    main()
