# blender --background --python /home/junzhewu/pohsun/SG-VLN/robot_env/decimate_usd_models_face_blender.py

import bpy
import os
import bmesh

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)  # also flush errors immediately

def clean_mesh(obj):
    """Clean mesh: remove doubles and loose geometry."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Remove doubles (merge by distance)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)

    # Delete loose vertices
    loose_verts = [v for v in bm.verts if not v.link_edges]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')

    bm.to_mesh(mesh)
    bm.free()


def export_usd(output_usd_path):
    """Export current Blender scene to USD."""
    try:
        bpy.ops.wm.usd_export(
            filepath=output_usd_path,
            export_textures=True,
            relative_paths=True,
            export_materials=False,
            export_animation=False,
            export_uvmaps=True,
            use_instancing=True
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
    
    # Check if the input file exists
    if not os.path.exists(input_usd_path):
        print(f"Error: Input file not found at {input_usd_path}")
        return

    # Clear the scene of existing objects before import
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    print(f"Importing USD file: {input_usd_path}")
    # Import the USD file
    try:
        bpy.ops.wm.usd_import(filepath=input_usd_path)
    except Exception as e:
        print(f"Error importing USD file: {e}")
        return

    original_face_count = 0
    decimated_face_count = 0
    # Iterate through all objects in the scene
    processed_meshes = set()
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.data not in processed_meshes:
            processed_meshes.add(obj.data)
            print(f"Processing mesh object: {obj.name}")
            
            # Count number of faces
            face_count = len(obj.data.polygons)
            original_face_count += face_count
            if face_count < 500 or len(obj.data.vertices) == 0:
                continue
            # print(f"Number of faces: {face_count}")

            # Select the object and make it active
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)

            # Add the Decimate modifier
            decimate_modifier = obj.modifiers.new(name="Decimate_Faces", type='DECIMATE')
            decimate_modifier.ratio = ratio
            decimate_modifier.use_dissolve_boundaries = False

            # Apply the modifier
            # Using bpy.ops.object.modifier_apply(modifier="Modifier Name") can fail in scripts.
            # A more robust method is to use the object.convert() operator.
            bpy.ops.object.modifier_apply(modifier="Decimate_Faces")
            
            # Count number of faces after decimation
            face_count = len(obj.data.polygons)
            # print(f"Number of faces after decimation: {face_count}")
            decimated_face_count += face_count

            # Deselect the object
            obj.select_set(False)
            
    export_usd(output_usd_path)

def decimate_all_usd_in_folder(input_folder, ratio=0.1):
    """Recursively decimate all 'instance.usd' files in a folder."""
    usd_files = []
    for root, dirs, files in os.walk(input_folder):
        if "instance_decimated.usd" in files:
            print(f"[SKIP] Already decimated: {root}")
            continue
        for file in files:
            if file.endswith('instance.usd'):
                usd_files.append(os.path.join(root, file))

    print(f"[INFO] Found {len(usd_files)} USD files in folder: {input_folder}")

    for idx, usd_path in enumerate(usd_files, 1):
        output_path = os.path.splitext(usd_path)[0] + "_decimated.usd"
        print(f"[INFO] ({idx}/{len(usd_files)}) Decimating: {usd_path} → {output_path}")
        try:
            decimate_usd_meshes(usd_path, output_path, ratio)
        except Exception as e:
            print(f"[ERROR] Failed to process {usd_path}: {e}")
            continue

# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    
    # Params:
    input_folder = "/home/junzhewu/pohsun/data_decimated/grscenes_commercial/models"
    decimation_ratio = 0.1

    # Main function call
    decimate_all_usd_in_folder(input_folder, decimation_ratio)
