import bpy
import sys
import os
import math
import argparse

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_scene():
    # Camera
    bpy.ops.object.camera_add(location=(0, -2, 1))
    cam = bpy.context.active_object
    cam.data.lens = 35
    bpy.context.scene.camera = cam
    
    # Light
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    bpy.context.active_object.data.energy = 5
    
    # Add another fill light
    bpy.ops.object.light_add(type='POINT', location=(-5, -5, 5))
    bpy.context.active_object.data.energy = 100

    return cam

def import_object(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.stl':
            # Blender 4.0+ changed the API for STL import to bpy.ops.wm.stl_import
            # We try the new API first.
            try:
                bpy.ops.wm.stl_import(filepath=file_path)
            except (AttributeError, RuntimeError):
                # Fallback to old API
                try:
                    bpy.ops.import_mesh.stl(filepath=file_path)
                except (AttributeError, RuntimeError) as e:
                    print(f"Failed to import STL using both wm.stl_import and import_mesh.stl: {e}")
                    return False

        elif ext == '.obj':
            # Blender 4.0+ changed the API for OBJ import to bpy.ops.wm.obj_import
            try:
                bpy.ops.wm.obj_import(filepath=file_path)
            except (AttributeError, RuntimeError):
                # Fallback to old API
                try:
                    bpy.ops.import_scene.obj(filepath=file_path)
                except (AttributeError, RuntimeError) as e:
                    print(f"Failed to import OBJ using both wm.obj_import and import_scene.obj: {e}")
                    return False

        elif ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=file_path)
        else:
            print(f"Unsupported file format: {ext}")
            return False
    except Exception as e:
        print(f"Error importing {file_path}: {e}")
        return False
    return True

def setup_material(obj):
    # Create a new material
    mat = bpy.data.materials.new(name="ObjectMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    # Set a random color or a specific color
    # Let's use a nice blueish color for now
    bsdf.inputs['Base Color'].default_value = (0.1, 0.4, 0.8, 1) # RGBA
    bsdf.inputs['Roughness'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.0
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def normalize_object():
    # Select all mesh objects
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        return
    
    # Join into one if multiple
    bpy.ops.object.select_all(action='DESELECT')
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    
    obj = bpy.context.active_object
    
    # Center origin to geometry
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # Move to (0,0,0)
    obj.location = (0, 0, 0)
    
    # Scale to fit in unit box
    max_dim = max(obj.dimensions)
    if max_dim > 0:
        scale = 1.0 / max_dim
        obj.scale = (scale, scale, scale)
        
    # Apply transform
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Setup material
    setup_material(obj)

def render_angles(output_dir, num_angles=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cam = bpy.context.scene.camera
    # Create an empty at center to track
    bpy.ops.object.empty_add(location=(0, 0, 0))
    target = bpy.context.active_object
    
    # Add track to constraint to camera
    const = cam.constraints.new(type='TRACK_TO')
    const.target = target
    const.track_axis = 'TRACK_NEGATIVE_Z'
    const.up_axis = 'UP_Y'
    
    radius = 2.0
    
    for i in range(num_angles):
        angle = (i / num_angles) * 2 * math.pi
        x = radius * math.sin(angle)
        y = -radius * math.cos(angle)
        z = 1.0 # slightly elevated
        
        cam.location = (x, y, z)
        bpy.context.view_layer.update()
        
        bpy.context.scene.render.filepath = os.path.join(output_dir, f"render_{i}.png")
        # Faster render settings
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        
        bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    # Get arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)
    
    reset_scene()
    if import_object(args.input):
        setup_scene()
        normalize_object()
        render_angles(args.output_dir)
