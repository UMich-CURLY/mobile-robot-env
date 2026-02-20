"""
基于 merge_poi.py 的 building 挖洞脚本
- 不读取 storefront 文件
- 仅使用 building、road、OSM 数据
- 在 amenity 点对应的 building 表面挖洞（固定尺寸）
- 将整个 scene 保存到 generated/{city}/ 文件夹，文件名加后缀
"""

import bpy
import math
import os
import xml.etree.ElementTree as ET
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable

# ============================================================
# CONFIG
# ============================================================

_bvh_cache = {}
_bvh_cache_max_age = 10

CONFIG = {
    "CITIES": ["MADRID"],
    "BASE_DIR": "D:/Desktop/ViCo/",
    
    # 建筑源：{city}/building 文件夹，导入其中所有 .usd 和 .glb 文件
    "BUILDING_FOLDER": "buildings",  # 相对于 {city}/ 的子文件夹名
    
    # 洞的固定尺寸（米），不依赖 storefront
    "HOLE_WIDTH": 4.0,
    "HOLE_HEIGHT": 3.5,
    "HOLE_DEPTH_MARGIN": 0.1,
    
    # 输出文件名后缀，最终为 {city}_cutholes.usd
    "OUTPUT_SUFFIX": "_cutholes",
    
    "BUILDING_NAME_KEYWORDS": [],
    
    # 测试模式：只处理前 N 个 amenity 点（设为 None 或 0 表示处理全部）
    "TEST_MAX_POIS": 5,
}

AMENITY_TYPES = [
    "restaurant", "cafe", "fast_food", "parking_entrance", "bar", "bank", "pub",
    "atm", "bicycle_rental", "parking", "pharmacy", "toilets", "theatre", "library",
    "dentist", "school", "post_office", "bureau_de_change", "doctors",
    "bicycle_repair_station", "clinic", "community_centre", "car_rental", "police",
    "arts_centre", "cinema", "kindergarten", "university", "coworking_space", "college",
]

# ============================================================
# 工具函数
# ============================================================

def purge_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for _ in range(3):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def ensure_collection(name: str):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

def import_usd_to_collection(filepath: str, collection_name: str, import_materials: bool = True):
    before = set(bpy.data.objects)
    if import_materials:
        try:
            bpy.ops.wm.usd_import(filepath=filepath, import_usd_preview=True)
        except TypeError:
            bpy.ops.wm.usd_import(filepath=filepath)
    else:
        bpy.ops.wm.usd_import(filepath=filepath)
    after = set(bpy.data.objects)
    new_objs = list(after - before)
    col = ensure_collection(collection_name)
    for obj in new_objs:
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        col.objects.link(obj)
    return new_objs, col

def import_buildings_from_folder(building_dir: str, collection_name: str):
    """
    Import all building files (.usd, .glb) from a directory into the given collection.
    Returns (total_imported_count, building_collection).
    """
    if not os.path.exists(building_dir):
        raise FileNotFoundError(f"Building directory not found: {building_dir}")
    building_files = []
    for f in os.listdir(building_dir):
        lower = f.lower()
        if lower.endswith('.usd') or lower.endswith('.usdc'):
            building_files.append(os.path.join(building_dir, f))
        elif lower.endswith('.glb') or lower.endswith('.gltf'):
            building_files.append(os.path.join(building_dir, f))
    building_files.sort()
    if not building_files:
        raise FileNotFoundError(f"No .usd/.glb files found in: {building_dir}")
    col = ensure_collection(collection_name)
    imported_count = 0
    for filepath in building_files:
        try:
            before = set(bpy.data.objects)
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ('.usd', '.usdc'):
                try:
                    bpy.ops.wm.usd_import(filepath=filepath, import_usd_preview=True)
                except TypeError:
                    bpy.ops.wm.usd_import(filepath=filepath)
            else:
                bpy.ops.import_scene.gltf(filepath=filepath)
            after = set(bpy.data.objects)
            new_objs = list(after - before)
            for obj in new_objs:
                for c in list(obj.users_collection):
                    c.objects.unlink(obj)
                col.objects.link(obj)
                if hasattr(obj, 'select_set'):
                    obj.select_set(False)
            imported_count += len(new_objs)
        except Exception as e:
            print(f"[Warning] Failed to import {os.path.basename(filepath)}: {e}")
    return imported_count, col

def iter_meshes(col, name_keywords=None):
    kws = [k.lower() for k in (name_keywords or [])]
    for obj in col.all_objects:
        if obj.type != "MESH":
            continue
        if kws:
            nm = obj.name.lower()
            if not any(k in nm for k in kws):
                continue
        yield obj

def _is_zero_location(vec, eps=1e-6):
    return abs(vec.x) <= eps and abs(vec.y) <= eps and abs(vec.z) <= eps

def _collect_hierarchy_objects(root_obj):
    objs = [root_obj]
    for child in root_obj.children:
        objs.extend(_collect_hierarchy_objects(child))
    return objs

def _get_root_level_objects(collection):
    all_objs = set(collection.all_objects)
    root_objs = [obj for obj in all_objs if (obj.parent is None or obj.parent not in all_objs)]
    root_named = [obj for obj in root_objs if obj.name.lower() in {"root", "/"}]
    if len(root_named) == 1:
        return list(root_named[0].children)
    return root_objs

def remove_nonzero_location_objects_under_root(collection):
    removed = 0
    root_level_objects = _get_root_level_objects(collection)
    objects_to_remove = []
    for obj in root_level_objects:
        if not _is_zero_location(obj.location):
            objects_to_remove.extend(_collect_hierarchy_objects(obj))
    objects_to_remove = list(set(objects_to_remove))
    for obj in objects_to_remove:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        except (ReferenceError, AttributeError):
            continue
        except Exception as e:
            print(f"[Cleanup] Warning: Failed to remove object {getattr(obj, 'name', 'Unknown')}: {e}")
    return removed

def evaluated_mesh(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(deps)
    mesh = obj_eval.to_mesh()
    return obj_eval, mesh

def build_bvh_for_object(obj, use_cache=False):
    import time
    if use_cache and obj in _bvh_cache:
        cache_time = _bvh_cache[obj][4]
        if time.time() - cache_time < _bvh_cache_max_age:
            return _bvh_cache[obj][:4]
    obj_eval, mesh = evaluated_mesh(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)
    if use_cache:
        _bvh_cache[obj] = (bvh, obj_eval, mesh, bm, time.time())
    return bvh, obj_eval, mesh, bm

def free_bvh_resources(obj_eval, mesh, bm, clear_cache=False):
    obj_in_cache = any(c[0] == obj_eval for c in _bvh_cache.values())
    if not obj_in_cache:
        obj_eval.to_mesh_clear()
        bm.free()

# ============================================================
# OSM / Road
# ============================================================

def load_amenity_points_from_osm(osm_file_path, amenity_type):
    if not os.path.exists(osm_file_path):
        raise FileNotFoundError(f"OSM file not found: {osm_file_path}")
    tree = ET.parse(osm_file_path)
    root = tree.getroot()
    amenity_points = []
    for element in root:
        if element.tag == 'node':
            has_amenity = False
            utm_x = utm_y = height = None
            for tag in element.findall('tag'):
                k, v = tag.get('k'), tag.get('v')
                if k == 'amenity' and v == amenity_type:
                    has_amenity = True
                elif k == 'utm_x':
                    utm_x = float(v)
                elif k == 'utm_y':
                    utm_y = float(v)
                elif k == 'height':
                    height = float(v)
            if has_amenity and utm_x is not None and utm_y is not None and height is not None:
                amenity_points.append((utm_x, utm_y, height, amenity_type))
    return amenity_points

def load_roads_from_osm(osm_file_path):
    if not os.path.exists(osm_file_path):
        raise FileNotFoundError(f"OSM file not found: {osm_file_path}")
    tree = ET.parse(osm_file_path)
    root = tree.getroot()
    nodes_dict = {}
    for element in root:
        if element.tag == 'node':
            node_id = element.get('id')
            utm_x = utm_y = None
            for tag in element.findall('tag'):
                k, v = tag.get('k'), tag.get('v')
                if k == 'utm_x':
                    utm_x = float(v)
                elif k == 'utm_y':
                    utm_y = float(v)
            if utm_x is not None and utm_y is not None:
                nodes_dict[node_id] = (utm_x, utm_y)
    roads = []
    for element in root:
        if element.tag == 'way':
            highway_type = None
            node_refs = []
            for child in element:
                if child.tag == 'nd':
                    node_refs.append(child.get('ref'))
                elif child.tag == 'tag':
                    k, v = child.get('k'), child.get('v')
                    if k == 'highway':
                        highway_type = v
            if highway_type and len(node_refs) >= 2:
                valid_node_refs = [ref for ref in node_refs if ref in nodes_dict]
                if len(valid_node_refs) >= 2:
                    roads.append({
                        'node_ids': valid_node_refs,
                        'highway_type': highway_type,
                        'nodes_dict': nodes_dict
                    })
    return roads

def find_nearest_road_direction(point_x, point_y, roads, max_search_radius=50.0):
    min_distance = float('inf')
    best_road_direction = None
    best_road_normal = None
    for road in roads:
        node_ids = road['node_ids']
        nodes_dict = road['nodes_dict']
        for i in range(len(node_ids) - 1):
            n1_id, n2_id = node_ids[i], node_ids[i + 1]
            if n1_id not in nodes_dict or n2_id not in nodes_dict:
                continue
            x1, y1 = nodes_dict[n1_id]
            x2, y2 = nodes_dict[n2_id]
            dx, dy = x2 - x1, y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue
            t = max(0.0, min(1.0, ((point_x - x1) * dx + (point_y - y1) * dy) / seg_len_sq))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            dist = math.sqrt((point_x - closest_x) ** 2 + (point_y - closest_y) ** 2)
            if dist < min_distance and dist <= max_search_radius:
                min_distance = dist
                road_dir_len = math.sqrt(seg_len_sq)
                if road_dir_len > 1e-6:
                    best_road_direction = Vector((dx / road_dir_len, dy / road_dir_len, 0.0))
                dir_x = closest_x - point_x
                dir_y = closest_y - point_y
                dir_len = math.sqrt(dir_x * dir_x + dir_y * dir_y)
                if dir_len > 1e-6:
                    best_road_normal = Vector((dir_x / dir_len, dir_y / dir_len, 0.0))
    return best_road_direction, min_distance if best_road_direction else None, best_road_normal

# ============================================================
# Building 查找 / 挖洞
# ============================================================

def best_nearest_hit(building_meshes, query_world: Vector):
    terrain_keywords = ['terrain', 'ground', 'land', 'floor', 'surface']
    best = None
    for obj in building_meshes:
        if any(k in obj.name.lower() for k in terrain_keywords):
            continue
        bvh, obj_eval, mesh, bm = build_bvh_for_object(obj)
        try:
            inv = obj.matrix_world.inverted()
            q_local = inv @ query_world
            res = bvh.find_nearest(q_local)
            if res is None:
                continue
            hit_local, normal_local, face_index, dist_local = res
            if hit_local is None or normal_local is None:
                continue
            hit_world = obj.matrix_world @ hit_local
            n_world = (obj.matrix_world.to_3x3() @ normal_local).normalized()
            dist = (hit_world - query_world).length
            if best is None or dist < best["dist"]:
                best = {"obj": obj, "hit_world": hit_world, "normal_world": n_world, "dist": dist}
        finally:
            free_bvh_resources(obj_eval, mesh, bm)
    if best is None:
        raise RuntimeError("No nearest face found.")
    return best

def check_face_occlusion(building_meshes, face_center_world, road_normal_3d, max_check_distance=50.0, exclude_obj=None, exclude_face_index=None):
    ray_direction = road_normal_3d.normalized()
    ray_origin = face_center_world + ray_direction * 0.1
    min_hit_distance = float('inf')
    for obj in building_meshes:
        if exclude_obj is not None and obj == exclude_obj:
            continue
        bvh, obj_eval, mesh, bm = build_bvh_for_object(obj)
        try:
            inv = obj.matrix_world.inverted()
            origin_local = inv @ ray_origin
            ray_dir_local = (inv.to_3x3() @ ray_direction).normalized()
            hit_local, _, face_index, distance = bvh.ray_cast(origin_local, ray_dir_local, max_check_distance)
            if hit_local is not None:
                if exclude_obj is None or obj != exclude_obj or face_index != exclude_face_index:
                    hit_world = obj.matrix_world @ hit_local
                    min_hit_distance = min(min_hit_distance, (hit_world - ray_origin).length)
        finally:
            free_bvh_resources(obj_eval, mesh, bm)
    return min_hit_distance < 0.5

def find_road_facing_building_face(building_meshes, query_world, road_normal, max_search_radius=5.0, num_samples=30):
    import random
    if road_normal is None:
        return None
    road_normal_3d = Vector((road_normal.x, road_normal.y, 0.0)).normalized()
    best_hit = None
    best_score = -float('inf')
    for i in range(num_samples):
        angle1 = random.uniform(0, 2 * math.pi)
        angle2 = random.uniform(0, math.pi)
        sample_dir = Vector((math.sin(angle2) * math.cos(angle1), math.sin(angle2) * math.sin(angle1), math.cos(angle2)))
        sample_point = query_world + sample_dir * random.uniform(0.5, max_search_radius)
        hit = best_nearest_hit(building_meshes, sample_point)
        normal_xy = Vector((hit["normal_world"].x, hit["normal_world"].y, 0.0))
        if normal_xy.length < 1e-6:
            continue
        normal_xy.normalize()
        dot_product = normal_xy.dot(road_normal_3d)
        if dot_product < 0.3:
            continue
        face_center = hit["hit_world"]
        hit_obj = hit["obj"]
        bvh, obj_eval, mesh, bm = build_bvh_for_object(hit_obj)
        try:
            inv = hit_obj.matrix_world.inverted()
            center_local = inv @ face_center
            res = bvh.find_nearest(center_local)
            if res is None:
                continue
            _, _, face_index, _ = res
            is_occluded = check_face_occlusion(building_meshes, face_center, road_normal_3d, max_check_distance=50.0, exclude_obj=hit_obj, exclude_face_index=face_index)
            if is_occluded:
                continue
        finally:
            free_bvh_resources(obj_eval, mesh, bm)
        score = dot_product * (1.0 / (1.0 + hit["dist"]))
        if score > best_score:
            best_score = score
            best_hit = hit
    return best_hit

def step2_find_nearest_building_face(building_meshes, query_world):
    any_hit = best_nearest_hit(building_meshes, query_world)
    nearest_building = any_hit["obj"]
    outer_hit = None
    if nearest_building and nearest_building.type == 'MESH':
        bvh, obj_eval, mesh, bm = build_bvh_for_object(nearest_building)
        try:
            inv = nearest_building.matrix_world.inverted()
            q_local = inv @ query_world
            max_outer_dist = -float('inf')
            for angle in [i * math.pi / 8 for i in range(16)]:
                ray_dir_xy = Vector((math.cos(angle), math.sin(angle), 0.0))
                for z_offset in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                    ray_dir = Vector((ray_dir_xy.x, ray_dir_xy.y, z_offset)).normalized()
                    ray_dir_local = (inv.to_3x3() @ ray_dir).normalized()
                    current_origin = q_local
                    for _ in range(5):
                        hit_local, normal_local, face_index, dist_local = bvh.ray_cast(current_origin, ray_dir_local, 50.0)
                        if hit_local is None:
                            break
                        hit_world = nearest_building.matrix_world @ hit_local
                        n_world = (nearest_building.matrix_world.to_3x3() @ normal_local).normalized()
                        dist_to_query = (hit_world - query_world).length
                        if abs(n_world.z) < 0.3 and dist_to_query > max_outer_dist:
                            max_outer_dist = dist_to_query
                            outer_hit = {"obj": nearest_building, "hit_world": hit_world, "normal_world": n_world, "dist": dist_to_query}
                        current_origin = hit_local + ray_dir_local * 0.01
        finally:
            free_bvh_resources(obj_eval, mesh, bm)
    if outer_hit is not None:
        any_hit = outer_hit
    if abs(any_hit["normal_world"].z) < 0.3:
        return any_hit
    terrain_keywords = ['terrain', 'ground', 'land', 'floor', 'surface']
    obj = nearest_building
    if any(k in obj.name.lower() for k in terrain_keywords):
        return any_hit
    best_vertical_hit = None
    bvh, obj_eval, mesh, bm = build_bvh_for_object(obj)
    try:
        inv = obj.matrix_world.inverted()
        q_local = inv @ query_world
        for face in bm.faces:
            face_center = face.calc_center_median()
            n_world = (obj.matrix_world.to_3x3() @ face.normal).normalized()
            if abs(n_world.z) > 0.3:
                continue
            hit_world = obj.matrix_world @ face_center
            dist = (hit_world - query_world).length
            if dist > 15.0:
                continue
            if best_vertical_hit is None or dist < best_vertical_hit["dist"]:
                best_vertical_hit = {"obj": obj, "hit_world": hit_world, "normal_world": n_world, "dist": dist}
    finally:
        free_bvh_resources(obj_eval, mesh, bm)
    return best_vertical_hit if best_vertical_hit else any_hit

def step3_ensure_road_facing_face(building_meshes, query_world, initial_hit, roads):
    if roads is None:
        return initial_hit, False
    road_direction, road_dist, road_normal = find_nearest_road_direction(query_world.x, query_world.y, roads)
    if road_normal is None:
        return initial_hit, False
    initial_normal = initial_hit["normal_world"].normalized()
    initial_normal_xy = Vector((initial_normal.x, initial_normal.y, 0.0))
    if initial_normal_xy.length > 1e-6:
        initial_normal_xy.normalize()
    road_normal_3d = Vector((road_normal.x, road_normal.y, 0.0)).normalized()
    dot_with_road = initial_normal_xy.dot(road_normal_3d)
    if dot_with_road > 0.3:
        return initial_hit, True
    elif abs(dot_with_road) < 0.1:
        return initial_hit, True
    else:
        target_building = initial_hit["obj"]
        road_facing_hit = find_road_facing_building_face([target_building], query_world, road_normal, max_search_radius=20.0, num_samples=50)
        if road_facing_hit and (road_facing_hit["hit_world"] - query_world).length < initial_hit["dist"] * 2.0:
            return road_facing_hit, True
        return initial_hit, False

def compute_cut_axes(hit_obj, hit_p, n_out, query_world, roads):
    """从 step4 提取：计算 xw, yw, zw, height_axis_cut"""
    zw = Vector((0, 0, 1))
    n_out_xy = Vector((n_out.x, n_out.y, 0.0))
    if n_out_xy.length > 1e-6:
        n_out_xy.normalize()
    yw_from_building = n_out_xy.copy() if n_out_xy.length > 1e-6 else Vector((0, 1, 0))
    yw = yw_from_building.copy()
    if roads:
        road_direction, _, _ = find_nearest_road_direction(query_world.x, query_world.y, roads)
        if road_direction:
            road_dir_3d = Vector((road_direction.x, road_direction.y, 0.0)).normalized()
            yw_candidate1 = Vector((-road_dir_3d.y, road_dir_3d.x, 0.0))
            yw_candidate2 = Vector((road_dir_3d.y, -road_dir_3d.x, 0.0))
            if n_out_xy.length > 1e-6:
                dot1, dot2 = yw_candidate1.dot(n_out_xy), yw_candidate2.dot(n_out_xy)
                yw_from_road = yw_candidate1 if dot1 >= dot2 else yw_candidate2
                best_dot = max(dot1, dot2)
                if best_dot > 0.99985:
                    yw = yw_from_road
                elif best_dot < 0.5:
                    yw = yw_from_building
                else:
                    yw = yw_from_building
            else:
                yw = yw_candidate1
    xw = Vector((yw.y, -yw.x, 0.0))
    zw_proj = zw - zw.dot(n_out) * n_out
    height_axis_cut = zw_proj.normalized() if zw_proj.length > 1e-6 else zw
    return xw, yw, zw, height_axis_cut

def remove_faces_in_area(target_obj, center_world, normal_world, width_axis, height_axis, width, height, depth_margin, poi_index):
    """在 building 表面切割矩形区域"""
    half_width = width * 0.5
    half_height = height * 0.5
    bpy.context.view_layer.objects.active = target_obj
    target_obj.select_set(True)
    original_mesh = target_obj.data
    bm = bmesh.new()
    bm.from_mesh(original_mesh)
    world_matrix = target_obj.matrix_world.copy()
    inv_matrix = world_matrix.inverted()
    bm.transform(world_matrix)
    bm.faces.ensure_lookup_table()
    center_local = center_world.copy()
    normal_local = normal_world.normalized()
    width_axis_local = width_axis.normalized()
    height_axis_local = height_axis.normalized()
    bvh = BVHTree.FromBMesh(bm)
    sample_faces = set()
    for iw in range(7):
        for ih in range(5):
            t_w = (iw / 6) * 2 - 1 if 6 > 0 else 0
            t_h = (ih / 4) * 2 - 1 if 4 > 0 else 0
            sample_point = center_local + width_axis_local * (t_w * half_width) + height_axis_local * (t_h * half_height)
            ray_origin = sample_point + normal_local * 2.0
            hit_loc, hit_normal, hit_face_idx, hit_dist = bvh.ray_cast(ray_origin, -normal_local, 5.0)
            if hit_face_idx is not None and hit_face_idx < len(bm.faces):
                hit_face = bm.faces[hit_face_idx]
                if hit_face.normal.normalized().dot(normal_local) > 0.5:
                    sample_faces.add(hit_face_idx)
    if len(sample_faces) == 0:
        for offset_n in [0.0, 0.5, 1.0, -0.5, -1.0]:
            test_point = center_local + normal_local * offset_n
            nearest_loc, _, nearest_idx, _ = bvh.find_nearest(test_point, 3.0)
            if nearest_idx is not None and nearest_idx < len(bm.faces):
                hit_face = bm.faces[nearest_idx]
                if hit_face.normal.normalized().dot(normal_local) > 0.5:
                    sample_faces.add(nearest_idx)
                    break
    if len(sample_faces) == 0:
        bm.transform(inv_matrix)
        bm.free()
        target_obj.select_set(False)
        return False
    faces_to_remove = [bm.faces[idx] for idx in sample_faces if idx < len(bm.faces) and bm.faces[idx].is_valid]
    actual_deleted_faces = 0
    if faces_to_remove:
        faces_to_cut = [f for f in faces_to_remove if f.is_valid]
        if faces_to_cut:
            try:
                geom_set = set(faces_to_cut)
                for face in faces_to_cut:
                    geom_set.update(face.edges)
                    geom_set.update(face.verts)
                geom_to_cut = list(geom_set)
                left_plane_point = center_local - width_axis_local * half_width
                bmesh.ops.bisect_plane(bm, geom=geom_to_cut, dist=0.001, plane_co=left_plane_point, plane_no=width_axis_local, clear_outer=False, clear_inner=False)
                bm.faces.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.verts.ensure_lookup_table()
                geom_to_cut = list(bm.faces) + list(bm.edges) + list(bm.verts)
                right_plane_point = center_local + width_axis_local * half_width
                bmesh.ops.bisect_plane(bm, geom=geom_to_cut, dist=0.001, plane_co=right_plane_point, plane_no=-width_axis_local, clear_outer=False, clear_inner=False)
                bm.faces.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.verts.ensure_lookup_table()
                geom_to_cut = list(bm.faces) + list(bm.edges) + list(bm.verts)
                bottom_plane_point = center_local - height_axis_local * half_height
                bmesh.ops.bisect_plane(bm, geom=geom_to_cut, dist=0.001, plane_co=bottom_plane_point, plane_no=height_axis_local, clear_outer=False, clear_inner=False)
                bm.faces.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.verts.ensure_lookup_table()
                geom_to_cut = list(bm.faces) + list(bm.edges) + list(bm.verts)
                top_plane_point = center_local + height_axis_local * half_height
                bmesh.ops.bisect_plane(bm, geom=geom_to_cut, dist=0.001, plane_co=top_plane_point, plane_no=-height_axis_local, clear_outer=False, clear_inner=False)
                bm.faces.ensure_lookup_table()
                faces_inside_rect = []
                for face in bm.faces:
                    face_center = face.calc_center_median()
                    to_face = face_center - center_local
                    proj_vec = to_face - to_face.dot(normal_local) * normal_local
                    proj_w = proj_vec.dot(width_axis_local)
                    proj_h = proj_vec.dot(height_axis_local)
                    dist_n = abs(to_face.dot(normal_local))
                    if abs(proj_w) <= half_width + 0.1 and abs(proj_h) <= half_height + 0.1 and dist_n < 1.0:
                        fn = face.normal.copy().normalized()
                        if fn.dot(normal_local) > 0.5:
                            faces_inside_rect.append(face)
                if faces_inside_rect:
                    bmesh.ops.delete(bm, geom=faces_inside_rect, context='FACES')
                    actual_deleted_faces += len(faces_inside_rect)
                    remaining = []
                    for face in bm.faces:
                        fc = face.calc_center_median()
                        to_f = fc - center_local
                        pv = to_f - to_f.dot(normal_local) * normal_local
                        pw, ph = pv.dot(width_axis_local), pv.dot(height_axis_local)
                        dist_n = abs(to_f.dot(normal_local))
                        if abs(pw) <= half_width + 0.1 and abs(ph) <= half_height + 0.1 and dist_n < 1.0 and face.normal.normalized().dot(normal_local) > 0.5:
                            remaining.append(face)
                    if remaining:
                        bmesh.ops.delete(bm, geom=remaining, context='FACES')
                        actual_deleted_faces += len(remaining)
            except Exception:
                valid_faces = [f for f in faces_to_remove if f.is_valid]
                if valid_faces:
                    bmesh.ops.delete(bm, geom=valid_faces, context='FACES')
                    actual_deleted_faces += len(valid_faces)
    bm.transform(inv_matrix)
    bm.to_mesh(target_obj.data)
    target_obj.data.update()
    bm.free()
    bpy.context.view_layer.update()
    target_obj.select_set(False)
    if actual_deleted_faces > 0:
        print(f"[Cut] Successfully deleted {actual_deleted_faces} face(s)")
        return True
    return False

def perform_cut_for_poi(building_meshes, hit_obj, hit_p, n_out, query_world, roads, hole_width, hole_height, depth_margin, poi_index):
    """对单个 POI 执行挖洞"""
    target_building_only = [hit_obj]
    xw, yw, zw, height_axis_cut = compute_cut_axes(hit_obj, hit_p, n_out, query_world, roads)
    target_z = None
    if hit_obj.type == 'MESH' and hit_obj.data.vertices:
        building_world_matrix = hit_obj.matrix_world
        building_z_coords = []
        for v in hit_obj.data.vertices:
            world_pos = building_world_matrix @ Vector((v.co.x, v.co.y, v.co.z))
            building_z_coords.append(world_pos.z)
        if building_z_coords:
            target_z = min(building_z_coords)
    if target_z is None:
        target_z = hit_p.z
    cut_center_z = target_z + hole_height * 0.5
    rect_center_candidate = Vector((hit_p.x, hit_p.y, cut_center_z))
    hit_rect = best_nearest_hit(target_building_only, rect_center_candidate)
    rect_center = Vector((hit_rect["hit_world"].x, hit_rect["hit_world"].y, cut_center_z)) if hit_rect["dist"] < 10.0 else rect_center_candidate
    return remove_faces_in_area(hit_obj, rect_center, n_out, xw, height_axis_cut, hole_width, hole_height, depth_margin, poi_index)

# ============================================================
# 主流程
# ============================================================

def export_scene_usd(output_path):
    """导出整个 scene 到 USD"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        win = bpy.context.window_manager.windows[0]
        area = None
        for a in win.screen.areas:
            if a.type in {'VIEW_3D', 'OUTLINER', 'PROPERTIES', 'TOPBAR'}:
                area = a
                break
        if area is None:
            area = win.screen.areas[0]
        region = None
        for r in area.regions:
            if r.type == 'WINDOW':
                region = r
                break
        if region is None:
            region = area.regions[-1]
        with bpy.context.temp_override(window=win, area=area, region=region):
            export_kw = dict(
                filepath=output_path,
                export_materials=True,
                export_textures=True,
            )
            for opt in [("relative_paths", True), ("generate_preview_surface", True)]:
                kw = {**export_kw, opt[0]: opt[1]}
                try:
                    bpy.ops.wm.usd_export('EXEC_DEFAULT', **kw)
                    break
                except TypeError:
                    continue
            else:
                bpy.ops.wm.usd_export('EXEC_DEFAULT', **export_kw)
        print(f"[Export] Saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_single_city(city, base_dir):
    """处理单个城市：从 {city}/building 导入建筑 + OSM，在 amenity 点挖洞，导出到 generated/{city}/"""
    building_folder = CONFIG.get("BUILDING_FOLDER", "building")
    building_dir = os.path.join(base_dir, city, building_folder)
    osm_file = os.path.join(base_dir, city, "road_data", "road_data_with_height.osm")
    output_dir = os.path.join(base_dir, "generated", city)
    suffix = CONFIG.get("OUTPUT_SUFFIX", "_cutholes")
    output_usd = os.path.join(output_dir, f"{city}{suffix}.usd")

    if not os.path.exists(building_dir):
        print(f"[ERROR] Building directory not found: {building_dir}")
        return
    if not os.path.exists(osm_file):
        print(f"[ERROR] OSM file not found: {osm_file}")
        return

    purge_scene()
    print(f"[Config] City: {city}")
    print(f"[Config] Building directory: {building_dir}")
    print(f"[Config] OSM file: {osm_file}")
    print(f"[Config] Output: {output_usd}")

    roads = None
    try:
        roads = load_roads_from_osm(osm_file)
        print(f"[Main] Loaded {len(roads)} roads")
    except Exception as e:
        print(f"[Warning] Failed to load roads: {e}")

    imported_count, building_col = import_buildings_from_folder(building_dir, "COL_Building")
    print(f"[Import] Loaded {imported_count} objects from building folder")
    remove_nonzero_location_objects_under_root(building_col)

    all_meshes = list(iter_meshes(building_col, CONFIG.get("BUILDING_NAME_KEYWORDS", [])))
    terrain_keywords = ['terrain', 'ground', 'land', 'floor', 'surface']
    non_building_keywords = ['lamp', 'light', 'streetlight', 'pole', 'post', 'sign', 'bench', 'fence', 'barrier', 'decorative', 'tree', 'bush', 'plant', 'shrub']
    building_meshes = []
    for obj in all_meshes:
        obj_name_lower = obj.name.lower()
        if any(k in obj_name_lower for k in terrain_keywords):
            continue
        if obj.name.startswith("Object_"):
            continue
        if any(k in obj_name_lower for k in non_building_keywords):
            continue
        try:
            if hasattr(obj, 'bound_box') and len(obj.bound_box) == 8:
                bbox_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
                bbox_min = Vector([min(c[i] for c in bbox_corners) for i in range(3)])
                bbox_max = Vector([max(c[i] for c in bbox_corners) for i in range(3)])
                bbox_size = bbox_max - bbox_min
                max_dim = max(bbox_size.x, bbox_size.y, bbox_size.z)
                if max_dim < 3.0:
                    continue
                if max_dim < 5.0 and bbox_size.x * bbox_size.y * bbox_size.z < 50.0:
                    continue
        except Exception:
            pass
        building_meshes.append(obj)

    if not building_meshes:
        print("[ERROR] No building meshes found")
        return

    print(f"[Import] Found {len(building_meshes)} building meshes")

    all_pois = []
    for amenity_type in AMENITY_TYPES:
        try:
            points = load_amenity_points_from_osm(osm_file, amenity_type)
            for utm_x, utm_y, height, _ in points:
                all_pois.append((utm_x, utm_y, height, amenity_type))
        except Exception:
            continue

    print(f"[Main] Found {len(all_pois)} amenity points")

    test_max = CONFIG.get("TEST_MAX_POIS") or 0
    if test_max > 0:
        all_pois = all_pois[:test_max]
        print(f"[Main] TEST mode: processing only first {len(all_pois)} POIs")

    hole_width = CONFIG.get("HOLE_WIDTH", 4.0)
    hole_height = CONFIG.get("HOLE_HEIGHT", 3.5)
    depth_margin = CONFIG.get("HOLE_DEPTH_MARGIN", 0.1)

    success_count = 0
    for poi_index, (utm_x, utm_y, height, amenity_type) in enumerate(tqdm(all_pois, desc="Cutting holes")):
        query_world = Vector((utm_x, utm_y, height))
        try:
            hit = step2_find_nearest_building_face(building_meshes, query_world)
            hit, road_facing = step3_ensure_road_facing_face(building_meshes, query_world, hit, roads)
            cut_ok = perform_cut_for_poi(
                building_meshes, hit["obj"], hit["hit_world"], hit["normal_world"],
                query_world, roads, hole_width, hole_height, depth_margin, poi_index
            )
            if cut_ok:
                success_count += 1
        except Exception as e:
            pass

    print(f"[Main] Successfully cut {success_count}/{len(all_pois)} holes")
    bpy.context.view_layer.update()
    export_scene_usd(output_usd)

def main():
    base_dir = CONFIG["BASE_DIR"]
    cities = CONFIG["CITIES"]
    print(f"\n{'='*80}")
    print(f"[Main] Processing cities: {', '.join(cities)}")
    print(f"[Main] Output: generated/{{city}}/{{city}}{CONFIG.get('OUTPUT_SUFFIX', '_cutholes')}.usd")
    print(f"{'='*80}")
    for city in cities:
        try:
            process_single_city(city, base_dir)
        except Exception as e:
            print(f"[ERROR] Failed for {city}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'='*80}")
    print("[Main] Done")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
