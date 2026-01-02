from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm
from pxr import UsdGeom, Gf, Usd
from utils.vis import visualize_mesh
try:
    import PyRecastDetour as pyrecast
except ImportError:
    raise ImportError("PyRecastDetour not found, run tools/install_pyrecastdetour.sh to install")

def get_all_stage_mesh(stage, prims, exclude_paths=[]):

    found_meshes = []

    # For each selected prim, go through its children and figure out if they are meshes
    for prim in prims:
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue

        if prim.GetPath() in exclude_paths:
            continue

        if prim.IsA(UsdGeom.Mesh):
            found_meshes.append(prim)
            continue
        # Traverse the scene graph and print the paths of prims, including instance proxies
        for x in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if UsdGeom.Imageable(x).ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if x.GetPath() in exclude_paths:
                continue
            if x.IsA(UsdGeom.Mesh):
                found_meshes.append(x)
    print("Length of found meshes: ", len(found_meshes))
    points, faces = get_mesh(found_meshes)
    # print("points",points)
    # print("faces",faces)
    return points, faces

def get_mesh(objs):
    points, faces = [],[]

    for i, obj in tqdm(enumerate(objs), total=len(objs)):
        f_offset = len(points)
        # f, p = convert_to_mesh(obj)#usd_stage.GetPrimAtPath(obj))
        f, p = meshconvert(obj)#usd_stage.GetPrimAtPath(obj))
        p = np.array(p)
        f = np.array(f)

        if len(p) == 0:
            continue

        if p.max()-p.min() > 1000:
            print(f"[WARNING]: Large mesh found: {obj.GetPath()}, Max: {p.max()}, Min: {p.min()}")

        if p.max()>1000 or p.min()<-1000:
            print(f"[WARNING]: Remote mesh found: {obj.GetPath()}, Max: {p.max()}, Min: {p.min()}")

        # print("points shape: ", p.shape)  n x 3
        # print("faces shape: ", f.shape)   n x 3
        points.extend(p)
        faces.extend(f + f_offset)
    print("points",len(points))
    print("faces",len(faces))
    return points, faces

def meshconvert(prim):

    # Create an XformCache object to efficiently compute world transforms
    xform_cache = UsdGeom.XformCache()

    # Get the mesh schema
    mesh = UsdGeom.Mesh(prim)
    
    # Get verts and triangles
    tris = mesh.GetFaceVertexIndicesAttr().Get()
    if not tris:
        return [], []
    tris_cnt = mesh.GetFaceVertexCountsAttr().Get()

    # Get the vertices in local space
    points_attr = mesh.GetPointsAttr()
    local_points = points_attr.Get()
    
    # Convert the VtVec3fArray to a NumPy array
    points_np = np.array(local_points, dtype=np.float64)
    
    # Add a fourth component (with value 1.0) to make the points homogeneous
    num_points = len(local_points)
    ones = np.ones((num_points, 1), dtype=np.float64)
    points_np = np.hstack((points_np, ones))

    # Compute the world transform for this prim
    world_transform = xform_cache.GetLocalToWorldTransform(prim)

    # Convert the GfMatrix to a NumPy array
    matrix_np = np.array(world_transform, dtype=np.float64).reshape((4, 4))

    # Transform all vertices to world space using matrix multiplication
    world_points = np.dot(points_np, matrix_np)
    if np.any(np.isnan(world_points)):
        print("world_points: ", world_points)
        print("world_points shape: ", world_points.shape)
        print("points_np: ", points_np)
        print("matrix_np: ", matrix_np)
        print("prim path: ", prim.GetPath())
        print("prim: ", prim)

    tri_list = convert_to_triangle_mesh(tris, tris_cnt)
    # tri_list = tri_list.flatten()

    world_points = world_points[:,:3]

    return np.array(tri_list), world_points

def convert_to_triangle_mesh(FaceVertexIndices, FaceVertexCounts):
    """
    Convert a list of vertices and a list of faces into a triangle mesh.
    
    A list of triangle faces, where each face is a list of indices of the vertices that form the face.
    """
    
    # Parse the face vertex indices into individual face lists based on the face vertex counts.

    faces = []
    start = 0
    for count in FaceVertexCounts:
        end = start + count
        face = FaceVertexIndices[start:end]
        faces.append(face)
        start = end

    # Convert all faces to triangles
    triangle_faces = []
    for face in faces:
        if len(face) < 3:
            newface = []  # Invalid face
        elif len(face) == 3:
            newface = [face]  # Already a triangle
        else:
            # Fan triangulation: pick the first vertex and connect it to all other vertices
            v0 = face[0]
            newface = [[v0, face[i], face[i + 1]] for i in range(1, len(face) - 1)]

        triangle_faces.extend(newface)
    
    return np.array(triangle_faces)

class NavmeshInterface:
    def __init__(self, up_axis='Y'): 
        self.built = False
        self.input_prim = None
        self.input_vert = None
        self.input_tri = None
        self.random_points = None
        self.wall_outline = []
        self.path_points = None
        self.start_pos = None
        self.end_pos = None
        self.nm = pyrecast.Navmesh()
        settings = self.nm.get_settings()
        settings["cellSize"] = 0.08
        settings["cellHeight"] = 0.1
        settings["agentHeight"] = 0.3
        settings["agentRadius"] = 0.3
        settings["agentMaxClimb"] = 0.05
        settings["agentMaxSlope"] = 26.0
        settings["regionMinSize"] = 0.5
        settings["regionMergeSize"] = 0.5
        settings["edgeMaxLen"] = 5.0
        settings["edgeMaxError"] = 1.3
        settings["vertsPerPoly"] = 6.0
        settings["detailSampleDist"] = 1.0
        settings["detailSampleMaxError"] = 1.0
        self.settings = settings
        # if z_up is true, we will need to do some conversion before sending to 
        # recast, then, we will convert it back to y_up (all functions will need to do that)
        self.z_up = up_axis == 'Z'
    
    def update_settings(self, settings):
        for key, value in settings.items():
            self.settings[self._snake_to_camel(key)] = value

    def is_path_valid(self):
        if self.path_points is None:
            print("[WARNING]: No path found")
            return False
        if len(self.path_points) < 2:
            return False
        if self.path_points[0] != self.start_pos or self.path_points[-1] != self.end_pos:
            print("[WARNING]: Path does not match start or end position")
            return False
        return True

    def setup_navmesh(self, selected_paths, exclude_paths, stage, scene_type=None):
        self.input_prim = [stage.GetPrimAtPath(x) for x in selected_paths]
        self.input_vert, self.input_tri = get_all_stage_mesh(stage, self.input_prim, exclude_paths=exclude_paths)
        if len(self.input_vert) == 0:
            print('[INFO]: No mesh found')
        self.input_vert = np.array(self.input_vert)
        print(f"[INFO]: Loaded {len(self.input_vert)} vertices and {len(self.input_tri)} triangles")
        print(f'[INFO]: bounding box: max={self.input_vert.max(axis=0)}, min={self.input_vert.min(axis=0)}')
        if np.any(np.isnan(self.input_vert)):
            print("[WARNING]: NaNs found in input vertices")
        self.input_vert = self._convert_up_axis(self.input_vert)

        # Constrain vertices with z > 10 to z = 10 (only for vc scenes)
        # Note: after _convert_up_axis, z is the third component (index 2)
        print(f"[INFO]: scene_type: {scene_type}")
        if scene_type == 'vc':
            z_mask = self.input_vert[:, 2] > 10
            if np.any(z_mask):
                num_constrained = np.sum(z_mask)
                self.input_vert[z_mask, 2] = 10
                print(f"[INFO]: Constrained {num_constrained} vertices with z > 10 to z = 10")
            
            # Filter out invalid faces (degenerate triangles with zero area or collapsed vertices)
            # A triangle is invalid if all three vertices are the same or if the triangle has zero area
            # valid_faces = []
            # for face in tqdm(self.input_tri, desc="Filtering faces"):
            #     v0, v1, v2 = self.input_vert[face[0]], self.input_vert[face[1]], self.input_vert[face[2]]
            #     # Check if triangle is degenerate (zero area or all vertices same)
            #     edge1 = v1 - v0
            #     edge2 = v2 - v0
            #     cross_product = np.cross(edge1, edge2)
            #     triangle_area = 0.5 * np.linalg.norm(cross_product)
            #     # Keep triangle if area is greater than a small threshold (1e-6)
            #     if triangle_area > 1e-6:
            #         valid_faces.append(face)
            
            # if len(valid_faces) < len(self.input_tri):
            #     num_removed = len(self.input_tri) - len(valid_faces)
            #     print(f"[INFO]: Removed {num_removed} degenerate faces after z constraint")
            #     self.input_tri = valid_faces

        verts_flat = []
        for vertex in tqdm(self.input_vert,desc="Loading vertices"):
            verts_flat.extend(vertex)
        # Convert faces to the format expected by init_by_raw
        # The example shows faces as [3, v0, v1, v2, 3, v0, v1, v2, ...]
        # where 3 indicates triangle (3 vertices per face)
        faces_flat = []
        for face in tqdm(self.input_tri,desc="Loading faces"):
            # cur_faces_flat = np.hstack([np.full((len(face), 1), 3), face]).flatten().tolist()
            cur_faces_flat = np.concatenate([[3], face]).tolist()
            # print("`cur_faces_flat: ", cur_faces_flat)
            faces_flat.extend(cur_faces_flat)

        # Initialize the navmesh with raw data
        print("[INFO]: Loading geometry from vertices and triangles. This will take a while, please wait.")
        self.nm.init_by_raw(verts_flat, faces_flat)
        print(f"[INFO]: Geometry loaded")

    def build_navmesh(self):
        print(f"[INFO]: settings: {self.settings}")
        self.nm.set_settings(self.settings)
        # Try watershed (0); if it fails, switch to monotone (1)
        self.nm.set_partition_type(1)
        print("[INFO]: Building navmesh, will take a while, please wait.")
        self.nm.build_navmesh()
        v, t, = self.get_navmesh_polygons()
        print(f'v shape: {v.shape}')
        print(f't shape: {t.shape}')
        log = self.nm.get_log()
        print(log)
        if not self.built:
            print('[WARNING]: Failed to build navmesh')

    def visualize_navmesh(self):
        if self.built:
            v, t, = self.navmesh_v, self.navmesh_t
            v = v.flatten()
            # create a usd color of blue with transparency
            color = Gf.Vec3f(0.051208995, 0.774935, 0.94585985)
            opacity = 0.89
            visualize_mesh('/World/ground/navmeshmesh', v, t, color, opacity)
            print("[INFO]: Visualized navmesh")
        else:
            print('[WARNING]: Navmesh not built') 

    def get_navmesh_polygons(self):
        trivert, polygon_indices, polygon_sizes = self.nm.get_navmesh_polygonization()
        trivert = np.asarray(trivert).reshape(-1,3)

        # Parse polygon_indices using polygon_sizes
        t = []
        index_offset = 0
        
        for poly_size in polygon_sizes:
            if poly_size == 3:
                # Triangle - add directly
                triangle = [polygon_indices[index_offset], 
                        polygon_indices[index_offset + 1], 
                        polygon_indices[index_offset + 2]]
                t.append(triangle)
            elif poly_size > 3:
                # Polygon with more than 3 vertices - use fan triangulation
                first_vertex = polygon_indices[index_offset]
                for i in range(1, poly_size - 1):
                    triangle = [first_vertex,
                            polygon_indices[index_offset + i],
                            polygon_indices[index_offset + i + 1]]
                    t.append(triangle)
            # Skip polygons with less than 3 vertices (invalid)
            index_offset += poly_size
        
        self.navmesh_v = np.array(trivert, dtype=np.float32)
        self.navmesh_t = np.array(t, dtype=np.int32)
        # print(f'navmesh_v shape: {self.navmesh_v.shape}')
        # print(f'navmesh_t shape: {self.navmesh_t.shape}')
        # print(f'navmesh_v: {self.navmesh_v[0]}')
        # print(f'navmesh_t: {self.navmesh_t[0]}')
        self.navmesh_v = self._convert_up_axis(self.navmesh_v, inverse=True)
        
        self.built = self.navmesh_v.shape[0] > 0
        return self.navmesh_v, self.navmesh_t
    
    def find_paths(self, start, end):
        """
        Input: start, end are lists of points [x, y, z]
        """
        start = self._convert_up_axis([start])[0]
        end = self._convert_up_axis([end])[0]

        path_points = self.nm.pathfind_straight(start, end, 1)
        path_points = np.array(path_points).reshape(-1, 3)
        path_points = self._convert_up_axis(path_points, inverse=True)
        # print(f"Path points: {path_points.shape}")
        
        # if path_points.shape[0] <= 1:
        #     print("[WARNING]: No valid path found")
        return path_points

    def save_navmesh(self, save_path):
        print("[INFO]: Exported navmesh to", save_path)
        self.nm.save_navmesh(save_path)

    def load_navmesh(self, load_path):
        print("[INFO]: Loaded navmesh from", load_path)
        self.nm.load_navmesh(load_path)
        self.built = True
        print(self.nm.get_log())
        print("[INFO]: Updating navmesh polygons")
        self.get_navmesh_polygons()

    """ ----- Navmesh helper functions ----- """
    # ----- Helper functions for up axis conversion ----- #
    def _convert_up_axis(self, vertices, inverse=False):
        '''
        Convert all data between navmesh interface and end-user to be in the correct up axis

        pyrecast assumes y-up, so only change when z-up is true
        inverse = True will convert y-up output back to z-up
        '''
        if self.z_up == False:
            return vertices
        
        vertices = np.array(vertices)
        v_copy = np.empty(shape=vertices.shape)
        if inverse:
            # Convert data we have been given in y-up into the z-up of scene (e.g. output from recast)
            v_copy[:, 0], v_copy[:, 1], v_copy[:, 2] = vertices[:, 0], -vertices[:, 2], vertices[:, 1]
        else: 
            # Convert the up axis from Y to Z
            v_copy[:, 0], v_copy[:, 1], v_copy[:, 2] = vertices[:, 0], vertices[:, 2], -vertices[:, 1]

        vertices = v_copy
        return vertices

    # ----- Helper functions for getting random points ----- #
    def sample_random_points(self, num_points):
        if not self.built or self.navmesh_v.shape[0] == 0:
            print("[WARNING]: Navmesh not built")
            return None

        v = self.navmesh_v
        t = self.navmesh_t
        random_poly_indices = np.random.randint(0, len(t), num_points)
        random_poly = t[random_poly_indices]
        weights = np.random.rand(random_poly.shape[0]*3).reshape(-1, 3)
        weights = weights[:,:,np.newaxis].repeat(3, axis=2)
        vertices = np.average(v[random_poly], weights=weights, axis=1)
        # print(f'random points: {vertices}')

        return vertices