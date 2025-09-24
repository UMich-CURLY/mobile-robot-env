from collections import defaultdict
import os
import numpy as np
import omni.physx

import PyRecastDetour as pyrecast

from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade, Vt, UsdUtils
import omni
from isaaclab.sim.utils import export_prim_to_file
# Import usd_utils functionality

def traverse_instanced_children(prim):
    """Get every Prim child beneath `prim`, even if `prim` is instanced.

    Important:
        If `prim` is instanced, any child that this function yields will
        be an instance proxy.

    Args:
        prim (`pxr.Usd.Prim`): Some Prim to check for children.

    Yields:
        `pxr.Usd.Prim`: The children of `prim`.

    """
    for child in prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
        yield child

        for subchild in traverse_instanced_children(child):
            yield subchild

def parent_and_children_as_mesh(parent_prim):
    if UsdGeom.Imageable(parent_prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
        return [], []
    if parent_prim.IsA(UsdGeom.Mesh):
        points, faces = get_mesh(parent_prim)
        return points, faces
    
    found_meshes = []
    for x in traverse_instanced_children(parent_prim):
        # Check if prim is visible in the scene or not
        if UsdGeom.Imageable(x).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if x.IsA(UsdGeom.Mesh):
            found_meshes.append(x)

    # children = parent_prim.GetAllChildren()
    # children = [child.GetPrimPath() for child in children]
    # points, faces = get_mesh(children)
    points, faces = get_mesh(found_meshes)
    
    return points, faces

def children_as_mesh(stage, parent_prim):
    children = parent_prim.GetAllChildren()
    children = [child.GetPrimPath() for child in children]
    points, faces = get_mesh(stage, children)
    
    return points, faces


def get_all_stage_mesh(stage, prims):

    found_meshes = []

    # For each selected prim, go through its children and figure out if they are meshes
    for prim in prims:
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue

        if prim.IsA(UsdGeom.Mesh):
            found_meshes.append(prim)
            continue
        # Traverse the scene graph and print the paths of prims, including instance proxies
        for x in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if UsdGeom.Imageable(x).ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if x.IsA(UsdGeom.Mesh):
                found_meshes.append(x)
    # print("Length of found meshes: ", len(found_meshes))
    points, faces = get_mesh(found_meshes)
   
    return points, faces

def get_mesh(objs):
    points, faces = [],[]
    i = 0
    loss = 0
    for obj in objs:
        f_offset = len(points)
        # f, p = convert_to_mesh(obj)#usd_stage.GetPrimAtPath(obj))
        f, p = meshconvert(obj)#usd_stage.GetPrimAtPath(obj))
        p = np.array(p)
        f = np.array(f)
        # print("p.shape: ", p.shape)
        # print("f.shape: ", f.shape)
        """ Added by Po-Hsun """
        # if(len(p) == 0 or len(f) == 0):
        #     # print("points len: ", len(p))
        #     # print("faces len: ", len(f))
        #     print("Prim path unable to form navmesh:", obj.GetPath())
        #     print("len points: ", len(p), " len faces: ", len(f))
        #     loss +=1
        #     continue
        """ Ended by Po-Hsun """
        # print("points shape: ", p.shape)  n x 3
        # print("faces shape: ", f.shape)   n x 3
        points.extend(p)
        faces.extend(f + f_offset)
        print("index: ", i)
        i += 1

    print("loss mesh: ", loss)
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

def create_points(nodes, prim_path="/World/Points", color=(0.0, 1.0, 1.0), width=2.0):
    '''Create and draw a Points on the stage following the nodes'''
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Points.Define(stage, prim_path)
    prim.CreatePointsAttr(nodes)
    prim.CreateWidthsAttr(np.array([width], dtype=float))
    prim.CreateDisplayColorAttr([color])

def create_curve(path, prim_path="/World/Path", color=(1, 0, 0), width=1.0 ):
    '''Create and draw a BasisCurve on the stage following the nodes'''

    if len(path) <= 1:
        print('[WARNING]: No valid path to visualize')
        return

    nodes = [Gf.Vec3f(float(pt[0]), float(pt[1]), float(pt[2])) for pt in path]

    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.BasisCurves.Define(stage, prim_path)
    prim.CreatePointsAttr(nodes)

    # Set the number of curve verts to be the same as the number of points we have
    curve_verts = prim.CreateCurveVertexCountsAttr()
    curve_verts.Set([len(nodes)])

    # Set the curve type to linear so that each node is connected to the next
    type_attr = prim.CreateTypeAttr()
    type_attr.Set('linear')
    type_attr = prim.GetTypeAttr().Get()
    # Set the width of the curve

    width_attr = prim.CreateWidthsAttr(np.array([width], dtype=float))

    width =  Vt.FloatArray.FromNumpy(np.asarray([width for x in range(len(nodes))]))

    width_attr.Set(width)

    # color_primvar = prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    UsdGeom.Primvar(prim.GetDisplayColorAttr()).SetInterpolation("constant")
    prim.GetDisplayColorAttr().Set([color])



def create_mesh(prim_path, points, indices, colors=None, opacity=None, use_prevsrf=True):
    '''
    Create a mesh in USD
    '''

    time = Usd.TimeCode.Default()
    stage = omni.usd.get_context().get_stage()

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(points, time)
    mesh.GetFaceVertexIndicesAttr().Set(indices, time)
    mesh.GetFaceVertexCountsAttr().Set([3] * len(indices), time)

    if use_prevsrf:

        mtl_path = Sdf.Path(f"/World/Looks/PreviewSurface_{prim_path.split('/')[-1]}")

        mtl = UsdShade.Material.Define(stage, mtl_path)
        shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(colors) 
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
        mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            
        # Bind the mesh
        UsdShade.MaterialBindingAPI(mesh).Bind(mtl)

    # Opacity seems to be broken
    else:
        UsdGeom.Primvar(mesh.GetDisplayColorAttr()).SetInterpolation("constant")
        UsdGeom.Primvar(mesh.GetDisplayOpacityAttr()).SetInterpolation("constant")
        if colors:
            mesh.GetDisplayColorAttr().Set(colors, time)
        if opacity:
            mesh.GetDisplayOpacityAttr().Set(opacity, time)

    return prim_path

class NavmeshInterface:
    def __init__(self, up_axis='Y', stage=None): 
        self.built = False
        self.input_prim = None
        self.input_vert = None
        self.input_tri = None
        self.random_points = None
        self.wall_outline = []
        self.settings = {}
        self.path_points = None
        self.stage = stage
        self.start_pos = None
        self.end_pos = None
        self.nm = pyrecast.Navmesh()

        # if z_up is true, we will need to do some conversion before sending to 
        # recast, then, we will convert it back to y_up (all functions will need to do that)
        self.z_up = up_axis == 'Z'

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

    def setup_navmesh(self, selected_paths):
        self.input_prim = [self.stage.GetPrimAtPath(x) for x in selected_paths]
        self.input_vert, self.input_tri = get_all_stage_mesh(self.stage, self.input_prim)
        if len(self.input_vert) == 0:
            print('[INFO]: No mesh found')
        print("[INFO]: Loading navmesh from vertices and triangles, will take a while, please wait.")
        self.input_vert = self._convert_up_axis(self.input_vert)

        print(f'bounding box: max={self.input_vert.max(axis=0)}, min={self.input_vert.min(axis=0)}')
        verts_flat = []
        for vertex in self.input_vert:
            verts_flat.extend(vertex)
        # Convert faces to the format expected by init_by_raw
        # The example shows faces as [3, v0, v1, v2, 3, v0, v1, v2, ...]
        # where 3 indicates triangle (3 vertices per face)
        faces_flat = []
        for face in self.input_tri:
            # cur_faces_flat = np.hstack([np.full((len(face), 1), 3), face]).flatten().tolist()
            cur_faces_flat = np.concatenate([[3], face]).tolist()
            # print("`cur_faces_flat: ", cur_faces_flat)
            faces_flat.extend(cur_faces_flat)

        # Initialize the navmesh with raw data
        self.nm.init_by_raw(verts_flat, faces_flat)
        print(f"[INFO]: Loaded navmesh from {np.array(verts_flat).shape[0]//3} vertices and {np.array(faces_flat).shape[0]//4}') triangles")

    def build_navmesh(self, settings={}):
        settings = self.nm.get_settings()
        # These mirror Sample::resetCommonSettings defaults
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
        # nvidia
        # settings["cellSize"] = 100.
        # settings["cellHeight"] = 100.
        # settings["agentHeight"] = 61
        # settings["agentRadius"] = 55
        # settings["agentMaxClimb"] = 100
        # settings["agentMaxSlope"] = 26.0
        self.nm.set_settings(settings)
        # Try watershed (0) like default demo; if it fails, switch to monotone (1)
        self.nm.set_partition_type(1)
        self.nm.build_navmesh()
        v, t, = self.get_navmesh_polygons()
        print(f'v shape: {v.shape}')
        print(f't shape: {t.shape}')
        self.built = v.shape[0] > 0
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
            create_mesh('/World/ground/navmeshmesh', v, t, color, opacity)
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

        self.navmesh_v = self._convert_up_axis(self.navmesh_v, inverse=True)

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
        print(f"Path points: {path_points.shape}")
        
        if path_points.shape[0] <= 1:
            print("[WARNING]: No valid path found")
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
        print(f'random points: {vertices}')

        return vertices