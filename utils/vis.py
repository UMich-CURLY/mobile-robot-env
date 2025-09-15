from pxr import Sdf, UsdGeom, Gf

def visualize_path(manager_env, path_xyz, target_xyz=None, dot_size=0.05, line_width=0.03):
    """Create USD prims to visualize waypoints (small dots) and a thin polyline."""
    stage = manager_env.scene.stage
    root_path = Sdf.Path("/World/PathVis")
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)

    GREEN = Gf.Vec3f(0.0, 1.0, 0.0)
    RED   = Gf.Vec3f(1.0, 0.0, 0.0)

    if path_xyz and len(path_xyz) >= 1:
        pts = [Gf.Vec3f(p[0], p[1], p[2]) for p in path_xyz]

        # Waypoint dots (green)
        pts_prim = UsdGeom.Points.Define(stage, root_path.AppendPath("Waypoints"))
        pts_prim.CreatePointsAttr(pts)
        pts_prim.CreateWidthsAttr([dot_size] * len(pts))
        # set per-point displayColor so the renderer definitely uses green
        pts_prim.CreateDisplayColorAttr([GREEN] * len(pts))

        # Polyline (red)
        curve = UsdGeom.BasisCurves.Define(stage, root_path.AppendPath("PathLine"))
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([len(pts)])
        curve.CreatePointsAttr(pts)
        curve.CreateWidthsAttr([line_width] * len(pts))
        # constant red for the whole curve (single color entry is fine)
        curve.CreateDisplayColorAttr([RED])

    if target_xyz is not None:
        # Target: bigger green dot
        tprim = UsdGeom.Points.Define(stage, root_path.AppendPath("Target"))
        tprim.CreatePointsAttr([Gf.Vec3f(*target_xyz)])
        tprim.CreateWidthsAttr([dot_size * 2.5])
        tprim.CreateDisplayColorAttr([GREEN])