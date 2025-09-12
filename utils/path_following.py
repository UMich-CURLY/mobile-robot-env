from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from pxr import UsdGeom, Sdf, Gf

def _wrap_to_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

def _world_to_body(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c*dx - s*dy, s*dx + c*dy

@dataclass
class FollowerParams:
    arrive_dist: float = 0.12
    arrive_yaw: float = 0.25
    max_v: float = 0.6
    max_yaw_rate: float = 1.0
    k_p_ang: float = 1.5
    kx: float = 0.6
    ky: float = 0.0
    min_forward: float = 0.05
    turn_in_place: bool = True
    keep_last_heading: bool = False

class PathFollower:
    def __init__(self, waypoints_xy: List[List[float]], params: Optional[FollowerParams] = None):
        if params is None:
            params = FollowerParams()
        self.waypoints: List[Tuple[float, float]] = [(float(p[0]), float(p[1])) for p in waypoints_xy]
        if not self.waypoints:
            raise ValueError("waypoints cannot be empty")
        self.params = params
        self.idx = 0
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    @property
    def current_index(self) -> int:
        return self.idx

    def reset(self, start_index: int = 0):
        self.idx = max(0, min(start_index, len(self.waypoints) - 1))
        self._done = False

    def step(self, x: float, y: float, yaw: float) -> Tuple[float, float, float, bool, int]:
        if self._done:
            return 0.0, 0.0, 0.0, True, self.idx

        px, py = self.waypoints[self.idx]
        dx, dy = px - x, py - y
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        ang_err = _wrap_to_pi(desired_yaw - yaw)
        p = self.params

        if p.turn_in_place and abs(ang_err) > p.arrive_yaw:
            vx = 0.0
            vy = 0.0
            omega = max(-p.max_yaw_rate, min(p.max_yaw_rate, p.k_p_ang * ang_err))
        else:
            ex_b, ey_b = _world_to_body(dx, dy, yaw)
            vx = max(-p.max_v, min(p.max_v, p.kx * ex_b))
            if abs(vx) < p.min_forward:
                vx = p.min_forward * (1.0 if ex_b >= 0 else -1.0)
            vy = max(-p.max_v, min(p.max_v, p.ky * ey_b)) if p.ky != 0.0 else 0.0
            omega = max(-p.max_yaw_rate, min(p.max_yaw_rate, p.k_p_ang * ang_err))

        if (dist < p.arrive_dist) and (abs(ang_err) < p.arrive_yaw):
            self.idx += 1
            if self.idx >= len(self.waypoints):
                self._done = True
                return 0.0, 0.0, 0.0, True, len(self.waypoints) - 1
            else:
                return 0.0, 0.0, 0.0, False, self.idx

        return vx, vy, omega, False, self.idx

def visualize_path(manager_env, path_xyz, target_xyz=None, dot_size=0.05, line_width=0.03):
    """
    Draw waypoints/line in USD stage. Call with full 3D path [[x,y,z], ...].
    """
    stage = manager_env.scene.stage
    root_path = Sdf.Path("/World/PathVis")
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)

    GREEN = Gf.Vec3f(0.0, 1.0, 0.0)
    RED   = Gf.Vec3f(1.0, 0.0, 0.0)

    if path_xyz and len(path_xyz) >= 1:
        pts = [Gf.Vec3f(p[0], p[1], p[2]) for p in path_xyz]

        pts_prim = UsdGeom.Points.Define(stage, root_path.AppendPath("Waypoints"))
        pts_prim.CreatePointsAttr(pts)
        pts_prim.CreateWidthsAttr([dot_size] * len(pts))
        pts_prim.CreateDisplayColorAttr([GREEN] * len(pts))

        curve = UsdGeom.BasisCurves.Define(stage, root_path.AppendPath("PathLine"))
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([len(pts)])
        curve.CreatePointsAttr(pts)
        curve.CreateWidthsAttr([line_width] * len(pts))
        curve.CreateDisplayColorAttr([RED])

    if target_xyz is not None:
        tprim = UsdGeom.Points.Define(stage, root_path.AppendPath("Target"))
        tprim.CreatePointsAttr([Gf.Vec3f(*target_xyz)])
        tprim.CreateWidthsAttr([dot_size * 2.5])
        tprim.CreateDisplayColorAttr([GREEN])

def quat_wxyz_to_yaw(wxyz):
    w, x, y, z = wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def get_base_xy_yaw(manager_env):
    pos = manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy()
    quat = manager_env.scene["robot"].data.root_state_w[0, 3:7].cpu().numpy()
    w, x, y, z = quat
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (float(pos[0]), float(pos[1])), yaw
