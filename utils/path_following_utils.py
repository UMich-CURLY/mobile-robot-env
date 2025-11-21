import os
import json
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))

def world_to_body(dx, dy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c*dx - s*dy, s*dx + c*dy

def set_yaw_to_forward(cam_pos, waypoints):
    last_pos = cam_pos[:2]
    new_waypoints = []
    for wp in waypoints:
        yaw_angle = np.arctan2((wp[1] - last_pos[1]), wp[0] - last_pos[0])
        new_waypoints.append([wp[0], wp[1], yaw_angle])
        last_pos = wp[:2]
    return new_waypoints

class WaypointFollower:
    def __init__(self,
        device,
        max_vel=[1.5, 1.0, 1.0],
        min_lin_speed=0.3,
        min_ang_speed=0.3,
        kp=[2.0, 1.5, 2.0],
        ki=[0.0, 0.0, 0.0], # [0.1, 0.1, 0.1]
        kd=[0.0, 0.0, 0.0], # [3, 3, 1]
        max_integral=[5.0, 5.0, 3.0],
        arrive_dist=0.05,
        arrive_yaw=np.pi/180.0*15.0,
        term_dist=0.05,
        term_yaw=np.pi/180.0*5.0,
        lookahead_distance=0,
    ):
        self.device = device
        self.max_vel = np.array(max_vel)
        self.min_lin_speed = min_lin_speed
        self.min_ang_speed = min_ang_speed
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.max_integral = np.array(max_integral)
        self.arrive_dist = arrive_dist
        self.arrive_yaw = arrive_yaw
        self.term_dist = term_dist
        self.term_yaw = term_yaw
        self.lookahead_distance = lookahead_distance
        self.current_wp_idx = 0
        self.error_integral = np.zeros(3)
        self.last_error = np.zeros(3)
        self.arrived_at_goal = False
    
    def reset(self):
        self.arrived_at_goal = False
        self.current_wp_idx = 0
        self.error_integral = np.zeros(3)
        self.last_error = np.zeros(3)
    
    def update(self, cam_pos, cam_quat, waypoints_world):
        device = self.device
        current_wp_idx = self.current_wp_idx

        # Empty or invalid path
        if waypoints_world is None or len(waypoints_world) == 0:
            return torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)

        num_wps = len(waypoints_world)
        # Clamp current index to valid range (do not append/alter original path)
        if current_wp_idx >= num_wps:
            current_wp_idx = num_wps - 1

        base_xy = np.array(cam_pos[:2])
        base_yaw = R.from_quat(cam_quat).as_euler('ZYX')[0]
        waypoints_world = np.array(waypoints_world)

        # 1) Find nearest forward waypoint to avoid going backwards
        #    Only consider waypoints from current_wp_idx forward
        dists = [
            (i, np.linalg.norm(waypoints_world[i, :2] - base_xy))
            for i in range(num_wps)
        ]
        nearest_idx, nearest_dist = min(dists, key=lambda t: t[1])
        current_wp_idx = max(current_wp_idx, nearest_idx)

        # 2) Choose a lookahead target a few meters ahead along the path
        #    Use a simple rule: first waypoint at distance >= self.lookahead_distance from robot
        target_idx = current_wp_idx
        for i in range(current_wp_idx, num_wps):
            if dists[i][1] >= self.lookahead_distance:
                target_idx = i
                break
        if target_idx != current_wp_idx:
            self.error_integral = np.zeros(3)
            self.last_error = np.zeros(3)
        current_wp_idx = target_idx
        target_wp = waypoints_world[target_idx]
        dist_to_goal = np.linalg.norm(target_wp[:2] - base_xy)

        # 3) Compute control to the lookahead target in body frame (pure-pursuit style)
        dx = target_wp[0] - base_xy[0]
        dy = target_wp[1] - base_xy[1]
        ang_err = wrap_to_pi(target_wp[2] - base_yaw)

        ex_b, ey_b = world_to_body(dx, dy, base_yaw)

        # Update error integral and derivative
        error = np.array([ex_b, ey_b, ang_err])
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -self.max_integral, self.max_integral)
        error_derivative = error - self.last_error
        self.last_error = error

        # Linear velocities: proportional to lookahead vector, clipped by given maxima
        # TODO: I and D terms should consider dt
        vel = self.kp * error + self.ki * self.error_integral + self.kd * error_derivative
        vel = np.clip(vel, -self.max_vel, self.max_vel)

        # enforce minimum velocities when far from goal
        lin_speed = np.linalg.norm(vel[:2])
        if dist_to_goal > self.arrive_dist and lin_speed < self.min_lin_speed:
            vel[:2] = vel[:2] / lin_speed * self.min_lin_speed
        if abs(ang_err) > self.arrive_yaw and abs(vel[2]) < self.min_ang_speed:
            vel[2] = vel[2] / abs(vel[2]) * self.min_ang_speed

        vx, vy, vw = vel

        print("================================================")
        print(f"cam_pos: {cam_pos}")
        print(f"target_wp: [{target_idx}] {target_wp[:2]} {np.rad2deg(target_wp[2])}")
        print(f"base_xy: {base_xy} base_yaw: {np.rad2deg(base_yaw)}")
        print(f"dx: {dx}, dy: {dy}, ang_err: {np.rad2deg(ang_err)}")
        print(f"ex_b: {ex_b}, ey_b: {ey_b}")
        print(f"vx: {vx}, vy: {vy}, vw: {vw}")


        is_final_segment = (current_wp_idx >= num_wps - 2)
        if is_final_segment:
            # 4) Near-goal handling to avoid singularity and oscillation
            if dist_to_goal < self.term_dist and abs(ang_err) < self.term_yaw:
                vx, vy, vw = 0.0, 0.0, 0.0
                current_wp_idx = num_wps - 1
                self.arrived_at_goal = True
                print(f"[PLAN] Reached goal")
        else:
            # 5) Waypoint progression: progress when near the current waypoint
            if dist_to_goal < self.arrive_dist and abs(ang_err) < self.arrive_yaw:
                current_wp_idx += 1
                print(f"[PLAN] Reached waypoint {current_wp_idx}/{num_wps}")

        command = torch.tensor([[vx, vy, vw]], device=device, dtype=torch.float32)
        self.current_wp_idx = current_wp_idx
        return command