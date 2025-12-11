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

def calc_yaw(current_pos, target_pos):
    return np.arctan2((target_pos[1] - current_pos[1]), target_pos[0] - current_pos[0])

def set_yaw_to_forward(cam_pos, waypoints):
    last_pos = cam_pos[:2]
    new_waypoints = []
    for wp in waypoints:
        yaw_angle = calc_yaw(last_pos, wp[:2])
        new_waypoints.append([wp[0], wp[1], yaw_angle])
        last_pos = wp[:2]
    return new_waypoints

class WaypointFollower:
    def __init__(self,
        device,
        max_vel=[1.5, 1.0, 1.0],
        min_vel=[0.3, 0.1, 0.3],
        kp=[2.0, 0.5, 2.0],
        ki=[0.0, 0.0, 0.0], # [0.1, 0.1, 0.1]
        kd=[0.0, 0.0, 0.0], # [3, 3, 1]
        max_integral=[5.0, 5.0, 3.0],
        arrive_dist=0.1,
        arrive_yaw=np.pi/180.0*15.0,
        term_dist=0.05,
        term_yaw=np.pi/180.0*5.0,
        lookahead_distance=0,
    ):
        self.device = device
        self.max_vel = np.array(max_vel)
        self.min_vel = np.array(min_vel)
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
    
    def update(self, cam_pos, cam_quat, waypoints_world, verbose=False):
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
        is_final_segment = (current_wp_idx == num_wps - 1)
        target_wp = waypoints_world[target_idx]

        # set different arrive thresholds for final segment
        if is_final_segment:
            arrive_dist = self.term_dist
            arrive_yaw = self.term_yaw
        else:
            arrive_dist = self.arrive_dist
            arrive_yaw = self.arrive_yaw

        # calculate xy diff
        diff_to_goal = target_wp[:2] - base_xy
        dist_to_goal = np.linalg.norm(diff_to_goal)

        # set yaw to next waypoint and ignore set yaw angle until reaching the waypoint position
        # or, if the set yaw is inf, set yaw to next waypoint too
        if dist_to_goal > arrive_dist or target_wp[2]>10.0:
            target_wp = np.array(target_wp)
            target_wp[2] = calc_yaw(base_xy, target_wp[:2])

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

        # enforce minimum velocities when far from goal
        for i, diff in enumerate([ex_b, ey_b]):
            if abs(diff) > arrive_dist and abs(vel[i]) < self.min_vel[i]:
                vel[i] = vel[i] / abs(vel[i]) * self.min_vel[i]
        if abs(ang_err) > arrive_yaw and abs(vel[2]) < self.min_vel[2]:
            vel[2] = vel[2] / abs(vel[2]) * self.min_vel[2]

        # if the angle error is too large, set linear velocity to 0
        if ang_err > np.deg2rad(30.0):
            vel[0] = 0.0
            vel[1] = 0.0

        vel = np.clip(vel, -self.max_vel, self.max_vel)
        vx, vy, vw = vel

        if verbose:
            print("================================================")
            print(f"cam_pos: {cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}")
            print(f"target_wp: [{target_idx}] {target_wp[0]:.2f}, {target_wp[1]:.2f}, {np.rad2deg(target_wp[2]):.2f}")
            print(f"base_xy: {base_xy[0]:.2f}, {base_xy[1]:.2f} base_yaw: {np.rad2deg(base_yaw):.2f}")
            print(f"dx: {dx:.2f}, dy: {dy:.2f}, ang_err: {np.rad2deg(ang_err):.2f}")
            print(f"ex_b: {ex_b:.2f}, ey_b: {ey_b:.2f}")
            print(f"vx: {vx:.2f}, vy: {vy:.2f}, vw: {vw:.2f}")
            print(f"current_wp_idx: {current_wp_idx}, num_wps: {num_wps}")


        if dist_to_goal < arrive_dist and abs(ang_err) < arrive_yaw:
            if is_final_segment:
                vx, vy, vw = 0.0, 0.0, 0.0
                current_wp_idx = num_wps - 1
                self.arrived_at_goal = True
                print(f"[PLAN] Reached goal")
            else:
                current_wp_idx += 1
                print(f"[PLAN] Reached waypoint {current_wp_idx}/{num_wps}")

        command = torch.tensor([[vx, vy, vw]], device=device, dtype=torch.float32)
        self.current_wp_idx = current_wp_idx
        return command