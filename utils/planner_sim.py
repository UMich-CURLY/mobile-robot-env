
import numpy as np
import matplotlib.pyplot as plt
import planner as pl # Import the provided planner.py
np.random.seed(1)
class SimulatedHolonomicRobot:
    """
    A simple simulator for a holonomic robot.
    The robot's state is (x, y, theta).
    It accepts velocity commands in its own body frame (v_x_robot, v_y_robot, omega).
    """
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta  # radians

    def update_state(self, v_x_robot, v_y_robot, omega, dt_sim):
        """
        Updates the robot's state based on control commands and simulation time step.
        Args:
            v_x_robot (float): Velocity command along the robot's x-axis.
            v_y_robot (float): Velocity command along the robot's y-axis.
            omega (float): Angular velocity command.
            dt_sim (float): Simulation time step.
        """
        # Calculate change in world frame based on robot frame velocities
        dx_world = (v_x_robot * np.cos(self.theta) - v_y_robot * np.sin(self.theta)) * dt_sim
        dy_world = (v_x_robot * np.sin(self.theta) + v_y_robot * np.cos(self.theta)) * dt_sim
        dtheta = omega * dt_sim

        self.x += dx_world
        self.y += dy_world

        self.x += np.random.randn()*dt_sim*0.1
        self.y += np.random.randn()*dt_sim*0.1

        self.theta += dtheta
        self.theta += np.random.randn()*dt_sim*0.5

        # Normalize theta to be within -pi to pi for consistency
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def get_pose(self):
        """Returns the current pose (x, y, theta) of the robot."""
        return self.x, self.y, self.theta

def main_simulation():
    """
    Main function to run the simulation and visualization.
    """
    plt.style.use('seaborn-v0_8-whitegrid') # Match style with planner.py

    # 1. Define Waypoints for the planner
    # These are the coarse waypoints the robot should try to follow.
    # You can change these to test different trajectories.
    # Example 1: A simple S-curve
    # points = np.array([
    #     [0, 0], [2, 2], [4, 0], [6, -2], [8, 0]
    # ])
    # Example 2: From planner.py (the noisy one)
    # points = np.array([
    #     [0, 0], [1, -2.5], [2, -3], [3, -2],
    #     [4, -2], [5, -1], [6, 1.2], [7, 2.2],
    #     [7.5, 4], [7, 6], [6, 7], [4.5, 7.2],
    #     [3, 6], [2, 4.5], [1.5, 3.5]
    # ])
    # Example 3: A sharper turn
    # points = np.array([
    # [0,0],[0.5,0.2],[1,0],[1.5,-0.2],[2,0]
    # ])

    # points = np.array([
    # [0,0],[0.5,0],[1,0],[1.5,0.5],[1,1],[0.5,1],[0,1]
    # ])*2
    points = np.array([
    [0,0],[0.5,0],[1,0.2],[1.5,0],[2,-0.2],[2.5,0]
    ])

    # 2. Initialize Planner
    lookahead_distance = 0.2  # Pure pursuit lookahead distance
    robot_planner = pl.Planner()#lookahead=lookahead_distance,max_vy=0,cruise_vel=0.6,max_vw=1.5)
    robot_planner.update_waypoints(points) # This generates the smoothed trajectory

    # Get the smoothed path from the planner for reference
    planned_wps_x = robot_planner.wps[:, 0]
    planned_wps_y = robot_planner.wps[:, 1]

    # 3. Initialize Robot
    # Start the robot at the beginning of the smoothed path
    initial_x =   planned_wps_x[0]
    initial_y =   planned_wps_y[0]
    # Estimate initial heading from the first two points of the smoothed path
    if len(planned_wps_x) > 1:
        initial_theta = np.arctan2(planned_wps_y[1] - planned_wps_y[0],
                                   planned_wps_x[1] - planned_wps_x[0])
    else:
        initial_theta = 0.0
    initial_theta = np.pi
    robot = SimulatedHolonomicRobot(x=initial_x, y=initial_y, theta=initial_theta)

    # 4. Simulation Parameters
    dt_sim = 0.01         # Simulation time step (seconds)
    max_sim_time = 60      # Maximum simulation time (seconds)
    num_sim_steps = int(max_sim_time / dt_sim)
    goal_threshold = 0.05  # Distance to final waypoint to consider goal reached

    # Store history for plotting
    robot_path_history = []  # List to store (x, y, theta) at each step
    control_effort_history = [] # List to store (v_x_cmd, v_y_cmd, omega_cmd)
    time_log = []

    # 5. Simulation Loop
    for step in range(num_sim_steps):
        current_time = step * dt_sim
        time_log.append(current_time)

        current_x, current_y, current_theta = robot.get_pose()
        robot_path_history.append((current_x, current_y, current_theta))

        # Check if goal is reached (close to the last point of the *smoothed* path)
        dist_to_goal = np.linalg.norm([current_x - planned_wps_x[-1],
                                       current_y - planned_wps_y[-1]])
        if dist_to_goal < goal_threshold:
            print(f"Goal reached at time {current_time:.2f}s, distance: {dist_to_goal:.3f}m.")
            break

        # Get control commands from the planner
        # planner.step() returns (v_x_robot_cmd, v_y_robot_cmd, omega_cmd)
        v_x_cmd, v_y_cmd, omega_cmd = robot_planner.step(current_x, current_y, current_theta)
        control_effort_history.append((v_x_cmd, v_y_cmd, omega_cmd))
        # Update robot state using the commands
        robot.update_state(v_x_cmd, v_y_cmd, omega_cmd, dt_sim)

    if step == num_sim_steps - 1:
        print(f"Max simulation time reached ({max_sim_time}s).")

    # Convert history lists to numpy arrays for easier plotting
    robot_path_history = np.array(robot_path_history)
    control_effort_history = np.array(control_effort_history)

    # 6. Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [3, 1.5]})
    plt.subplots_adjust(hspace=0.3)

    # --- Trajectory Plot (axs[0]) ---
    ax_traj = axs[0]
    # Plot original coarse waypoints
    ax_traj.plot(points[:, 0], points[:, 1], 'ro', markersize=8, label='Original Waypoints')
    # Plot the smoothed path generated by the planner
    ax_traj.plot(planned_wps_x, planned_wps_y, 'g--', lw=2, label='Planned Smoothed Path')
    # Plot the actual path taken by the robot
    if robot_path_history.size > 0:
        ax_traj.plot(robot_path_history[:, 0], robot_path_history[:, 1], 'b-', lw=1.5, label='Robot Path')

        # Visualize robot heading at intervals
        num_heading_arrows = 25
        path_len = len(robot_path_history)
        arrow_indices = np.linspace(0, path_len - 1, num_heading_arrows, dtype=int)
        arrow_length = lookahead_distance * 0.4 # Adjust for visual clarity

        for i in arrow_indices:
            x, y, theta = robot_path_history[i]
            ax_traj.arrow(x, y,
                          arrow_length * np.cos(theta),
                          arrow_length * np.sin(theta),
                          head_width=lookahead_distance*0.15, head_length=lookahead_distance*0.2,
                          fc='blue', ec='blue', alpha=0.6)
        
        # Mark start and end of robot's path
        ax_traj.scatter(robot_path_history[0, 0], robot_path_history[0, 1],
                        color='cyan', s=100, ec='black', label='Robot Start', zorder=5)
        ax_traj.scatter(robot_path_history[-1, 0], robot_path_history[-1, 1],
                        color='magenta', s=100, ec='black', label='Robot End', zorder=5)

    # Mark the target goal point
    ax_traj.scatter(planned_wps_x[-1], planned_wps_y[-1],
                    color='lime', marker='x', s=150, lw=3, label='Target Goal', zorder=5)

    ax_traj.set_title('Robot Trajectory Following Simulation', fontsize=16)
    ax_traj.set_xlabel('X coordinate (m)', fontsize=12)
    ax_traj.set_ylabel('Y coordinate (m)', fontsize=12)
    ax_traj.legend(fontsize=10)
    ax_traj.axis('equal')
    ax_traj.grid(True)

    # --- Control Efforts Plot (axs[1]) ---
    ax_ctrl = axs[1]
    if control_effort_history.size > 0:
        # Time vector should match the length of control_effort_history
        plot_time = time_log[:len(control_effort_history)]

        ax_ctrl.plot(plot_time, control_effort_history[:, 0], label='$v_x$ cmd (robot frame) [m/s-like]')
        ax_ctrl.plot(plot_time, control_effort_history[:, 1], label='$v_y$ cmd (robot frame) [m/s-like]')
        ax_ctrl.plot(plot_time, control_effort_history[:, 2], label='$\omega$ cmd [rad/s-like]')
        ax_ctrl.set_title('Control Efforts Over Time', fontsize=16)
        ax_ctrl.set_xlabel('Time (s)', fontsize=12)
        ax_ctrl.set_ylabel('Control Command Value', fontsize=12)
        ax_ctrl.legend(fontsize=10)
        ax_ctrl.grid(True)
    else:
        ax_ctrl.text(0.5, 0.5, "No control data to plot.", ha='center', va='center')


    plt.show()

if __name__ == "__main__":
    main_simulation()