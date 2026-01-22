import heapq
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def generate_kernels():
    """
    Generate 4 different 3x3 kernels where each has one 1 and one -1 at opposite ends.

    Returns:
        np.ndarray: An array of shape (4, 3, 3) containing the four kernels.
    """
    kernels = np.zeros((4, 3, 3))  # Initialize 4 kernels with zeros
    
    positions = [((0, 0), (2, 2)),  # Top-left to bottom-right
                ((0, 2), (2, 0)),  # Top-right to bottom-left
                ((0, 1), (2, 1)),  # Top-center to bottom-center
                ((1, 0), (1, 2))]  # Middle-left to middle-right

    for i, (pos1, pos2) in enumerate(positions):
        kernels[i, pos1[0], pos1[1]] = 1   # Set 1 at one end
        # kernels[i, 1, 1] = 1   # Set 1 at one end
        kernels[i, pos2[0], pos2[1]] = -1  # Set -1 at the opposite end
    
    return kernels

def load_bev_map(bev_map_path):
    bev_data = np.load(bev_map_path)
    bev_depth = bev_data['depth'][:,:,0]
    info = json.load(open(f"{bev_map_path.replace('.npz', '_info.txt')}"))
    return bev_depth, info

def generate_obstacle_map(bev_depth, max_climb_height=0.25):
    image_width, image_height = bev_depth.shape
    height_map = np.ones((image_width, image_height))*(-1000.0)
    height_map[bev_depth>0] = bev_depth[bev_depth>0]

    # hand desgined gradient kernel that has physical unit
    kernels = generate_kernels()
    filtered_img = []
    for kernel in kernels:
        img = cv2.filter2D(src=height_map, ddepth=-1, kernel=kernel)
        filtered_img.append(img)
    filtered_img = np.stack(filtered_img)  # Shape: (4, height, width)
    # take the max across the 4 kernels to get the final gradient map
    gradient_map = np.max(np.abs(filtered_img), axis=0)

    # calculate obstacle map
    obstacle_map = (gradient_map > max_climb_height).astype(np.uint8)

    return obstacle_map


class AStarPlanner:
    def __init__(
        self,
        meters_per_cell=0.05,
        step_size=0.1,
        step_size_yaw=45,
        reach_threshold=1.0,
        reach_yaw=45.0,
        robot_width=0.4,
        robot_height=0.8,
    ):

        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.step_size_px = step_size//meters_per_cell
        self.step_size_yaw = step_size_yaw
        self.reach_threshold = reach_threshold//meters_per_cell
        self.reach_yaw = reach_yaw
        self.robot_width = robot_width # meters
        self.robot_height = robot_height # meters

        # resolution, default 5cm per cell
        self.meters_per_cell = meters_per_cell

        # conservative circle radius: half diagonal (rectangle -> circumscribed circle)
        self.robot_radius = 0.5 * np.hypot(self.robot_width, self.robot_height)
        # when farther than this margin, we skip expensive rectangle test
        self.safe_margin = self.robot_radius * 1.15  # 15% cushion

    def plan(self, occupancy_grid, start, goal, time_budget=0.5):
        # compute nearest distance to obstacle
        dist_px = distance_transform_edt(occupancy_grid.astype(np.uint8) == 0)
        self.dist_m = dist_px * self.meters_per_cell

        # Adjust start and goal yaw
        start = self.find_closest_yaw(start)
        start = self.find_noncollide_neighbor(start, occupancy_grid)
        goal = self.find_closest_yaw(goal)
        goal = self.find_noncollide_neighbor(goal, occupancy_grid)

        # start A* planning
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        cost_so_far = {self.get_id(start): 0} # g_cost
        all_nodes = set()
        start_time = time.time()

        while open_list:
            if time.time() - start_time > time_budget:
                print("Time budget exceeded, returning furthest node")
                break
            current_priority, current_node = heapq.heappop(open_list)

            distance_to_goal = np.linalg.norm(np.array(current_node[:2]) - np.array(goal[:2]))
            yaw_diff = np.abs(self.warp_to_pi(current_node[2] - goal[2]))
            if distance_to_goal<self.reach_threshold and yaw_diff<self.reach_yaw:
                # reached goal!
                came_from[goal] = current_node
                return self.reconstruct_path(came_from, start, goal)

            current_cost = cost_so_far[self.get_id(current_node)]
            for neighbor in self.get_neighbors(current_node, occupancy_grid):
                new_cost = current_cost + self.g_cost(current_node, neighbor)
                dist = np.linalg.norm([neighbor[1] - current_node[1], neighbor[0] - current_node[0]])
                if dist >= 0.5*self.step_size_px:
                    heading_angle = np.arctan2(neighbor[1] - current_node[1], neighbor[0] - current_node[0])
                    angle_diff = np.abs(self.warp_to_pi(heading_angle - neighbor[2]))
                    new_cost += angle_diff*50.0
                # new_cost = current_cost + self.g_cost(current_node, neighbor) + angle_diff*5.0
                # print(f"current node: {current_node}, cost to goal: {self.g_cost(neighbor, goal)}")
                neighbor_grid = self.get_id(neighbor)

                if neighbor_grid not in cost_so_far or new_cost < cost_so_far[neighbor_grid]:
                    cost_so_far[neighbor_grid] = new_cost
                    all_nodes.add(neighbor)
                    priority = new_cost + self.h_cost(neighbor, goal)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current_node
        print("No path found, returning furthest node")
        if len(all_nodes) == 0:
            return
        best_node = min(all_nodes, key=lambda x: self.h_cost(x, goal))
        raw_path = self.reconstruct_path(came_from, start, best_node)
        self.raw_path = raw_path

        # process the path
        print(f"A* planning time: {time.time() - start_time} seconds")
        if raw_path is None:
            print("A* failed to find a path")
            return
        start_time = time.time()
        final_path = self.simplify_path(raw_path, occupancy_grid)
        self.final_path = final_path
        return final_path

    def get_neighbors(self, node, occupancy_grid, relax=False, collision_check=True):
        neighbors = []
        step_size_px = self.step_size_px
        step_size_yaw = self.step_size_yaw
        for rel_yaw in range(-step_size_yaw, step_size_yaw+1, step_size_yaw):
            new_yaw = np.deg2rad(rel_yaw)+node[2]
            dx = np.cos(new_yaw)*step_size_px
            dy = np.sin(new_yaw)*step_size_px
            # rotate and forward one step size
            neighbors.append((dx, dy,  np.deg2rad(rel_yaw)))
            if rel_yaw != 0:
                # pure rotation
                neighbors.append((0, 0, np.deg2rad(rel_yaw)))
        # backward one step size
        # neighbors.append((0, -step_size_px, 0))

        if collision_check:
            if relax:
                check_func = lambda x: self.check_collision(x, occupancy_grid, relax=relax)
            else:
                check_func = lambda x: self.check_collision(x, occupancy_grid, relax=relax, parent_node=node)
        else:
            check_func = lambda x: False
        result = []
        for dx, dy, dw in neighbors:
            neighbor = (node[0] + dx, node[1] + dy, self.warp_to_pi(node[2] + dw))
            if 0 <= neighbor[0] < occupancy_grid.shape[0] and 0 <= neighbor[1] < occupancy_grid.shape[1]:
                if not check_func(neighbor):
                    # Only traverse free space
                    result.append(neighbor)
        # print(f"{len(neighbors)}->{len(result)}")
        # print(f"neighbors: {node}->{result}")
        return result
    
    def find_closest_yaw(self, node):
        yaw_list = [np.deg2rad(yaw) for yaw in range(-180, 180+1, self.step_size_yaw)]
        best_yaw = np.argmin([np.abs(self.warp_to_pi(yaw - node[2])) for yaw in yaw_list])
        return (node[0], node[1], yaw_list[best_yaw])

    def find_noncollide_neighbor(self, start, occupancy_grid):
        max_attempts = 5
        open_list = []
        visited_list = []

        start_hit = self.check_collision_roi(start, occupancy_grid)[1]
        heapq.heappush(open_list, (start_hit, start))
        best_node = start
        best_hit = start_hit

        def get_neighbors(node):
            return [
                (node[0] + self.step_size_px, node[1], node[2]),
                (node[0] - self.step_size_px, node[1], node[2]),
                (node[0], node[1] + self.step_size_px, node[2]),
                (node[0], node[1] - self.step_size_px, node[2]),
                # (node[0], node[1], node[2] + self.step_size_yaw),
                # (node[0], node[1], node[2] - self.step_size_yaw),
            ]

        for i in range(max_attempts):
            if best_hit == 0 or len(open_list) == 0:
                break
            node = heapq.heappop(open_list)[1]
            visited_list.append(self.get_id(node))
            neighbors = get_neighbors(node)
            for neighbor in neighbors:
                if self.get_id(neighbor) not in visited_list:
                    hit = self.check_collision_roi(neighbor, occupancy_grid, robot_width=self.robot_width, robot_height=self.robot_height)[1]
                    if hit < best_hit:
                        best_hit = hit
                        best_node = neighbor
                    heapq.heappush(open_list, (hit, neighbor))
        
        # print(f"start: {start}, start hit: {start_hit}, best node: {best_node}, best hit: {best_hit}")
        return tuple(best_node)

    def check_collision_roi(self, node, obstacle_map, parent_node=None, robot_width=None, robot_height=None):
        # Build rotated rect
        if robot_width is None:
            robot_width = self.robot_width
        if robot_height is None:
            robot_height = self.robot_height
        robot_w = int(robot_width // self.meters_per_cell)
        robot_h = int(robot_height // self.meters_per_cell)
        rect = (node[:2], (robot_w, robot_h), np.rad2deg(node[2]) + 90.0)

        # check out of bounds
        box = cv2.boxPoints(rect)
        
        if parent_node is not None:
            rect_parent = (parent_node[:2], (robot_w, robot_h), np.rad2deg(parent_node[2]) + 90.0)
            box_parent = cv2.boxPoints(rect_parent)
            
            # Ensure both boxes are float32 before concatenation for convexHull
            box = box.astype(np.float32)
            box_parent = box_parent.astype(np.float32)
            
            # ConvexHull expects float32 or int32 points
            box = cv2.convexHull(np.concatenate((box, box_parent)))
            box = box.reshape(-1, 2)

        # Convert to integer coordinates for boundingRect
        box_int = np.int32(box)
        x, y, w, h = cv2.boundingRect(box_int)
        
        H, W = obstacle_map.shape[:2]
        if x >= W or y >= H or x + w <= 0 or y + h <= 0:
            return True
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return True

        # Crop ROI once; build a local mask by shifting polygon into ROI coordinates
        roi = obstacle_map[y0:y1, x0:x1]
        
        # Ensure box is in correct format for fillConvexPoly
        # It needs to be integer coordinates relative to the ROI
        box_shifted = (box - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, box_shifted, 255)
        hit = cv2.countNonZero((roi > 0).astype(np.uint8) & (mask > 0))
        return hit>5, hit, roi, mask

    def crop_minAreaRect(self, img, rect):
        # check out of bounds
        box = cv2.boxPoints(rect)
        x, y, w, h = cv2.boundingRect(box)
        H, W = img.shape[:2]
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)

        # Crop ROI once; build a local mask by shifting polygon into ROI coordinates
        roi = img[y0:y1, x0:x1]
        box_shifted = (box - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, box_shifted, 255)
        return roi
    

    def check_collision(self, node, obstacle_map, relax=False, parent_node=None):
        """Check if the node is in collision with the obstacle map.
        Return True if in collision, False otherwise.
        """
        # r = int(round(node[1]))
        # c = int(round(node[0]))
        r = int(round(node[1]))
        c = int(round(node[0]))
        if r < 0 or r >= self.dist_m.shape[0] or c < 0 or c >= self.dist_m.shape[1]:
            return True  # out of bounds collides
        # cheap early-out: far from obstacles => no collision
        if parent_node is None and self.dist_m[r, c] >= self.safe_margin:
            return False

        # skip roi check if relax is True
        if relax:
            return False

        # near obstacles => do exact rectangle test on original grid
        return self.check_collision_roi(node, obstacle_map.astype(np.uint8), parent_node=parent_node)[0]

    def plot_rect(self, node, obstacle_map, color=100, width=1):
        robot_w = self.robot_width//self.meters_per_cell
        robot_h = self.robot_height//self.meters_per_cell
        rect = (node[:2], (robot_w, robot_h), np.rad2deg(node[2])+90)
        # flip obstacle_map vertically
        wall = self.crop_minAreaRect(obstacle_map, rect)
        vis_img = obstacle_map.copy()
        if vis_img.dtype == bool:
            vis_img = vis_img.astype(np.uint8) * 255
        elif vis_img.max() <= 1:
            vis_img = (vis_img*255).astype(np.uint8)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(vis_img,[box],0,color,1)
        return vis_img, wall
    
    def get_id(self, node):
        return (int(node[0]), int(node[1]), round(np.rad2deg(node[2])))
        # return (int(node[0]), int(node[1]))

    def g_cost(self, p1, p2):
        diff = [p1[0]-p2[0], p1[1]-p2[1], self.warp_to_pi(p1[2]-p2[2])]
        diff[2] *= 10
        return np.linalg.norm(diff)

    def h_cost(self, p1, p2):
        return np.linalg.norm([p1[0]-p2[0], p1[1]-p2[1]])

    def reconstruct_path(self, came_from, start, goal):
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
    
    def warp_to_pi(self, yaw):
        return (yaw + np.pi) % (2 * np.pi) - np.pi

    def _world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid indices."""
        grid_x = int((world_x - self.map_origin_x) / self.meters_per_cell)
        grid_y = int((world_y - self.map_origin_y) / self.meters_per_cell)
        return (grid_y, grid_x)  # Note: A* expects (row, col)

    def _grid_to_world(self, grid_row, grid_col):
        """Convert grid indices to world coordinates."""
        # world_x = self.map_origin_x + grid_col * self.meters_per_cell
        # world_y = self.map_origin_y + grid_row * self.meters_per_cell
        world_x = self.map_origin_x + grid_col * 0.25
        world_y = self.map_origin_y + grid_row * 0.25

        return (world_x, world_y)

    def simplify_path(self, path, occupancy_grid):
        """Remove unnecessary points from the path."""
        if len(path) < 3:
            return path
        
        simplified = [path[0]]  # Always keep start point
        
        i = 0
        while i < len(path) - 1:
            furthest = i + 1        # Look ahead to see how far we can go in a straight line
            
            for j in range(i + 2, min(i + 6, len(path))):  # Look ahead max 5 points
                if self._can_connect_directly(path[i], path[j], occupancy_grid) and path[i][2] == path[j][2]:
                    furthest = j
                else:
                    break
            
            simplified.append(path[furthest])
            i = furthest
        
        return simplified

    def _can_connect_directly(self, point1, point2, occupancy_grid, interval=2.0):
        """Check if two points can be connected with a straight line (no obstacles)."""
        point1, point2 = np.array(point1), np.array(point2)
        
        # Sample points along the line between point1 and point2
        dist = np.linalg.norm(point2[:2] - point1[:2])
        steps = dist/interval
        
        for i in range(1, int(steps)):  # Skip start and end points
            t = i / steps
            x, y, yaw = point1 + t*(point2 - point1)
            if self.check_collision((x, y, yaw), occupancy_grid):
                return False
        return True
    
    def _create_waypoints(self, path_points, spacing=0.1):
        """Create evenly spaced waypoints along the path."""
        if len(path_points) < 2:
            # Single point path
            world_x, world_y = self._grid_to_world(path_points[0][0], path_points[0][1])
            return [(world_x, world_y)]
        
        waypoints = []
        
        # Convert all points to world coordinates
        world_path = []
        for row, col, yaw in path_points:
            world_x, world_y = self._grid_to_world(col, row)
            world_path.append(np.array([world_x, world_y, yaw]))
        return world_path


if __name__ == "__main__":
    # Create a 100x100 occupancy grid (1 = obstacle, 0 = free)
    occupancy_grid_map = np.zeros((100, 100), dtype=np.uint8)

    ## Add a few large obstacle clusters
    occupancy_grid_map[2:20, 16:50] = 1
    occupancy_grid_map[50:75, 40:75] = 1
    occupancy_grid_map[70:98, 6:36] = 1
    occupancy_grid_map[10:15, 70:78] = 1
    occupancy_grid_map[48:55, 75:85] = 1
    occupancy_grid_map[15:25, 50:55] = 1

    ## Add boundary walls
    occupancy_grid_map[0, :] = 1
    occupancy_grid_map[-1, :] = 1
    occupancy_grid_map[:, 0] = 1
    occupancy_grid_map[:, -1] = 1

    # ----------------------------------------------------------
    # Initialize A* planner
    astar = AStarPlanner(meters_per_cell=0.05)
    astar.occupancy_grid = occupancy_grid_map == 0

    plt.figure()
    plt.imshow(occupancy_grid_map, cmap='binary')
    plt.title("Inflated Obstacles With Original Boundaries in Grid Frame")
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.colorbar(label="Occupancy")

    ## Plot original boundaries
    obstacle_mask = (occupancy_grid_map > 0).astype(np.uint8)
    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    label_added = False
    for contour in contours:
        contour = contour.squeeze(axis=1)
        if contour.ndim == 2 and contour.shape[0] > 1:
            xs = np.r_[contour[:, 0], contour[0, 0]]
            ys = np.r_[contour[:, 1], contour[0, 1]]
            plt.plot(
                xs,
                ys,
                color='cyan',
                linewidth=1.2,
                label='Original Boundary' if not label_added else None,
            )
            label_added = True

    start = (15,40, np.deg2rad(0))
    goal = (85, 85, np.deg2rad(90))

    vis_img = astar.occupancy_grid.copy()
    for point in [start, goal]:
        vis_img, wall = astar.plot_rect(point, vis_img)
    plt.imshow(vis_img, cmap='binary')
    for x, y, yaw in [start, goal]:
        plt.quiver(x, y, np.cos(yaw), np.sin(yaw),
                angles='xy', scale_units='xy', scale=0.1, color='blue', width=0.01, label='Yaw Direction')
    plt.show()

    print("in utils.py, the start is", start, "and goal is ", goal)
    path = astar.plan(occupancy_grid_map, start, goal)

    print(f"path: {np.array(path).shape}")
    # path = astar.plan(start, goal)
    vis_img = astar.occupancy_grid.copy()
    for x, y, yaw in astar.path_grid:
        vis_img, wall = astar.plot_rect((x, y, yaw), vis_img, 30)
    plt.imshow(vis_img, cmap='binary')
    if astar.path_grid:
        path = np.array(astar.path_grid)
        plt.scatter(path[:, 0], path[:, 1], color='red', s=3, label='Planned Path')
        plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=1.5)
        plt.scatter(start[0], start[1], color='blue', s=50, marker='.', label='Start')
        plt.scatter(goal[0], goal[1], color='blue', s=50, marker='*', label='Goal')
        plt.legend()

        x,y,yaw = path[:,0], path[:,1], path[:,2]

        plt.quiver(x, y, np.cos(yaw), np.sin(yaw), np.arange(len(yaw)),
                angles='xy', scale_units='xy', scale=0.1, cmap='viridis', width=0.01, label='Yaw Direction')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.axis("equal")
        plt.title("Waypoints with Yaw Directions in World Frame")
        plt.show()