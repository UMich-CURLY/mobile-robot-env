import heapq
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

class AStarPlanner:
    def __init__(
            self,
            occupancy_grid,
            meters_per_cell=0.05,
            time_budget=0.5,
            robot_width=0.55,
            robot_height=1.1,
            step_size=10.0,
            step_size_yaw=45,
            reach_threshold=5.0,
        ):
        self.occupancy_grid = occupancy_grid
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0

        self.robot_width = robot_width # meters
        self.robot_height = robot_height # meters
        self.time_budget = time_budget
        self.step_size = step_size
        self.step_size_yaw = step_size_yaw
        self.reach_threshold = reach_threshold
        # resolution, default 5cm per cell
        self.meters_per_cell = meters_per_cell

        # compute nearest distance to obstacle
        dist_px = distance_transform_edt(self.occupancy_grid.astype(np.uint8) == 0)
        self.dist_m = dist_px * self.meters_per_cell

        # conservative circle radius: half diagonal (rectangle -> circumscribed circle)
        robot_radius = 0.5 * np.hypot(self.robot_width, self.robot_height)
        # when farther than this margin, we skip expensive rectangle test
        self.safe_margin = robot_radius * 1.15  # 15% cushion
    
    def warp_to_pi(self, yaw):
        return (yaw + np.pi) % (2 * np.pi) - np.pi

    def get_neighbors(self, node):
        neighbors = []
        step_size = self.step_size
        step_size_yaw = self.step_size_yaw
        # rel_yaw = 0
        # new_yaw = np.deg2rad(rel_yaw)+node[2]
        # dx = np.cos(new_yaw)*step_size
        # dy = np.sin(new_yaw)*step_size
        # neighbors.append((dx, dy,  0))
        for rel_yaw in range(-step_size_yaw, step_size_yaw+1, step_size_yaw):
            new_yaw = np.deg2rad(rel_yaw)+node[2]
            dx = np.cos(new_yaw)*step_size
            dy = np.sin(new_yaw)*step_size
            neighbors.append((dx, dy,  0))
            if rel_yaw != 0:
                neighbors.append((0, 0, np.deg2rad(rel_yaw)))

        result = []
        for dx, dy, dw in neighbors:
            neighbor = (node[0] + dx, node[1] + dy, self.warp_to_pi(node[2] + dw))
            if 0 <= neighbor[0] < self.occupancy_grid.shape[0] and 0 <= neighbor[1] < self.occupancy_grid.shape[1]:
                if not self.check_collision(neighbor, self.occupancy_grid):  # Only traverse free space (0)
                    result.append(neighbor)
        # print(f"{len(neighbors)}->{len(result)}")
        # print(f"neighbors: {node}->{result}")
        return result

    # def crop_minAreaRect(self, img, rect):
    #     box = cv2.boxPoints(rect)
    #     box = np.intp(box)
    #     # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    #     width = int(rect[1][0])
    #     height = int(rect[1][1])
    #     src_pts = box.astype("float32")
    #     dst_pts = np.array([[0, height-1],
    #                         [0, 0],
    #                         [width-1, 0],
    #                         [width-1, height-1]], dtype="float32")
    #     M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    #     warped = cv2.warpPerspective(img, M, (width, height))
    #     return warped
    
    # def check_collision(self, node, obstacle_map):
    #     robot_w = self.robot_width//self.meters_per_cell
    #     robot_h = self.robot_height//self.meters_per_cell
    #     rect = (node[:2], (robot_w, robot_h), np.rad2deg(node[2])+90)
    #     wall = self.crop_minAreaRect(obstacle_map, rect)
    #     wall_pixels = wall.sum()
    #     return wall_pixels > 0

    def check_collision_roi(self, node, obstacle_map):
        # Build rotated rect (keep your width/height in pixels via //)
        robot_w = int(self.robot_width // self.meters_per_cell)
        robot_h = int(self.robot_height // self.meters_per_cell)
        rect = (node[:2], (robot_w, robot_h), np.rad2deg(node[2]) + 90.0)

        # check out of bounds
        box = cv2.boxPoints(rect)
        x, y, w, h = cv2.boundingRect(box)
        H, W = obstacle_map.shape[:2]
        if x >= W or y >= H or x + w <= 0 or y + h <= 0:
            return True
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return True

        # Crop ROI once; build a local mask by shifting polygon into ROI coordinates
        roi = obstacle_map[y0:y1, x0:x1]
        box_shifted = (box - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, box_shifted, 255)
        hit = cv2.countNonZero((roi > 0).astype(np.uint8) & (mask > 0)) > 0
        return bool(hit)

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
    

    def check_collision(self, node, obstacle_map):
        r = int(round(node[1]))
        c = int(round(node[0]))
        if r < 0 or r >= self.dist_m.shape[0] or c < 0 or c >= self.dist_m.shape[1]:
            return True  # out of bounds collides

        # cheap early-out: far from obstacles => no collision
        if self.dist_m[r, c] >= self.safe_margin:
            return False

        # near obstacles => do exact rectangle test on original grid
        return self.check_collision_roi(node, obstacle_map.astype(np.uint8))
    
    def plot_rect(self, node, obstacle_map, color=100, width=1):
        robot_w = self.robot_width//self.meters_per_cell
        robot_h = self.robot_height//self.meters_per_cell
        rect = (node[:2], (robot_w, robot_h), np.rad2deg(node[2])+90)
        # flip obstacle_map vertically
        wall = self.crop_minAreaRect(obstacle_map, rect)
        vis_img = obstacle_map.copy()
        if vis_img.max() == 1:
            vis_img = vis_img*255
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(vis_img,[box],0,color,width)
        return vis_img, wall
    
    def get_id(self, node):
        return (int(node[0]), int(node[1]), round(np.rad2deg(node[2])))
        # return (int(node[0]), int(node[1]))

    def g_cost(self, p1, p2):
        return np.linalg.norm([p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]])

    def h_cost(self, p1, p2):
        return np.linalg.norm([p1[0]-p2[0], p1[1]-p2[1]])

    def plan(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        cost_so_far = {self.get_id(start): 0} # g_cost
        all_nodes = set()
        start_time = time.time()

        while open_list:
            if time.time() - start_time > self.time_budget:
                print("Time budget exceeded, returning furthest node")
                break
            current_priority, current_node = heapq.heappop(open_list)

            reach_threshold = self.reach_threshold
            distance_to_goal = np.linalg.norm(np.array(current_node[:2]) - np.array(goal[:2]))
            if distance_to_goal<reach_threshold:
                # reached goal!
                came_from[goal] = current_node
                return self.reconstruct_path(came_from, start, goal)

            current_cost = cost_so_far[self.get_id(current_node)]
            for neighbor in self.get_neighbors(current_node):
                new_cost = current_cost + self.g_cost(current_node, neighbor)
                # print(f"current node: {current_node}, cost to goal: {self.g_cost(neighbor, goal)}")
                neighbor_grid = self.get_id(neighbor)

                if neighbor_grid not in cost_so_far or new_cost < cost_so_far[neighbor_grid]:
                    cost_so_far[neighbor_grid] = new_cost
                    all_nodes.add(neighbor)
                    priority = new_cost + self.h_cost(neighbor, goal)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current_node
        print("No path found, returning furthest node")
        best_node = min(all_nodes, key=lambda x: self.h_cost(x, goal))
        return self.reconstruct_path(came_from, start, best_node)

    def reconstruct_path(self, came_from, start, goal):
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
    
    def _world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid indices."""
        grid_x = int((world_x - self.map_origin_x) / self.meters_per_cell)
        grid_y = int((world_y - self.map_origin_y) / self.meters_per_cell)
        return (grid_y, grid_x)  # Note: A* expects (row, col)

    def _grid_to_world(self, grid_row, grid_col):
        """Convert grid indices to world coordinates."""
        world_x = self.map_origin_x + grid_col * self.meters_per_cell
        world_y = self.map_origin_y + grid_row * self.meters_per_cell
        return (world_x, world_y)
    
    
    def _process_path(self, raw_path):
            """Process the raw A* path to make it smoother and better for following."""
            # Step 1: Remove unnecessary points (keep path simple)
            simplified_path = self._simplify_path(raw_path)
            self.path_grid = simplified_path
            
            # Step 2: Create evenly spaced waypoints
            waypoints = self._create_waypoints(simplified_path, spacing=0.1)
            
            return waypoints

    def _simplify_path(self, path):
        """Remove unnecessary points from the path."""
        if len(path) < 3:
            return path
        
        simplified = [path[0]]  # Always keep start point
        
        i = 0
        while i < len(path) - 1:
            furthest = i + 1        # Look ahead to see how far we can go in a straight line
            
            for j in range(i + 2, min(i + 6, len(path))):  # Look ahead max 5 points
                if self._can_connect_directly(path[i], path[j]) and path[i][2] == path[j][2]:
                    furthest = j
                else:
                    break
            
            simplified.append(path[furthest])
            i = furthest
        
        return simplified

    def _can_connect_directly(self, point1, point2, interval=2.0):
        """Check if two points can be connected with a straight line (no obstacles)."""
        point1, point2 = np.array(point1), np.array(point2)
        
        # Sample points along the line between point1 and point2
        dist = np.linalg.norm(point2[:2] - point1[:2])
        steps = dist/interval
        
        for i in range(1, int(steps)):  # Skip start and end points
            t = i / steps
            x, y, yaw = point1 + t*(point2 - point1)
            if self.check_collision((x, y, yaw), self.occupancy_grid):
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
            world_x, world_y = self._grid_to_world(row, col)
            world_path.append(np.array([world_x, world_y, yaw]))
        return world_path


    def plan_and_publish_path(self, current_grid_pose, goal_pose):
        """Main path planning function."""
        # Check if we have everything needed
        # if not self._can_plan():              # probably need this afterwards
        #     return
        
        # Convert world coordinates to grid coordinates
        start_grid = current_grid_pose
        goal_grid = goal_pose

        # Run A* path planning
        time_start = time.time()
        raw_path = self.plan(start_grid, goal_grid)
        print(f"A* planning time: {time.time() - time_start} seconds")
        if raw_path is None:
            print("A* failed to find a path")
            return

        time_start = time.time()
        # Process the path to make it smoother and better spaced    
        final_path = self._process_path(raw_path)
        print(f"path processing time: {time.time() - time_start} seconds")
        return final_path


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
    astar = AStarPlanner(occupancy_grid_map)

    plt.figure()
    plt.imshow(astar.occupancy_grid, cmap='binary')
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

    path = astar.plan_and_publish_path(start, goal)

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
