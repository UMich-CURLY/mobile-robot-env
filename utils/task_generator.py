import yaml
import os
import re
from scipy.spatial import KDTree
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__+'/..')) + "/path_navmesh")
# from utils import navmesh_utils
import numpy as np
from utils.episode import VLNEpisodes

class TaskGenerator:
    def __init__(self, args, env, sim, config_path):
        self.args = args
        self.env = env
        self.sim = sim
        self.manager_env = env.unwrapped.unwrapped
        self.task_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.episode_config = self.task_config['scene']['nvidia_park']['episodes']
        self.episode_goals = self.episode_config['goals']
        self.num_episodes = self.episode_config['number']
    
    def reset(self):
        self.load_scene()
        self.parse_scene()
        # self.generate_navmesh()
        return True

    def parse_scene(self):
        # find target prims
        self.prim_list = [x for x in self.manager_env.scene.stage.Traverse()]
        print(f'Loaded {len(self.prim_list)} prims')
        self.goal_dict = {}
        for goal, goal_rule in self.episode_goals.items():
            goal_prim = [x for x in self.prim_list if re.search(goal_rule, x.GetName())]
            goal_prim = [x for x in goal_prim if x.HasAttribute('xformOp:translate')]
            self.goal_dict[goal] = {
                'prim': goal_prim,
                'pos': [list(x.GetAttribute('xformOp:translate').Get()) for x in goal_prim]
            }
            print(f'{goal}: Found {len(goal_prim)} prims')
        # self.goal_kd_tree = KDTree(self.goal_positions.values())
    
    def generate_navmesh(self):
        navmesh_path = self.args.navmesh_path
        navmesh_interface = navmesh_utils.NavmeshInterface(up_axis='Z', stage=self.manager_env.scene.stage)
        self.navmesh_interface = navmesh_interface
        if os.path.exists(navmesh_path):
            navmesh_interface.setup_navmesh_from_file(navmesh_path)
        else:
            selected_paths = ["/World/ground/terrain"]
            navmesh_interface.setup_navmesh(selected_paths)

        # Build the navmesh
        navmesh_interface.build_navmesh({
            "cellSize": 0.2,
            "cellHeight": 0.2,
            "agentHeight": 0.61,
            "agentRadius": 0.55,
            "agentMaxClimb": 0.1,
            "agentMaxSlope": 26.0,
            "regionMinSize": 4,
            "regionMergeSize": 20,
            "edgeMaxLen": 5.0,
            "edgeMaxError": 1.3,
            "vertsPerPoly": 6.0,
            "detailSampleDist": 6.0,
            "detailSampleMaxError": 1.0,
            "partitionType": 0
        })

        # Visualize the navmesh
        navmesh_interface.visualize_navmesh()
        
        # sample random points
        self.random_points = navmesh_interface.get_random_points(self.num_episodes*10)
        navmesh_utils.create_points(self.random_points, prim_path='/World/RandomPoints')

        self.sampled_traj = []
        while len(self.sampled_traj) < self.num_episodes:
            start = self.random_points[np.random.randint(0, len(self.random_points))]
            random_goal = np.random.choice(list(self.goal_dict.keys()))
            random_goal_pos = self.goal_dict[random_goal]['pos']
            end = random_goal_pos[np.random.randint(0, len(random_goal_pos))]
            path = self.navmesh_interface.find_paths([start], [end], searchSize=[1000.0,1000.0,1000.0])
            if len(path) > 0:
                dist_to_start = np.linalg.norm(start - path[0])
                dist_to_end = np.linalg.norm(end - path[-1])
                if dist_to_start > 3.0 or dist_to_end > 3.0:
                    continue
                print(f'dist_to_start: {dist_to_start}, dist_to_end: {dist_to_end}')
                self.sampled_traj.append(path)
                navmesh_utils.create_curve(path, prim_path=f'/World/Path_{random_goal}{len(self.sampled_traj)}')


        print(f'Sampled {len(self.sampled_traj)} trajectories')
    
    def generate_episodes(self):
        episodes = VLNEpisodes()
        for i, traj in enumerate(self.sampled_traj):
            episodes.data.append({
                'scene_id': self.scene_id,
                'episode_index': i,
                
            })
        return episodes