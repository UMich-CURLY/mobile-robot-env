from copy import deepcopy
import json
from pathlib import Path
import yaml
import os
import re
from scipy.spatial import KDTree
import sys
import time
import numpy as np
from utils.episode import VLNEpisode

# UI
import utils.navmesh_utils as navmesh_utils
from utils.vis import visualize_points, visualize_curve

class TaskGenerator:
    """
    Task Generator for InNOut benchmark.

    @note scene_config = items under scene_type + items under scene_id
    @note scene_id = scene_type + scene_name
    @note episode_id = scene_type + scene_name + episode_name
    @note sim is in charge of communication with client and keyboard
    @note env is in charge of internal environment control
    @note during testing, ui updates task_config, task_config updates env
    """
    def __init__(self, args):
        self.args = args
        self.task_config = yaml.load(open(args.tg_config_path, 'r'), Loader=yaml.FullLoader)
        self.parse_config(self.task_config)
        # setup navmesh tools
        self.navmesh_interface = navmesh_utils.NavmeshInterface(up_axis='Z')

    def parse_config(self, task_config):
        self.scene_id_list = []
        self.navmesh_preset_list = [*task_config['navmesh'].keys()]
        self.rule_pattern_list = ["path", "name"]
        for scene_type, scene in task_config['scene'].items():
            self.scene_id_list.extend([f'{scene_type}_{x}' for x in scene['episodes'].keys()])

    def _parse_scene_id(self, scene_id):
        scene_type = scene_id.split('_')[0]
        scene_name = scene_id[len(scene_type)+1:]
        return scene_type, scene_name

    def get_scene_config(self, scene_id):
        scene_type, scene_name = self._parse_scene_id(scene_id)
        scene_config = deepcopy(self.task_config['scene'][scene_type])
        scene_config.update(scene_config['episodes'][scene_name])
        del scene_config['episodes']
        scene_config['scene_id'] = f'{scene_type}_{scene_name}'
        scene_config['scene_type'] = scene_type
        return scene_config
    
    def update_config(self, scene_config):
        scene_id = scene_config['scene_id']
        scene_type, scene_name = self._parse_scene_id(scene_id)
        scene_type_config = self.task_config['scene'][scene_type]
        scene_name_config = scene_type_config['episodes'][scene_name]
        # update by key
        for key, value in scene_config.items():
            if key in scene_type_config:
                scene_type_config[key] = scene_config[key]
            if key in scene_name_config:
                scene_name_config[key] = scene_config[key]
        
    def save_config(self):
        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True
        with open(self.args.tg_config_path, "w") as f:
            yaml.dump(self.task_config, f, Dumper=NoAliasDumper)

    def generate_episodes(self, env, scene_id):
        # bind objects
        self.env = env
        self.manager_env = env.manager_env
        # load scene config
        self.scene_config = self.get_scene_config(scene_id)
        self.num_episodes = self.scene_config['episode_number']
        self.rule_pattern = self.scene_config.get('rule_pattern', 'name')
        # generate task
        total_goal_found = self.parse_scene()
        if total_goal_found == 0:
            print(f'[ERROR]: No goal found')
            return []
        self.sample_episodes()
        return self.generated_episodes

    def parse_scene(self):
        # find target prims
        self.prim_list = [x for x in self.manager_env.scene.stage.Traverse()]
        # [x for x in self.prim_list if str(x.GetPath()).startswith("/World/ground/terrain/Brownstone03/Geometry/Specialty_Equipment/")]
        print(f'Loaded {len(self.prim_list)} prims')
        self.goal_dict = {}
        total_goal_found = 0
        for goal, goal_rule in self.scene_config['goal_rules'].items():
            if self.rule_pattern == "path":
                # Convert prim path to string and normalize path separators
                goal_prim = []
                for x in self.prim_list:
                    prim_path_str = str(x.GetPrimPath()).replace('\\', '/')
                    if re.search(goal_rule, prim_path_str):
                        goal_prim.append(x)
                        print(f"  Matched: {prim_path_str}")
            else:
                goal_prim = [x for x in self.prim_list if re.search(goal_rule, x.GetName())]
            self.goal_dict[goal] = {
                'prim': goal_prim,
            }
            print(f'[INFO]: Found {len(goal_prim)} {goal}')
            total_goal_found += len(goal_prim)
        print(f'[INFO]: Total goal found: {total_goal_found}')
        # self.goal_kd_tree = KDTree(self.goal_positions.values())
        return total_goal_found
    
    def sample_episodes(self):
        navmesh_interface = self.navmesh_interface
        scene_folder = Path(self.args.scene_folder)
        navmesh_path = str(scene_folder / f"navmesh/{self.scene_config['scene_id']}_navmesh.bin")
        if os.path.exists(navmesh_path):
            navmesh_interface.load_navmesh(navmesh_path)
        else:
            selected_paths = ["/World/ground/terrain"]
            start_time = time.time()
            navmesh_interface.setup_navmesh(selected_paths, self.scene_config.get("navmesh_exclude", []), self.manager_env.scene.stage, scene_type=self.scene_config.get("scene_type"))
            navmesh_interface.build_navmesh()
            end_time = time.time()
            print(f"[INFO]: Navmesh build time: {end_time - start_time:.2f} seconds")
            navmesh_interface.save_navmesh(navmesh_path)

        # Visualize the navmesh
        navmesh_interface.visualize_navmesh()

        self.generated_episodes = []
        while len(self.generated_episodes) < self.num_episodes:
            # sample random points
            random_points = navmesh_interface.sample_random_points(1)
            start = random_points[0]
            random_goal = np.random.choice(list(self.goal_dict.keys()))
            goal_prim_list = self.goal_dict[random_goal]['prim']
            goals = []
            for goal_prim in goal_prim_list:
                # we use the position calculated with bounding box instead
                prim_path = goal_prim.GetPrimPath()
                goal_pos = self.env.get_prim_position(prim_path)
                path = navmesh_interface.find_paths(start, goal_pos)
                if len(path) > 0:
                    dist_to_start = np.linalg.norm(start - path[0])
                    dist_to_end = np.linalg.norm(goal_pos - path[-1])
                    if dist_to_start > 1.0 or dist_to_end > 1.0:
                        continue
                    print(f'[INFO]: dist_to_start: {dist_to_start}, dist_to_end: {dist_to_end}')
                    goals.append({
                        'instance': str(prim_path),
                        'type': 'object',
                        'location': goal_pos,
                        'radius': self.env.get_prim_radius(prim_path),
                        'reference_path': path.tolist()
                    })
            if len(goals) > 0:
                episode = VLNEpisode(
                    data=self.scene_config,
                    instruction=random_goal,
                    episode_id=len(self.generated_episodes),
                    goals=goals,
                    start_position=start.tolist(),
                    start_rotation=[1.0, 0.0, 0.0, 0.0] # TODO: get random rotation
                )
                visualize_points(random_points, prim_path="/World/RandomPoints", width=0.8)
                visualize_curve(path, prim_path=f"/World/Path_{goal_prim.GetName()}", width=0.4)
                self.generated_episodes.append(episode)

        print(f'[INFO]: Generated {len(self.generated_episodes)} episodes')

# class TaskGeneratorUI:
#     def __init__(self, manager_env):
#         self.manager_env = manager_env
    
#     def setup_ui(self):
#         manager_env = self.manager_env

        