from copy import deepcopy
import json
from pathlib import Path
import yaml
import os
import re
import sys
import shutil
import time
import numpy as np
from collections import deque
import cv2
from utils.episode import VLNEpisode, save_episodes
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
        # State for episode checking
        self.episode_queue = deque()
        self.current_episode = None
        self.current_episode_start_time = None
        self.check_status_callback = None

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
    
    def bind_vln_sim(self, vln_sim):
        self.vln_sim = vln_sim
        self.env = vln_sim.env
        self.manager_env = vln_sim.manager_env

    def generate_episodes(self, scene_id):
        # load scene config
        self.scene_config = self.get_scene_config(scene_id)
        self.num_episodes = self.scene_config['episode_number']
        self.rule_pattern = self.scene_config.get('rule_pattern', 'name')
        # generate task
        total_goal_found = self.parse_scene()
        if total_goal_found == 0:
            print(f'[ERROR]: No goal found')
            return []
        print(f'[TG] Generating {self.num_episodes} episodes')
        self.vln_sim.visualize_waypoints = True
        self.generated_episodes = []
        self._generate_episodes()

    def _generate_episodes(self):
        if len(self.generated_episodes) < self.num_episodes:
            new_episodes = self.sample_episodes(self.num_episodes)
            print(f'[TG] Sampled {len(new_episodes)} episodes')
            self.check_episodes(new_episodes)
        else:
            print(f'[TG] All episodes generated!!!')
            save_episodes(self.generated_episodes, f"episodes/{self.scene_config['scene_id']}.json")
    
    def stop_generation(self):
        if self.check_status_callback is not None:
            self.vln_sim.remove_callback('on_step_finished', self.check_status_callback)
            self.check_status_callback = None
            print(f'[TG] Generation stopped')
        self.vln_sim.clear_waypoints()

    def check_episodes(self, episodes):
        """
        Check episodes asynchronously using callbacks.
        Uses a state machine pattern with episode queue.
        """
        self.episode_queue = deque(episodes)
        self.filtered_episodes = []
        self._check_episode()
    
    def _check_episode(self):
        """Start checking the next episode in the queue."""
        if len(self.episode_queue) == 0:
            print(f'[TG] All episodes checked, saved {len(self.filtered_episodes)} episodes')
            self.generated_episodes.extend(self.filtered_episodes)
            self._generate_episodes()
            return
        
        self.current_episode = self.episode_queue.popleft()
        self.current_episode['episode_id'] = len(self.filtered_episodes)
        self.current_episode_start_time = time.time()
        print(f'======================')
        print(f'[TG] Checking episode {self.current_episode.episode_id}')

        data_folder = f"{self.args.scene_folder}/episode_data/{self.current_episode.episode_label}"
        pose_path = f"{data_folder}/pose.txt"
        # remove data folder if exists
        if os.path.exists(data_folder):
            shutil.rmtree(data_folder, ignore_errors=True)
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(f"{data_folder}/rgb", exist_ok=True)
        print(f'[TG] Saving data to {data_folder}')

        # Create new callback for this episode
        def check_status():
            if self.vln_sim.current_episode.episode_label != self.current_episode.episode_label:
                return
            check_done = False
            success = False
            img_saving_interval = 1
            timeout = 100.0
            # save data for vln
            if self.vln_sim.obs_index%img_saving_interval == 0:
                # save rgb image
                img_index = self.vln_sim.obs_index//img_saving_interval
                img_path = f"{data_folder}/rgb/{img_index}.png"
                cv2.imwrite(img_path, self.vln_sim.obs["pov_rgb"].cpu().numpy()[0][...,::-1])
                # append pose to txt
                with open(pose_path, 'a') as f:
                    f.write(f"{[img_index]+self.vln_sim.obs['pov_pose'].cpu().numpy()[0].tolist()}\n")
            # check if episode is done
            self.env.measure_manager.update_measures()
            measurements = self.env.measure_manager.get_measurements()
            if self.vln_sim.obs_index==0 and measurements["distance_to_goal"] < 5.0:
                print(f'[TG] episode {self.current_episode.episode_id} starting position is too close to goal')
                check_done = True
            elif self.vln_sim.waypoint_follower.arrived_at_goal:
                print(f'[TG] episode {self.current_episode.episode_id} completed')
                # check episode quality by metrics
                if measurements["oracle_success"] != 1.0:
                    print(f'[TG] episode {self.current_episode.episode_id} failed')
                elif measurements["sim_duration"] < 5.0:
                    print(f'[TG] episode {self.current_episode.episode_id} completed but duration is too short')
                elif measurements["path_length"] < 5.0:
                    print(f'[TG] episode {self.current_episode.episode_id} completed but path length is too short')
                else:
                    print(f'[TG] episode {self.current_episode.episode_id} completed')
                    success = True
                check_done = True
            elif measurements["sim_duration"] > timeout:
                print(f'[TG] episode {self.current_episode.episode_id} timed out')
                check_done = True
            elif "terminations" in self.vln_sim.info:
                print(f'[TG] episode {self.current_episode.episode_id} terminated due to {self.vln_sim.info["terminations"]["termination_reason"]}')
                check_done = True
            # episode is done, check if it is successful
            if check_done:
                self.vln_sim.remove_callback('on_step_finished', self.check_status_callback)
                self.check_status_callback = None
                if success:
                    self.filtered_episodes.append(self.current_episode)
                else:
                    # remove data folder
                    shutil.rmtree(data_folder, ignore_errors=True)
                self._check_episode()
        
        # Reset and start the episode
        print(f'[TG] Reset episode {self.current_episode.episode_id}')
        self.vln_sim.reset(self.current_episode)
        print(f'[TG] Set reference waypoints')
        self.vln_sim.set_ref_waypoints(self.current_episode)
        self.check_status_callback = check_status
        self.vln_sim.add_callback('on_step_finished', check_status)

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
            print(f'[TG] Found {len(goal_prim)} {goal}')
            total_goal_found += len(goal_prim)
        print(f'[TG] Total goal found: {total_goal_found}')
        return total_goal_found
    
    def sample_episodes(self, num_episodes):
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
            print(f"[TG] Navmesh build time: {end_time - start_time:.2f} seconds")
            navmesh_interface.save_navmesh(navmesh_path)

        # Visualize the navmesh
        # navmesh_interface.visualize_navmesh()

        generated_episodes = []
        while len(generated_episodes) < num_episodes:
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
                    print(f'[TG] dist_to_start: {dist_to_start}, dist_to_end: {dist_to_end}')
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
                    episode_id=len(generated_episodes),
                    goals=goals,
                    start_position=start.tolist(),
                    start_rotation=[1.0, 0.0, 0.0, 0.0] # TODO: get random rotation
                )
                # visualize_points(random_points, prim_path="/World/RandomPoints", width=0.8)
                # visualize_curve(path, prim_path=f"/World/Path_{goal_prim.GetName()}", width=0.4)
                generated_episodes.append(episode)

        return generated_episodes
# class TaskGeneratorUI:
#     def __init__(self, manager_env):
#         self.manager_env = manager_env
    
#     def setup_ui(self):
#         manager_env = self.manager_env

        