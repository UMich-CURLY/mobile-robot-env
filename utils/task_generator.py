from copy import deepcopy
import json
from pathlib import Path
import yaml
import os
import re
import sys
import shutil
import time
import contextlib
import numpy as np
from collections import deque
import cv2
from utils.episode import VLNEpisode, save_episodes
import isaacsim.core.utils.prims as prim_utils
import utils.navmesh_utils as navmesh_utils
from utils.vis import visualize_points, visualize_curve
from utils.path_following_utils import calc_yaw
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from threading import Lock
from queue import Queue

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
        self.data_folder = os.path.join(args.scene_folder, "episode_data")
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(os.path.join(self.data_folder, "status"), exist_ok=True)

    def parse_config(self, task_config):
        self.scene_id_list = []
        self.navmesh_preset_list = [*task_config['navmesh'].keys()]
        self.rule_pattern_list = ["path", "name", "gr", "vc"]
        for scene_type, scene in task_config['scene'].items():
            self.scene_id_list.extend([f'{scene_type}_{x}' for x in scene['episodes'].keys()])
        # set default value
        for scene_type in task_config['scene'].keys():
            scene_type_config = task_config['scene'][scene_type]
            for scene_name in scene_type_config['episodes'].keys():
                scene_config = scene_type_config['episodes'][scene_name]
                if not hasattr(scene_config, "ceiling_height"):
                    scene_config["ceiling_height"] = 1.0

    def parse_scene_id(self, scene_id):
        scene_type = scene_id.split('_')[0]
        scene_name = scene_id[len(scene_type)+1:]
        return scene_type, scene_name

    def get_scene_config(self, scene_id):
        # merge scene_type config (e.g., nv) with scene config (e.g., nv_apartment)
        scene_type, scene_name = self.parse_scene_id(scene_id)
        scene_config = deepcopy(self.task_config['scene'][scene_type])
        scene_config.update(scene_config['episodes'][scene_name])
        del scene_config['episodes']
        scene_config['scene_id'] = f'{scene_type}_{scene_name}'
        scene_config['scene_type'] = scene_type
        return scene_config
    
    def update_config(self, scene_config):
        scene_id = scene_config['scene_id']
        scene_type, scene_name = self.parse_scene_id(scene_id)
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
        self.env.env.cfg.sim.render_interval = 50
        self.scene_config = self.get_scene_config(scene_id)
        self.num_episodes = self.scene_config['episode_number']
        self.rule_pattern = self.scene_config.get('rule_pattern', 'name')
        # requirements
        self.min_path_length = 5.0
        self.max_path_length = 50.0
        self.min_duration = 5.0
        self.timeout = 100.0
        self.total_samples = 0
        # generate task
        total_goal_found = self.parse_scene()
        if total_goal_found == 0:
            print(f'[ERROR]: No goal found')
            return []
        print(f'[TG] Generating {self.num_episodes} episodes')
        self.vln_sim.visualize_waypoints = True
        self.generated_episodes = []
        self.generate_finished = False
        self._generate_episodes()

    def _generate_episodes(self):
        if len(self.generated_episodes) < self.num_episodes or self.total_samples >= 5*self.num_episodes:
            new_episodes = self.sample_episodes(self.num_episodes)
            print(f'[TG] Sampled {len(new_episodes)} episodes')
            self.total_samples += len(new_episodes)
            self.check_episodes(new_episodes)
        else:
            self.generate_finished = True
            self.stop_generation()
            print(f'[TG] All {self.num_episodes} episodes generated!!!')
    
    def stop_generation(self):
        if self.check_status_callback is not None:
            self.vln_sim.remove_callback('step_finished', self.check_status_callback)
            self.check_status_callback = None
            print(f'[TG] Generation stopped')
        self.vln_sim.clear_waypoints()
        self.env.env.cfg.sim.render_interval = 5

    def check_episodes(self, episodes):
        """
        Check episodes asynchronously using callbacks.
        Uses a state machine pattern with episode queue.
        """
        self.episode_queue = deque(episodes)
        self._check_episode()
    
    def _check_episode(self):
        """Start checking the next episode in the queue."""
        if len(self.episode_queue) == 0 or len(self.generated_episodes)>=self.num_episodes:
            print(f'[TG] checking finished')
            self._generate_episodes()
            return
        
        self.current_episode = self.episode_queue.popleft()
        self.current_episode['episode_id'] = len(self.generated_episodes)
        self.current_episode_start_time = time.time()
        print(f'======================')
        print(f'[TG] Checking episode {self.current_episode.episode_id}')
        print(f'[TG] target: {self.current_episode["objnav"]}, path length: {self.current_episode["goals"][0]["path_length"]:.2f}m')

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
            # save data for vln
            if self.vln_sim.obs_index%img_saving_interval == 0:
                # save rgb image
                img_index = self.vln_sim.obs_index//img_saving_interval
                img_path = f"{data_folder}/rgb/{img_index}.png"
                cv2.imwrite(img_path, self.vln_sim.obs["pov_rgb"].cpu().numpy()[0][...,::-1])
                # append pose to txt
                # TODO: change this to waypoints
                with open(pose_path, 'a') as f:
                    f.write(f"{[img_index]+self.vln_sim.obs['pov_pose'].cpu().numpy()[0].tolist()}\n")
            # check if episode is done
            self.env.measure_manager.update_measures()
            measurements = self.env.measure_manager.get_measurements()
            if self.vln_sim.obs_index==0 and measurements["distance_to_goal"] < self.min_path_length:
                print(f'[TG] episode {self.current_episode.episode_id} starting position is too close to goal')
                check_done = True
            elif self.vln_sim.waypoint_follower.arrived_at_goal:
                print("[TG] Measures: ", ", ".join([f"{k}={v:.2f}" for k, v in measurements.items()]))
                # check episode quality by metrics
                if measurements["oracle_success"] != 1.0:
                    print(f'[TG] episode {self.current_episode.episode_id} failed')
                elif measurements["sim_duration"] < self.min_duration:
                    print(f'[TG] episode {self.current_episode.episode_id} completed but duration is too short')
                elif measurements["path_length"] < self.min_path_length:
                    print(f'[TG] episode {self.current_episode.episode_id} completed but path length is too short')
                else:
                    print(f'[TG] episode {self.current_episode.episode_id} completed successfully')
                    success = True
                check_done = True
            elif measurements["sim_duration"] > self.timeout:
                print(f'[TG] episode {self.current_episode.episode_id} timed out')
                check_done = True
            elif "terminations" in self.vln_sim.info:
                print(f'[TG] episode {self.current_episode.episode_id} terminated due to {self.vln_sim.info["terminations"]["termination_reason"]}')
                check_done = True
            # episode is done, check if it is successful
            if check_done:
                self.vln_sim.remove_callback('step_finished', self.check_status_callback)
                self.check_status_callback = None
                if success:
                    self.generated_episodes.append(self.current_episode)
                    episode_path = f"episodes/{self.scene_config['scene_id']}.json"
                    save_episodes(self.generated_episodes, episode_path)
                    print(f'[TG] {len(self.generated_episodes)} episodes saved to {episode_path}')
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
        self.vln_sim.add_callback('step_finished', check_status)

    def parse_scene(self):
        # find target prims
        self.prim_list = [x for x in self.manager_env.scene.stage.Traverse()]
        # [x for x in self.prim_list if str(x.GetPath()).startswith("/World/ground/terrain/Brownstone03/Geometry/Specialty_Equipment/")]
        print(f'Loaded {len(self.prim_list)} prims')
        self.goal_dict = {}
        total_goal_found = 0
        for goal, goal_rule in self.scene_config.get("goal_rules", {}).items():
            if self.rule_pattern == "path":
                # Convert prim path to string and normalize path separators
                goal_prim = []
                for x in self.prim_list:
                    prim_path_str = str(x.GetPrimPath())
                    if re.search(goal_rule, prim_path_str):
                        goal_prim.append(x)
                        print(f"  Matched: {prim_path_str}")
            elif self.rule_pattern == "name":
                goal_prim = [x for x in self.prim_list if re.search(goal_rule, x.GetName())]
            self.goal_dict[goal] = {
                'prim': goal_prim,
            }
            print(f'[TG] Found {len(goal_prim)} {goal}')
            total_goal_found += len(goal_prim)
        if self.rule_pattern == "gr":
            for x in self.prim_list:
                prim_path_str = str(x.GetPrimPath())
                match_result = re.search(r"/([^/]*?)/(model_[^/]*)/Instance$", prim_path_str)
                if match_result:
                    goal = match_result.group(1)
                    object_name = match_result.group(2)
                    self.goal_dict.setdefault(goal, {"prim": []})["prim"].append(x)
                    print(f"  Matched: {goal} {object_name}")
                    total_goal_found += 1
        print(f'[TG] Total goal found: {total_goal_found}')
        return total_goal_found
    
    def check_navmesh(self, scene_id):
        scene_config = self.get_scene_config(scene_id)
        navmesh_interface = self.navmesh_interface
        navmesh_interface.update_settings(self.task_config['navmesh'][scene_config['navmesh_preset']])
        scene_folder = Path(self.args.scene_folder)
        os.makedirs(scene_folder / "navmesh", exist_ok=True)
        navmesh_path = str(scene_folder / f"navmesh/{scene_config['scene_id']}_navmesh.bin")
        if os.path.exists(navmesh_path):
            print(f"[TG] Navmesh found, loading...")
            navmesh_interface.load_navmesh(navmesh_path)
        else:
            print(f"[TG] Navmesh not found, building...")
            selected_paths = ["/World/ground/terrain"]
            start_time = time.time()
            navmesh_interface.setup_navmesh(selected_paths, scene_config.get("navmesh_exclude", []), self.manager_env.scene.stage, scene_type=scene_config.get("scene_type"))
            navmesh_interface.build_navmesh()
            end_time = time.time()
            print(f"[TG] Navmesh build time: {end_time - start_time:.2f} seconds")
            navmesh_interface.save_navmesh(navmesh_path)
    
    def sample_episodes(self, num_episodes):
        self.check_navmesh(self.scene_config['scene_id'])
        navmesh_interface = self.navmesh_interface

        # sample goals uniformly
        unique_goals = []
        sampled_goals = []
        while len(sampled_goals) < num_episodes:
            if len(unique_goals) == 0:
                unique_goals = set(self.goal_dict.keys())
            random_goal = np.random.choice(list(unique_goals))
            sampled_goals.append(random_goal)
            unique_goals.remove(random_goal)
        sampled_goals = sorted(sampled_goals)
        # generate paths from random points to each goal
        generated_episodes = []
        pbar = tqdm(sampled_goals, desc="Generating episodes")
        for goal_name in pbar:
            path_found = False
            retry_count = 0
            while not path_found and retry_count < 100:
                goal_item = self.goal_dict[goal_name]
                # sample random points
                start_pos = navmesh_interface.sample_random_points(1)[0]
                goal_prim_list = goal_item['prim']
                goals = []
                for goal_prim in goal_prim_list:
                    # we use the position calculated with bounding box instead
                    prim_path = goal_prim.GetPrimPath()
                    goal_pos = self.env.get_prim_position(prim_path)
                    path = navmesh_interface.find_paths(start_pos, goal_pos)
                    if len(path) > 0:
                        dist_to_start = np.linalg.norm(start_pos - path[0])
                        dist_to_end = np.linalg.norm(goal_pos - path[-1])
                        obj_radius = self.env.get_prim_radius(prim_path)
                        # skip if the path does not connect to the start or end
                        if dist_to_start > 1.0 or dist_to_end > obj_radius+1.0:
                            continue
                        # skip if the path is too short or too long
                        path_length = np.linalg.norm(path[1:] - path[:-1], axis=1).sum() + dist_to_start + dist_to_end
                        if path_length < self.min_path_length:
                            continue
                        if path_length > self.max_path_length:
                            continue
                        goals.append({
                            'instance': str(prim_path),
                            'type': 'object',
                            'location': goal_pos,
                            'radius': obj_radius,
                            'path_length': path_length,
                            'reference_path': path.tolist(),
                        })
                if len(goals) > 0:
                    path_found = True
                else:
                    retry_count += 1
            if path_found:
                pbar.set_description(f'[TG] Found {len(goals)} paths for goal {goal_name}')
                closest_goal_idx = int(np.argmin([x['path_length'] for x in goals]))
                closest_goal = goals[closest_goal_idx]
                # set initial yaw angle to the next waypoint and add some noise
                yaw_angle = calc_yaw(start_pos[:2], closest_goal['reference_path'][1][:2])
                yaw_angle += np.random.uniform(-np.deg2rad(30.0), np.deg2rad(30.0))
                quat_xyzw = R.from_euler('z', yaw_angle).as_quat().tolist()
                quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
                remove_keys = ["episode_number", "goal_rules", "navmesh_exclude", "rule_pattern", "navmesh_preset"]
                episode = VLNEpisode(
                    data={k: v for k, v in self.scene_config.items() if k not in remove_keys},
                    objnav=goal_name,
                    instruction="",
                    episode_id=len(generated_episodes),
                    goals=goals,
                    start_position=start_pos.tolist(),
                    start_rotation=quat_wxyz, # wxyz
                    closest_goal_idx=closest_goal_idx,
                )
                # visualize_points(random_points, prim_path="/World/RandomPoints", width=0.8)
                # visualize_curve(path, prim_path=f"/World/Path_{goal_prim.GetName()}", width=0.4)
                generated_episodes.append(episode)
            else:
                pbar.set_description(f'[TG] Failed to find paths for goal {goal_name}')
        return generated_episodes
    
    def get_world_bb(self, scene_id):
        scene_type, scene_name = self.parse_scene_id(scene_id)
        navmesh_exclude = self.task_config["scene"][scene_type]["episodes"][scene_name]["navmesh_exclude"]
        for prim_path in navmesh_exclude:
            exclude_prim = self.manager_env.scene.stage.GetPrimAtPath(prim_path)
            prim_utils.set_prim_visibility(exclude_prim, False)
        world_bb = self.env.get_prim_bounding_box("/World/ground/terrain")
        for prim_path in navmesh_exclude:
            exclude_prim = self.manager_env.scene.stage.GetPrimAtPath(prim_path)
            prim_utils.set_prim_visibility(exclude_prim, True)
        return world_bb

    def toggle_ceiling(self, scene_id):
        if not hasattr(self, 'ceiling_data') or self.ceiling_data["scene_id"] != scene_id:
            print(f"[TG] Detecting ceiling")
            prim_list = []
            world_bb = self.get_world_bb(scene_id)
            for prim in self.manager_env.scene.stage.Traverse():
                if str(prim.GetPath()).startswith("/World/ground/terrain"):
                    prim_bb = self.env.get_prim_bounding_box(prim.GetPrimPath())
                    if prim_bb[2] > world_bb[5]-1.0 and prim_bb[2] < 1e3:
                        prim_list.append(prim)
            self.ceiling_data = {
                "prim_list": prim_list,
                "prim_hidden": False,
                "scene_id": scene_id,
            }
        self.ceiling_data["prim_hidden"] = not self.ceiling_data["prim_hidden"]
        for prim in self.ceiling_data["prim_list"]:
            prim_utils.set_prim_visibility(prim, not self.ceiling_data["prim_hidden"])
        print(f"[TG] Ceiling is now {'visible' if self.ceiling_data['prim_hidden'] else 'hidden'}")
    
    def create_bev_camera(self, scene_id):
        # create bev camera
        world_bb = np.array(self.get_world_bb(scene_id))
        world_center = (world_bb[:3] + world_bb[3:]) / 2
        world_size = world_bb[3:] - world_bb[:3]
        camera_pos = world_center + np.array([0, 0, 10.0])
        padding = 1.0
        image_width = min(int((world_size[0]+padding*2) * 100), 3840)
        image_height = int(image_width * world_size[1] / world_size[0])
        self.bev_camera = self.env.create_camera(
            prim_path="/World/bev_camera",
            perspective=False,
            pos=camera_pos,
            quat_opengl=[1, 0, 0, 0],
            horizontal_aperture=(world_size[0]+padding*2)*10,
            clipping_range_min=0.1,
            clipping_range_max=200.0,
            width=image_width,
            height=image_height
        )
        print(f"[TG] Created BEV map camera")

    def create_bev_map(self, scene_id, file_name="bev_map", clip_range="ceiling", ceiling_height=None):
        """ When clip_range is "ceiling", the bev image sees everything under the ceiling.
            When clip_range is "robot", the bev image only see things under robot height + 0.5m,
            which can be used for occupancy.
        """
        # init task queue and lock if not initialized
        if not hasattr(self, 'check_bev_map_queue'):
            self.bev_camera_queue = Queue()
            self.bev_camera_lock = Lock()
            def check_bev_map_queue():
                if self.bev_camera_queue.empty() or self.bev_camera_lock.locked():
                    return
                scene_id, file_name, clip_range, ceiling_height = self.bev_camera_queue.get()
                self._create_bev_map(scene_id, file_name, clip_range, ceiling_height)
            self.vln_sim.add_callback('step_finished', check_bev_map_queue)
            self.check_bev_map_queue = check_bev_map_queue
        # add task to queue
        self.bev_camera_queue.put((scene_id, file_name, clip_range, ceiling_height))

    def _create_bev_map(self, scene_id, file_name, clip_range, ceiling_height):
        self.bev_camera_lock.acquire()
        self.create_bev_camera(scene_id)
        # update camera
        if ceiling_height is None:
            scene_config = self.get_scene_config(scene_id)
            ceiling_height = scene_config.get("ceiling_height", 1.0)
        self.update_bev_camera_clip(scene_id, clip_range, ceiling_height)
        # hide robot
        robot_prim = self.manager_env.scene.stage.GetPrimAtPath("/World/envs/env_0/Robot")
        prim_utils.set_prim_visibility(robot_prim, False)
        def save_bev_map():
            try:
                if self.bev_camera.frame>-1 and self.env.env_step>3:
                    rgb = self.bev_camera.data.output['rgba'].cpu().numpy()[0]
                    depth = self.bev_camera.data.output['distance_to_image_plane'].cpu().numpy()[0]
                    data_folder = f"{self.data_folder}/{scene_id}"
                    os.makedirs(data_folder, exist_ok=True)
                    np.savez(f"{data_folder}/{file_name}.npz", rgb=rgb, depth=depth)
                    cv2.imwrite(f"{data_folder}/{file_name}_rgb.png", rgb[...,[2,1,0,3]])
                    cv2.imwrite(f"{data_folder}/{file_name}_depth.png", depth)
                    print(f"[TG] Saved BEV map to {data_folder}/{file_name}")
                    prim_utils.set_prim_visibility(robot_prim, True)
                    self.vln_sim.remove_callback('step_finished', save_bev_map)
                    self.bev_camera_lock.release()
                    prim_utils.delete_prim(self.bev_camera.cfg.prim_path)
                else:
                    print(f"[TG] BEV camera is not initialized yet, frame: {self.bev_camera.frame}, env_step: {self.env.env_step}")
            except Exception as e:
                print(f"[TG] Error saving BEV map: {e}")
        self.vln_sim.add_callback('step_finished', save_bev_map)

    def update_bev_camera_clip(self, scene_id, clip_range, ceiling_height):
        print(f"[TG] Clip bev camera to {clip_range} with ceiling height {ceiling_height}")
        world_bb = np.array(self.get_world_bb(scene_id))
        world_center = (world_bb[:3] + world_bb[3:]) / 2
        camera_pos = world_center + np.array([0, 0, 10.0])
        if clip_range=="ceiling":
            clipping_range_min = camera_pos[2] - (world_bb[5]-ceiling_height)
        elif clip_range=="robot":
            clipping_range_min = camera_pos[2] - (self.env.get_cam_pose()[0][2]+0.3)
        clipping_range_min = max(0, clipping_range_min)
        clipping_range_max = clipping_range_min+1000
        camera_prim = self.manager_env.scene.stage.GetPrimAtPath("/World/bev_camera")
        clipping_range = camera_prim.GetAttribute('clippingRange').Get()
        clipping_range[0] = clipping_range_min
        clipping_range[1] = clipping_range_max
        camera_prim.GetAttribute('clippingRange').Set(clipping_range)
        # update task config
        scene_type, scene_name = self.parse_scene_id(scene_id)
        self.task_config["scene"][scene_type]["episodes"][scene_name]["ceiling_height"] = ceiling_height

    def save_status(self, scene_id, **kwargs):
        status_file = os.path.join(self.data_folder, "status", f"{scene_id}.json")
        status_data = {}
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                try:
                    status_data = json.load(f)
                except json.JSONDecodeError:
                    pass

        status_data.update(kwargs)
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f)

    def timing_status(self, scene_id, status_name, **kwargs):
        @contextlib.contextmanager
        def context():
            self.save_status(scene_id, status=status_name)
            start_time = time.time()
            try:
                yield
                duration = time.time() - start_time
                kwargs[f"{status_name}"] = "success"
                kwargs[f"{status_name}_time"] = duration
                self.save_status(scene_id, **kwargs)
            except Exception as e:
                duration = time.time() - start_time
                kwargs[f"{status_name}"] = "error"
                kwargs[f"{status_name}_time"] = duration
                kwargs[f"{status_name}_error"] = str(e)
                self.save_status(scene_id, **kwargs)
                raise e
        return context()

    def load_status(self, scene_id):
        status_file = os.path.join(self.data_folder, "status", f"{scene_id}.json")
        if not os.path.exists(status_file):
            return None
        with open(status_file, 'r') as f:
            status_data = json.load(f)
            return status_data.get("status", None)