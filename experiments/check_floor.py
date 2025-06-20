import os
import gzip
import json
import numpy as np
import re
import json
import random

from habitat import Env, logger, make_dataset
from habitat.config.default import get_config
from habitat.config import read_write

def load_json_gz(file_path):
    """
    Loads and extracts a .json.gz dataset file.
    """
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_goal_positions(json_data):
    goal2position = {}
    for key, value in json_data["goals_by_category"].items():
        object_type = key.split("glb_")[-1]
        goal2position[object_type] = []
        for goal in value:
            position = goal.get("position")
            goal2position[object_type].append(position)

    return goal2position

def at_same_floor(start_position, goal_positions, height_threshold=1):
    upstair_goal = False
    downstair_goal = False
    start_y = start_position[1]
    for goal in goal_positions:
        goal_y = goal[1]
        height_diff = goal_y - start_y
        if height_diff > height_threshold:
            upstair_goal = True
        elif height_diff < -height_threshold:
            downstair_goal = True
        if abs(height_diff) < height_threshold:
            return True, upstair_goal, downstair_goal

    return False, upstair_goal, downstair_goal

def get_scene_id(scene_path):
    # Get the scene_id from the path
    scene_id = scene_path.split("/")[-1]
    scene_id = re.sub(r'\.basis\.glb$', '', scene_id)
    return scene_id

def analyze_dataset(folder_path):
    """
    Analyzes all .json.gz files in a folder, counting total episodes and unique object categories.
    """
    total_episodes = 0
    object_categories = set()
    category_counts = {}
    scene_counts = 0
    at_same_floor_count = 0
    not_same_floor_count = 0
    not_same_floor_scene = {}
    upstair_episode_ids = {}
    downstair_episode_ids = {}
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json.gz"):
            file_path = os.path.join(folder_path, file_name)
            #print(f"Processing {file_name}...")
            scene_counts += 1
            
            json_data = load_json_gz(file_path)
            goal2position = get_goal_positions(json_data)
            episodes = json_data.get("episodes", [])
            total_episodes += len(episodes)
            
            for episode in episodes:
                # print(episode["episode_id"])
                if "object_category" in episode:
                    category = episode["object_category"]
                    object_categories.add(category)
                    if category in category_counts:
                        category_counts[category] += 1
                    else:
                        category_counts[category] = 1
                    start_position = episode["start_position"]
                    
                    same_floor_flag, upstair_goal, downstair_goal = at_same_floor(start_position, goal2position[category])
                    
                    if same_floor_flag:
                        at_same_floor_count += 1
                    else:
                        if file_name not in not_same_floor_scene:
                            not_same_floor_scene[file_name] = 1
                        else:
                            not_same_floor_scene[file_name] += 1
                        not_same_floor_count += 1
                        
                        scene_id = get_scene_id(episode["scene_id"])
                        episode_id = episode["episode_id"]
                        print(episode_id)
                        if upstair_goal:
                            upstair_episode_ids[scene_id].append(f"{scene_id}_{episode_id}")
                        if downstair_goal:
                            downstair_episode_ids[scene_id].append(f"{scene_id}_{episode_id}")
    
    with open("upstair_episode_id.json", "w") as f:
        json.dump(upstair_episode_ids, f)
    with open("downstair_episode_id.json", "w") as f:
        json.dump(downstair_episode_ids, f)
    # upstair_episode_ids = np.save("upstair_episode_id.npy", np.array(upstair_episode_ids))
    # downstair_episode_ids = np.save("downstair_episode_id.npy", np.array(downstair_episode_ids)) 
                    

    print(f"Total scenes: {scene_counts}")
    print(f"Total episodes across all scenes: {total_episodes}")
    print(f"Unique object categories ({len(object_categories)}): {object_categories}")
    print("Episode count per category:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")
    print(at_same_floor_count, "episodes where goal and start position can be at the same floor")
    print(not_same_floor_count, "episodes where goal and start position cannot be at the same floor")
    print("Scene at different floors: ")
    for scene, num in not_same_floor_scene.items():
        print("scene: ", scene, "num episodes: ", num)
        
def analyze_from_habitat_dataset(env, dataset_name="hm3d"):
    total_episodes = 0
    object_categories = set()
    category_counts = {}
    scene_counts = 0
    at_same_floor_count = 0
    not_same_floor_count = 0
    not_same_floor_scene = {}
    upstair_episode_ids = {}
    downstair_episode_ids = {}
    
    # print(env._task._dataset.category_to_task_category_id)
    # raise ValueError("Check the category to task category id")
    for i in range(len(env.episodes)):
        episode = env.episodes[i]
        scene_id = get_scene_id(episode.scene_id)
        episode_id = episode.episode_id
        start_position = episode.start_position
        print(f"------------Processing episode: {scene_id}_{episode_id}--------")
        goal_positions = []
        for goal in episode.goals:
            goal_positions.append(goal.position)
        
        same_floor_flag, upstair_goal, downstair_goal = at_same_floor(start_position, goal_positions, height_threshold=2)
            
        if same_floor_flag:
                at_same_floor_count += 1
        else:
            if scene_id not in not_same_floor_scene:
                not_same_floor_scene[scene_id] = 1
            else:
                not_same_floor_scene[scene_id] += 1
            not_same_floor_count += 1
        
            print(episode_id)
            if upstair_goal:
                if scene_id not in upstair_episode_ids:
                    upstair_episode_ids[scene_id] = []
                upstair_episode_ids[scene_id].append(f"{scene_id}_{episode_id}")
            if downstair_goal:
                if scene_id not in downstair_episode_ids:
                    downstair_episode_ids[scene_id] = []
                downstair_episode_ids[scene_id].append(f"{scene_id}_{episode_id}")
    
    with open(f"{dataset_name}_upstair_episode_id.json", "w") as f:
        json.dump(upstair_episode_ids, f, indent=4)
    with open(f"{dataset_name}_downstair_episode_id.json", "w") as f:
        json.dump(downstair_episode_ids, f, indent=4)
        
    # upstair_episode_ids = np.save("upstair_episode_id.npy", np.array(upstair_episode_ids))
    # downstair_episode_ids = np.save("downstair_episode_id.npy", np.array(downstair_episode_ids)) 

    print(f"Total scenes: {scene_counts}")
    print(f"Total episodes across all scenes: {total_episodes}")
    print(f"Unique object categories ({len(object_categories)}): {object_categories}")
    print("Episode count per category:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")
    print(at_same_floor_count, "episodes where goal and start position can be at the same floor")
    print(not_same_floor_count, "episodes where goal and start position cannot be at the same floor")
    print("Scene at different floors: ")
    for scene, num in not_same_floor_scene.items():
        print("scene: ", scene, "num episodes: ", num)
        
config = get_config("configs/objectnav_procthor-hab_rgbd.yaml")
random.seed(config.habitat.seed)
np.random.seed(config.habitat.seed)
# torch.manual_seed(config.SEED)
# torch.set_grad_enabled(False)
with read_write(config):
    config.habitat.dataset.split = 'val'
    config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
env = Env(config=config)

HM3D_V01_FOLDER = "/workspace/sda1/hm3d/objectnav_hm3d_v1/val/content/"
HM3D_V02_FOLDER = "/workspace/sda1/hm3d/objectnav_hm3d_v2/val/content/"
VLN_GAME_HM3D = "/workspace_sdc/tiamat_ws/VLN-Game/data/datasets/objectgoal_hm3d/val/content/"
# print("### Analyzing VLN-GAME HM3D v0.2 Dataset ###")
analyze_from_habitat_dataset(env)

# config = get_config(config_paths=["configs/objectnav_mp3d.yaml"])
# random.seed(config.SEED)
# np.random.seed(config.SEED)
# # torch.manual_seed(config.SEED)
# # torch.set_grad_enabled(False)
# config.defrost()
# config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
# config.freeze()
# env = Env(config=config)

# print("### Analyzing MP3D Dataset ###")
# analyze_from_habitat_dataset(env, dataset_name="mp3d")
# print("### Analyzing HM3D v0.1 Dataset ###")
# analyze_dataset(HM3D_V01_FOLDER)

# print("\n### Analyzing HM3D v0.2 Dataset ###")
# analyze_dataset(HM3D_V02_FOLDER)

