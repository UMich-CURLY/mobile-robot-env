import json
import yaml
from copy import deepcopy
import os

class VLNEpisode(dict):
    def __init__(self, data=None, **kwargs):
        """ Pass in data or kwargs to update the default episode data """
        super().__init__()
        self.update({
            "scene_id": "test",
            "scene_type": "test",
            "episode_id": 0,
            "path": "test.usd",
            "scene_scale": 1.0,
            "collider": True,
            "align_ground": True,
            "navmesh_preset": "default",
            "objnav": "table",
            "instruction": "Find the table and stop.",
            "closest_goal_idx": 0,
            "goals": [
                {
                    "instance": "table",
                    "type": "object",
                    "location": [-50.37,-19.45, 0.6],
                    "radius": 0.5,
                    "reference_path": [
                        [-52.37,-19.45, 0.6],
                        [-52.37,-3.48, 0.6],
                        [-60.77,-3.48, 0.6]
                    ]
                }
            ],
            "start_position": [0.0, 0.0, 0.6],
            "start_rotation": [1.0, 0.0, 0.0, 0.0],
        })
        if data is not None:
            self.update(deepcopy(data))
        if kwargs:
            self.update(deepcopy(kwargs))
            # print(f'Overwriting episode data with kwargs: {kwargs}')
    def copy(self):
        return deepcopy(self)
    
    @property
    def episode_label(self):
        """ episode label = scene_id + episode_id """
        return f"{self['scene_id']}_{self['episode_id']}"

    @property
    def scene_id(self):
        return self['scene_id']

    @property
    def episode_id(self):
        return self['episode_id']

    @property
    def episode_info(self):
        info = deepcopy(self)
        return json.dumps(self, indent=4)

    def __getitem__(self, key):
        if key == "episode_label":
            return self.episode_label
        if key == "episode_info":
            return self.episode_info
        return super().__getitem__(key)

    @classmethod
    def from_json(self, json_path):
        data = json.load(open(json_path))
        episodes = [VLNEpisode(x) for x in data]
        return episodes

    @classmethod
    def from_json_folder(self, json_folder):
        json_paths = [os.path.join(json_folder, x) for x in os.listdir(json_folder) if x.endswith(".json")]
        episodes = []
        for json_path in json_paths:
            try:
                episodes.extend(self.from_json(json_path))
            except Exception as e:
                print(f"Error loading episode from {json_path}: {e}")
                continue
        episodes.sort(key=lambda x: (x.scene_id, x.episode_id))
        return episodes

def load_episode_set(json_folder):
    #detect every file that ends with _set.txt
    set_files = [x for x in os.listdir(json_folder) if x.endswith("_set.txt")]
    episode_set_list = {}
    for set_file in set_files:
        with open(os.path.join(json_folder, set_file), 'r') as f:
            lines = [x.strip() for x in f.read().splitlines()]
            lines = [x for x in lines if not x.startswith("#")]
            episode_set_list[set_file.replace("_set.txt", "")] = lines
    return episode_set_list

def save_episodes(episodes, json_path):
    json.dump(episodes, open(json_path, 'w'), indent=4)