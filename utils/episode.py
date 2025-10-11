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
            "navmesh": "default",
            "instruction": "table",
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
    
    @property
    def episode_label(self):
        """ episode label = scene_id + episode_id """
        return f"{self['scene_id']}_{self['episode_id']}"
    
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
            episodes.extend(self.from_json(json_path))
        return episodes
    

def save_episodes(episodes, json_path):
    json.dump(episodes, open(json_path, 'w'), indent=4)