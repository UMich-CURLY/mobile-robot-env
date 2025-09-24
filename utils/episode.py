import json
import yaml
import copy

class VLNEpisodes():
    def __init__(self, data=None, **kwargs):
        """ Pass in kwargs to update the default single episode data """
        self.data = data if data is not None else [self.get_default_data()]
        if kwargs:
            self.data[0].update(kwargs)
            print(f'Overwriting episode data with kwargs: {kwargs}')
    def get_default_data(self):
        return {
            "scene_id": "test",
            "episode_id": "test_0",
            "scene_path": "test.usd",
            "scene_type": "test",
            "instruction": "table",
            "scene_scale": 1.0,
            "collider": True,
            "align_ground": True,
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
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": [1.0, 0.0, 0.0, 0.0],
        }
    
    @property
    def episode_ids(self):
        """ episode id = scene_id + episode_id """
        return [episode["episode_id"] for episode in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index]
        elif index in self.episode_ids:
            return self.data[self.episode_ids.index(index)]
        else:
            raise IndexError(f"Episode id {index} not found")
    

    @classmethod
    def from_json(self, json_path, format="default"):
        data = json.load(open(json_path))
        if format == "default":
            episodes = VLNEpisodes()
            episodes.data = data
            return episodes
        elif format == "grscenes":
            raise NotImplementedError("GRScenes format is not implemented yet")
    
    def save_to_json(self, json_path):
        json.dump(self.data, open(json_path, 'w'), indent=4)