import json

class VLNEpisodes():
    def __init__(self, **kwargs):
        """ Pass in kwargs to update the default single episode data """
        # default data
        self.data = [{
            "episode_id": "",
            "scene_id": "",
            "instruction": "",
            "goal_instances": [],
            "goal_locations": [],
            "start_position": [],
            "start_rotation": [],
            "reference_path": [],
        }]
        self.data[0].update(kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def from_json(self, json_path, format="default"):
        data = json.load(open(json_path))
        if format == "default":
            episodes = VLNEpisodes()
            episodes.data = data
            return episodes
        elif format == "grscenes":
            raise NotImplementedError("GRScenes format is not implemented yet")