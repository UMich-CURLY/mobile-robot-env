# find every json file in episodes folder
import os
import json

for file in os.listdir("episodes"):
    if file.endswith(".json"):
        with open(os.path.join("episodes", file), "r") as f:
            data = json.load(f)
            for episode in data:
                episode["episode_id"] = data.index(episode)
            with open(os.path.join("episodes", file), "w") as f:
                json.dump(data, f, indent=4)