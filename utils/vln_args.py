import argparse

def add_vln_args(parser: argparse.ArgumentParser):
    arg_group = parser.add_argument_group("VLN benchmark", description="Arguments for VLN benchmark.")
    arg_group.add_argument("--episode_type", type=str, default="default", help="Type of the episode.", choices=["default", "grscenes"])
    arg_group.add_argument("--episode_path", type=str, default=None, help="Path to the episode JSON file.")
    arg_group.add_argument("--scene_folder", type=str, default=None, help="Path to the scene USD file.")