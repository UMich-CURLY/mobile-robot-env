import argparse

def add_vln_args(parser: argparse.ArgumentParser):
    arg_group = parser.add_argument_group("VLN benchmark", description="Arguments for VLN benchmark.")
    arg_group.add_argument("--episode_type", type=str, default="default", help="Type of the episode.", choices=["default", "grscenes"])
    arg_group.add_argument("--episode_path", type=str, default=None, help="Path to the episode JSON file.")
    arg_group.add_argument("--scene_folder", type=str, default=None, help="Path to the scene USD file.")
    arg_group.add_argument("--navmesh_path", type=str, default=None, help="Path to the navmesh file.")
    arg_group.add_argument("--disable_camera", type=bool, default=False, help="Disable camera.")
    arg_group.add_argument("--num_envs", type=int, default=1, help="How many robots to simulate.")
    arg_group.add_argument("--test_id", type=str, default=None, help="Test specific episode id.")
    arg_group = parser.add_argument_group("Task Generation")
    arg_group.add_argument("--tg_config_path", type=str, default=None, help="Path to the task config file.")