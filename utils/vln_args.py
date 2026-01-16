import os
import sys
from dotenv import dotenv_values

def add_vln_args(parser):
    arg_group = parser.add_argument_group("VLN benchmark", description="Arguments for VLN benchmark.")
    arg_group.add_argument("--episode_folder", type=str, default="episodes", help="Path to the episodes folder.")
    arg_group.add_argument("--episode_label", type=str, default="none", help="Test specific episode label. Format: {scene_id}_{episode_id}), e.g., nv_apartment_0.")
    arg_group.add_argument("--test_scene_id", type=str, default="none", help="Scene id used for task generation.")
    arg_group.add_argument("--scene_folder", type=str, default=None, help="Path to the scene USD file.")
    arg_group.add_argument("--navmesh_path", type=str, default=None, help="Path to the navmesh file.")
    arg_group.add_argument("--disable_camera", type=bool, default=False, help="Disable camera.")
    arg_group.add_argument("--num_envs", type=int, default=1, help="How many robots to simulate.")
    arg_group = parser.add_argument_group("Debug Options")
    arg_group.add_argument("--disable_termination", type=bool, default=False, help="Disable episodetermination.")
    arg_group = parser.add_argument_group("Task Generation")
    arg_group.add_argument("--tg_config_path", type=str, default="episodes/task_config.yaml", help="Path to the task config file.")
    arg_group = parser.add_argument_group("Server", description="Arguments for socket server.")
    arg_group.add_argument("--disable_socket_server", type=bool, default=False, help="Disable socket server.")
    arg_group.add_argument("--host", type=str, default="localhost", help="Host for socket server.")
    arg_group.add_argument("--port", type=int, default=12300, help="Port for socket server.")
    arg_group.add_argument("--foxglove_port", type=int, default=0, help="Port for foxglove server.")
    arg_group.add_argument("--task_type", type=str, default="objnav", help="")

def parse_args(parser):
    arg_list = sys.argv[1:]
    args = parser.parse_args()
    dot_args = dotenv_values(".env")
    for key, value in dot_args.items():
        if key.lower() in args.__dict__ and "--"+key.lower() not in arg_list:
            arg_list += [f"--{key.lower()}", value]
            print(f"Using {key} from .env: {dot_args[key]}")
    arg_list += ["--kit_args", "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"]
    print(f"Using kit args: {arg_list}")
    args = parser.parse_args(arg_list)
    return args