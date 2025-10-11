import os
import sys
import dotenv

def add_vln_args(parser):
    arg_group = parser.add_argument_group("VLN benchmark", description="Arguments for VLN benchmark.")
    arg_group.add_argument("--episode_folder", type=str, default="episodes", help="Path to the episodes folder.")
    arg_group.add_argument("--episode_id", type=str, default="test_generator_0", help="Test specific episode id.")
    arg_group.add_argument("--scene_folder", type=str, default=None, help="Path to the scene USD file.")
    arg_group.add_argument("--navmesh_path", type=str, default=None, help="Path to the navmesh file.")
    arg_group.add_argument("--disable_camera", type=bool, default=False, help="Disable camera.")
    arg_group.add_argument("--num_envs", type=int, default=1, help="How many robots to simulate.")
    arg_group = parser.add_argument_group("Task Generation")
    arg_group.add_argument("--tg_config_path", type=str, default="episodes/task_config.yaml", help="Path to the task config file.")
    arg_group = parser.add_argument_group("Socket Server", description="Arguments for socket server.")
    arg_group.add_argument("--disable_socket_server", type=bool, default=False, help="Disable socket server.")
    arg_group.add_argument("--socket_server_host", type=str, default="localhost", help="Host for socket server.")
    arg_group.add_argument("--socket_server_port", type=int, default=12300, help="Port for socket server.")

def parse_args(parser):
    arg_list = sys.argv[1:]
    args = parser.parse_args()
    dotenv.load_dotenv()
    for key, value in os.environ.items():
        if key.lower() in args.__dict__:
            arg_list += [f"--{key.lower()}", value]
            print(f"Using {key} from .env: {value}")
    args = parser.parse_args(arg_list)
    return args