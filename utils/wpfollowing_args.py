import argparse

def add_wpfollowing_args(parser: argparse.ArgumentParser):
    parser.add_argument("--use_plan", action="store_true", help="Enable waypoint following")
    parser.add_argument("--waypoint_stride", type=int, default=1, help="Stride for waypoints")
    parser.add_argument("--base_height", type=float, default=0.6, help="Base robot height offset")
