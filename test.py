import time
from utils.socket_client import request_sensor_data
from argparse import ArgumentParser


parser = ArgumentParser()
parser.description = "Welcome to the robot client for the SG VLN project"
parser.add_argument("--host",type=str,default='localhost',help="the host name of the remote robot server")
args = parser.parse_args()

while True:
    start_ts = time.time()
    data = request_sensor_data(args.host)
    end_ts = time.time()
    print(f"[Request] Time: {end_ts - start_ts:.3f}s")