from PIL import Image
import numpy as np
import socket
import pickle
import time
from collections import deque
import math

from utils.protocol import *
from argparse import ArgumentParser
from utils.planner import fit_smoothing_spline
HFOV = 72
#preprogrammed waypoints to execute by pressing enter.
# WAYPOINTS = np.array([
# [0,0],[0.5,-0.3],[1,0.2],[1.5,0],[2,-0.2],[2.5,0]
# ])
WAYPOINTS = np.array([
 [ 0.         , 0.        ],
 [ 0.64484999 , 0.00898335],
 [ 0.98256157 , 0.00693384],
 [ 1.1899422  , 0.14765533],
 [ 1.34841325 , 0.50032   ],
 [ 1.60909961 , 0.71981093],
 [ 1.96884275 , 0.81072732],
 [ 2.21724862 , 0.66540045],
 [ 2.48945878 , 0.48726669],
 [ 2.60769912 , 0.23764456],
 [ 2.54893836 ,-0.07430095],
 [ 2.34404152 ,-0.36827063],
 [ 2.11184586 ,-0.39968225],
 [ 1.85050507 ,-0.32115638],
 [ 1.46089797 ,-0.2089194 ],
 [ 1.26725344 ,-0.35803746]])
# WAYPOINTS = np.array([

# [0,0]
# ])


MAGNIFICATION_OPTIONS = [1,2,4,6,8]
magnification_choice = 3

parser = ArgumentParser()
parser.description = "Welcome to the robot client for the SG VLN project"
parser.add_argument("--host",type=str,default='localhost',help="the host name of the remote robot server")
args = parser.parse_args()
BORDER = 30
ROBOT_VIS_CENTER = np.array([BORDER*2+640+320,240+BORDER])
screen_width, screen_height = 640*2+BORDER*3, 720+20


from utils.socket_client import request_sensor_data,send_action_message,request_planner_state

run = True
data = None

lastx = 0 
lasty = 0

points = deque(np.zeros((0,2),dtype = float),maxlen=3000)
vx,vy,omg = 0,0,0

init_T = None
curr_T = None

from scipy.spatial.transform import Rotation
from copy import deepcopy

waypointmsg = WaypointMessage()
translations = None

start_time = time.time()
frame_cnt = 0

while run:
    if translations is None:
        send_action_message(VelMessage(vx,vy,omg), host=args.host)

    try:
        start_ts = time.time()
        data = request_sensor_data(args.host)
        end_ts = time.time()
        print(f"[TOTAL] Time: {end_ts - start_ts:.3f}s")
        print('-'*20)
        # start_ts = time.time()
        # my_dict = request_planner_state(args.host)
        # end_ts = time.time()
        # print(f"[request_planner_state] Time: {end_ts - start_ts:.3f}s")
        # print('-'*20)

    except socket.timeout:
            print(f"Socket timeout during operation with {args.host}")
    except socket.error as e:
        print(f"Socket error: {e}")
    except pickle.UnpicklingError as e:
        print(f"Pickle error: {e}. Received data might be corrupt or not a pickle.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", exc_info=True)


    frame_cnt += 1
    print(f"[INFO] fps: {frame_cnt/(time.time() - start_time):.1f}")