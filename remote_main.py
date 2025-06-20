# from sensor_msgs.msg import CompressedImage, Image
# from nav_msgs.msg import Odometry
# from cv_bridge import CvBridge

import numpy as np
import struct
# from tf2_ros import TransformBroadcaster
# from geometry_msgs.msg import TransformStamped
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import copy
from utils.protocol import *
import argparse
import os
import sys
sys.path.append(".")
sys.stdout.reconfigure(line_buffering=True)
import time

from typing import Dict
import numpy as np
import torch
from copy import deepcopy
import shutil
import imageio

from arguments import get_args

from agents.objnav_agent_remote import ObjectNav_Agent
import cv2
from collections import defaultdict
import jsonpickle
#  from habitat.sims.habitat_simulator.actions import (
#     HabitatSimActions,
#     HabitatSimV1ActionSpaceConfiguration,
# )

# from utils.perception_utils.gt_perception import get_gt_goal_positions, GTPerception
# from utils.scene_graph_utils import get_gt_scenegraph
from utils.vis import pad_frames_to_same_size
from utils.shortest_path_follower import ShortestPathFollowerCompat
# from utils.db_utils import connect_db, generate_schema, insert_data, analyze_data_size
from constants import category_to_id, episode_labels_table, categories_21

import socket
import pickle
import time
import struct

import threading
import queue # For thread-safe communication
from multiprocessing import Event
reset_flag = threading.Event() # To signal a reset
def input_thread_func(command_queue):
    """Thread function to listen for keyboard input."""
    # print("Input thread started. Type 'reset' or 'quit'.")
    while True:
        try:
            # This will block the input_thread, but not the main_thread
            command = input() # Or sys.stdin.readline().strip()
            if command:
                command_queue.put(command.lower())
            if command.lower() == 'quit':
                break
        except EOFError: # Happens if stdin is closed (e.g. detached)
            print("Input thread: EOF detected, exiting.")
            command_queue.put('quit') # Ensure main loop knows
            break
        except Exception as e:
            print(f"Input thread error: {e}")
            # Potentially put an error or quit command in queue
            break
    print("Input thread finished.")


# --- Configuration ---
SERVER_HOST = '35.3.201.75' # IP address of the sensor machine (use 'localhost' if running on the same machine)
# SERVER_HOST = '35.7.34.48' #Avery's Desktop
# SERVER_HOST = 'localhost'
SERVER_HOST = '35.7.32.217'

SERVER_PORT = 12345
REQUEST_MESSAGE = b"GET_SENSOR_DATA"
REQUEST_INTERVAL_SEC = 1.0 # 1 Hz

def recv_all(sock, n):
    """Helper function to receive n bytes from a socket."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None # Connection closed
        data.extend(packet)
    return data
def decompress_payload(compressed_payload_dict):
    """
    Decompresses 'rgb_image' and 'depth_image' in the payload dictionary
    if they were PNG-compressed.

    Args:
        compressed_payload_dict (dict): The dictionary received from the server,
                                        potentially with compressed image data as bytes.

    Returns:
        dict: A new dictionary with image bytes replaced by NumPy arrays.
    """
    decompressed_dict = compressed_payload_dict.copy()

    # Decompress RGB Image
    if isinstance(decompressed_dict.get('rgb_image'), bytes):
        encoded_bytes = decompressed_dict['rgb_image']
        # Convert bytes back to NumPy array for imdecode
        np_arr = np.frombuffer(encoded_bytes, dtype=np.uint8)
        rgb_image_np = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if rgb_image_np is not None:
            decompressed_dict['rgb_image'] = rgb_image_np
            # Optionally, verify against stored shape/dtype if they were sent
            # stored_shape = decompressed_dict.get('rgb_image_shape')
            # stored_dtype = decompressed_dict.get('rgb_image_dtype')
            # if stored_shape and rgb_image_np.shape != stored_shape:
            #     print(f"Warning: Decompressed RGB shape {rgb_image_np.shape} differs from original {stored_shape}")
            # if stored_dtype and str(rgb_image_np.dtype) != stored_dtype:
            #     print(f"Warning: Decompressed RGB dtype {rgb_image_np.dtype} differs from original {stored_dtype}")
        else:
            print("Warning: RGB image PNG decoding failed.")
            decompressed_dict['rgb_image'] = None # Or handle error
        # Clean up compression-specific keys
        decompressed_dict.pop('rgb_image_compressed_format', None)
        decompressed_dict.pop('rgb_image_shape', None)
        decompressed_dict.pop('rgb_image_dtype', None)


    # Decompress Depth Image
    if decompressed_dict.get('depth_image_compressed_format') == 'png' and \
       isinstance(decompressed_dict.get('depth_image'), bytes):
        encoded_bytes = decompressed_dict['depth_image']
        np_arr = np.frombuffer(encoded_bytes, dtype=np.uint8)
        # cv2.IMREAD_UNCHANGED is crucial for 16-bit depth images
        depth_image_np = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if depth_image_np is not None:
            decompressed_dict['depth_image'] = depth_image_np
        else:
            print("Warning: Depth image PNG decoding failed.")
            decompressed_dict['depth_image'] = None
        # Clean up compression-specific keys
        decompressed_dict.pop('depth_image_compressed_format', None)
        decompressed_dict.pop('depth_image_shape', None)
        decompressed_dict.pop('depth_image_dtype', None)

    return decompressed_dict


def save_step_data(step_data_queue, args, stop_event=None):
    commit_interval = 10
    db_state_list = {}
    db_folder = f'{args.dump_location}/{args.exp_name}/steps/'
    os.makedirs(os.path.dirname(db_folder), exist_ok=True)
    get_time = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    process_label = args.process_label
    while not (stop_event.is_set() and step_data_queue.empty()):
        try:
            # get data from the queue
            if step_data_queue.empty():
                yield
                continue
            episode_steps_dict = {}
            while step_data_queue.qsize() > 0:
                step_data = step_data_queue.get()
                episode_label = step_data['episode_label']
                episode_steps_dict.setdefault(episode_label, []).append(step_data)
            # save data to db for each episode
            for episode_label, step_list in episode_steps_dict.items():
                # create a new db
                if episode_label not in db_state_list:
                    print(f"[db] [{process_label}] [{episode_label}] create {db_folder}{episode_label}_tmp.db")
                    if os.path.exists(f'{db_folder}{episode_label}_tmp.db'):
                        os.remove(f'{db_folder}{episode_label}_tmp.db')
                    con = connect_db(f'{db_folder}{episode_label}_tmp.db')
                    db_state_list[episode_label] = con
                    con.raw_sql("SET checkpoint_threshold = '100G';")
                    con.raw_sql("BEGIN TRANSACTION;")
                    # create a table according to the first data
                    schema = generate_schema(step_list)
                    con.create_table('step_data', schema=schema, overwrite=True)
                    # print(f"[db] [{process_label}] Created table step_data, schema:\n{schema}")
                    # analyze_data_size(step_list[0], prefix="")
                con = db_state_list[episode_label]
                step_data = step_list[0]
                step_id = step_data['step']
                action = step_data['action']
                # commit the transaction if required
                if step_id>0 and step_id%commit_interval == 0:
                    print(f"[db] [{process_label}] [{episode_label}] committing transaction, time={get_time()}")
                    sys.stdout.flush()
                    start_time = time.time()
                    con.raw_sql("COMMIT;")
                    print(f"[db] [{process_label}] [{episode_label}] committed in {time.time() - start_time:.2f}s")
                    con.raw_sql("BEGIN TRANSACTION;")
                # insert data
                start_time = time.time()
                insert_data(con, 'step_data', step_list)
            print(f"[db] [{process_label}] [{episode_label}] Inserted {len(step_list)} step data in {time.time() - start_time:.2f}s, step={step_id}, time={get_time()}")
            # commit, checkpoint and close the db
            if step_id == 499 or action == 0:
                print(f"[db] [{process_label}] [{episode_label}] episode end, committing and checkpointing, time={get_time()}")
                con.raw_sql("COMMIT;")
                start_time = time.time()
                con.raw_sql("CHECKPOINT;")
                print(f"[db] [{process_label}] [{episode_label}] checkpointed in {time.time() - start_time:.2f}s")
                con.disconnect()
                # move tmp db to final db
                db_path = f'{db_folder}{episode_label}.db'
                if os.path.exists(db_path):
                    os.remove(db_path)
                print(f"[db] [{process_label}] [{episode_label}] moving {db_folder}{episode_label}_tmp.db to {db_path}")
                os.rename(f'{db_folder}{episode_label}_tmp.db', db_path)
        except Exception as e:
            print(f"[db] [{process_label}] Error encountered: {e}")
            import traceback
            traceback.print_exc()
            if args.debug:
                raise
        yield
    sys.stdout.flush()

def main():

    position = np.zeros(3)
    args = get_args()
    torch.cuda.set_device(args.gpu_id)
    args.process_label = 0
    args.real_world=True
    args.hfov = 54.7
    
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    if args.backend=="ros":
        print("using ros to communicate actions")
        ros_client = roslibpy.Ros(host=SERVER_HOST, port=9090)
        ros_client.run()
        talker = roslibpy.Topic(ros_client, '/cmd_vel', 'geometry_msgs/Twist')
        print("Agent Socket Client Started")

    # logging
    args.exp_name = args.exp_name + "-" + args.detector
    log_dir = "{}/{}/logs/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    video_save_dir = '{}/{}/episodes_video'.format(
                args.dump_location, args.exp_name)



    agent = ObjectNav_Agent(args)
    sim_info = {'env':'reality','scene_name':'UofM','episode_id':'test'}
    
    id = -1
    global loop
    print("VLN initialization complete!!!")

    command_queue = queue.Queue()
    listener_thread = threading.Thread(target=input_thread_func, args=(command_queue,), daemon=True)
    listener_thread.start()
    
    step_data_queue = queue.Queue()
    stop_event = Event()
    
    # step_saver = save_step_data(step_data_queue, args, stop_event)
    ask_for_episode = False
    while True:
        if reset_flag.is_set():
            print("Program acknowledged reset. Restarting main loop...")
            reset_flag.clear()
        id+=1
        
        loop = True
        # label = input("plz enter episode label >")
        # target = input("plz enter target object >")
        if ask_for_episode:
            print("please enter episode name >",end='',flush=True)
            label = command_queue.get()
        else:
            label = "initial episode"
            ask_for_episode = True
        print("please enter target object >",end='',flush=True)
        target = command_queue.get()

        agent.reset(label, id, id)
        payload = None # Initialize payload to None for this cycle
        
        print("starting episode. type 'reset' and hit [ENTER] to start a new episode")
        while True:
         
            try:
                command = command_queue.get_nowait()
                print(f"Main loop received command: {command}")
                if command == 'reset' or reset_flag.is_set():
                    print("Resetting main loop...")
                    reset_flag.set() # Signal the reset

                    if args.save_video:
                        video_save_path = '{}/{}/episodes_video/eps_{}_vis.mp4'.format(args.dump_location, args.exp_name, label)
                        same_size_frames = pad_frames_to_same_size(agent.vis_frames)
                        imageio.mimsave(video_save_path, same_size_frames, fps=2)
                        print(f"[{args.process_label}] Video saved to {video_save_path}")
                    break # Exit current main_loop to restart it
                elif command == 'quit':
                    print("Quitting main loop...")
                    return False # Signal program to terminate
            except queue.Empty:
                pass # No command yet, continue with main loop
            try:
                print(f"\n[{time.strftime('%H:%M:%S')}] Attempting to connect to {SERVER_HOST}:{SERVER_PORT}...")
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(5.0) # Timeout for connection and operations
                client_socket.connect((SERVER_HOST, SERVER_PORT))
                print("Connected to server.")

                # 1. Send request
                client_socket.sendall(REQUEST_MESSAGE)
                print(f"Sent request: {REQUEST_MESSAGE.decode()}")

                # 2. Receive the length of the pickled data (8 bytes, unsigned long long)
                raw_msglen = recv_all(client_socket, 8)
                if not raw_msglen:
                    print("Connection closed by server before sending data length.")
                    continue 
                
                msglen = struct.unpack('>Q', raw_msglen)[0]
                print(f"Expecting pickled data of length: {msglen} bytes")

                # 3. Receive the pickled data
                pickled_payload = recv_all(client_socket, msglen)
                if not pickled_payload:
                    print("Connection closed by server before sending full payload.")
                    continue

                # 4. Deserialize
                payload = pickle.loads(pickled_payload)
                payload = decompress_payload(payload)
                print("Successfully received and unpickled data.")
                
                # --- Process Data ---
                if payload and payload.get("success", False):
                    rgb_image = payload.get("rgb_image")
                    depth_image = payload.get("depth_image")
                    pose = payload.get("pose")['pose']

                    position[0] = pose['position']['x']
                    position[1] = pose['position']['y']
                    position[2] = pose['position']['z']

                    rotation = np.quaternion(
                        pose['orientation']['w'],
                        pose['orientation']['x'],
                        pose['orientation']['y'],
                        pose['orientation']['z']
                    )
                    server_timestamp_ns = payload.get("timestamp_server_ns")
                    
                    latency_ms = (time.time_ns() - server_timestamp_ns) / 1_000_000 if server_timestamp_ns else -1
                    print(f"  Server timestamp (ns): {server_timestamp_ns}, Approx E2E Latency: {latency_ms:.2f} ms")

                    obs = {'rgb':rgb_image,'depth':depth_image.astype(float)/1000 ,'objectgoal':target}
                    sim_info['position'] = position
                    sim_info['rotation'] = rotation
                    action,pathx,pathy = agent.act(obs, sim_info, step_data_queue, None, None)

                  

                    # actions = [{
                    #         'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    #         'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    #     },{
                    #         'linear': {'x': 0.35, 'y': 0.0, 'z': 0.0},
                    #         'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    #     },{
                    #         'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    #         'angular': {'x': 0.0, 'y': 0.45, 'z': 0.0}
                    #     },{
                    #         'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    #         'angular': {'x': 0.0, 'y': -0.45, 'z': 0.0}
                    #     },{
                    #         'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    #         'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    #     },{
                    #         'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    #         'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    #     }]

                    actions = [{
                            'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                            'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                        },{
                            'linear': {'x': 0.35, 'y': 0.0, 'z': 0.0},
                            'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                        },{
                            'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                            'angular': {'x': 0.0, 'y': 0.5, 'z': 0.0}
                        },{
                            'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                            'angular': {'x': 0.0, 'y': -0.5, 'z': 0.0}
                        },{
                            'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                            'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                        },{
                            'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                            'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                        }]
                    if args.backend=="ros":
                        talker.publish(actions[action])
                    elif args.backend=="isaac":
                        if(pathx is not None):
                            msg = WaypointMessage()
                            msg.x=pathx.tolist()
                            msg.y=pathy.tolist()
                        else:
                            msg = VelMessage(0,0,0)
                        
                        if action == 0:
                            msg = VelMessage(0,0,0)
                            print("agent decided to stop, resetting")
                            reset_flag.set()
                        
                        
                        payload = msg.type+" "+jsonpickle.encode(msg)
                        # messages = ["VEL 0 0 0 ","VEL 0.5 0 0","VEL 0 0 0.3","VEL 0 0 -0.3"]
                        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client_socket.settimeout(5.0) # Timeout for connection and operations

                        client_socket.connect((SERVER_HOST, SERVER_PORT))
                        print(payload)
                        client_socket.sendall(payload.encode()) #for isaac sim

                    else:
                        print("WARNING: invalid action backend. action is only printed.")

                    print("action: %s" % action)

                elif payload: # success == False or no success key
                    print(f"  Server responded with an issue: {payload.get('message', 'Unknown error')}")
                else: # Should not happen if recv_all worked
                    print("  Received empty or invalid payload from server.")

            except socket.timeout:
                print(f"Socket timeout during operation with {SERVER_HOST}:{SERVER_PORT}")
            except socket.error as e:
                print(f"Socket error: {e}")
            except pickle.UnpicklingError as e:
                print(f"Pickle error: {e}. Received data might be corrupt or not a pickle.")
            finally:
                if 'client_socket' in locals():
                    client_socket.close()
        stop_event.set()
        # step_saver.next()

if __name__ == '__main__':
    main()