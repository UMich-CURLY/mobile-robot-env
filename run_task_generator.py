import argparse
import torch
from pathlib import Path
import sys
import subprocess
import threading
import queue
import os
import signal
import time

# start simulation
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="batch task generation")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for task generation.")

import utils.rsl_rl_cli_args as rsl_rl_cli_args  # isort: skip
import utils.vln_args as vln_cli_args

rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)

AppLauncher.add_app_launcher_args(parser)
args = vln_cli_args.parse_args(parser)

# batch task generation
import yaml

task_config = yaml.load(open(args.tg_config_path, 'r'), Loader=yaml.FullLoader)

scene_configs = task_config['scene']
tasks = []
port_start = args.port
for scene_type in ["vc"]: # "grCommercial", "nv", "vc", "grHome"
# for scene_type in scene_configs.keys():
    for scene_name in scene_configs[scene_type]['episodes'].keys():
        # if not scene_name.endswith("_store"):
        #     continue
        tasks.append(f"{scene_type}_{scene_name}")
print(f"Running {len(tasks)} tasks: {tasks}")

# Filter arguments to remove num_workers
cmd_args = []
skip_next = False
for arg in sys.argv[1:]:
    if skip_next:
        skip_next = False
        continue
    if arg == "--num_workers" or arg == "--port":
        skip_next = True
    elif arg.startswith("--num_workers=") or arg.startswith("--port="):
        continue
    else:
        cmd_args.append(arg)

task_queue = queue.Queue()
for task in tasks:
    task_queue.put(task)

# Global variables for signal handling
stop_event = threading.Event()
active_processes = {}
process_lock = threading.Lock()

def signal_handler(signum, frame):
    print(f"\n[MAIN] Signal {signum} received. Shutting down workers...")
    stop_event.set()
    
    with process_lock:
        for w_id, proc in active_processes.items():
            if proc.poll() is None:
                print(f"[MAIN] Killing subprocess for worker {w_id} (pid: {proc.pid})")
                try:
                    proc.kill()
                except Exception as e:
                    print(f"[MAIN] Error killing process {proc.pid}: {e}")
    # We exit with code 1 after initiating shutdown
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def worker(worker_id, port_number):
    log_file_path = f"logs/worker_{worker_id}.log"
    os.makedirs("logs", exist_ok=True)
    
    # We open the file in 'w' mode initially to clear it, then keep it open? 
    # Or just open it once and write to it.
    with open(log_file_path, "w") as log_file:
        while not stop_event.is_set():
            try:
                scene_id = task_queue.get(block=False)
            except queue.Empty:
                break
            
            # Log header
            header = f"\n{'='*20}\nStarting task: {scene_id}\n{'='*20}\n"
            log_file.write(header)
            log_file.flush()
            print(f"[worker {worker_id}] Starting task: {scene_id}")

            cmd = [sys.executable, "isaac_lab_server_spot_task_generation.py"] + cmd_args + ["--test_scene_id", scene_id] + ["--port", str(port_number)]

            print(f"[worker {worker_id}] Running command: {cmd}")
            log_file.write(f"[worker {worker_id}] Running command: {cmd}\n")
            log_file.flush()
            
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # Set CUDA_VISIBLE_DEVICES
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                gpu_id = worker_id % num_gpus
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                print(f"[worker {worker_id}] Using GPU: {gpu_id}")
                log_file.write(f"[worker {worker_id}] Using GPU: {gpu_id}\n")

            process = None
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env
                )
                
                with process_lock:
                    active_processes[worker_id] = process

                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    print(f"[worker {worker_id}] {line}", end='')
                
                process.wait()

                if process.returncode != 0 and process.returncode != -signal.SIGTERM:
                    err_msg = f"\nTask {scene_id} failed with exit code {process.returncode}\n"
                    log_file.write(err_msg)
                    print(f"[worker {worker_id}] {err_msg}")
            
            except Exception as e:
                err_msg = f"\nError running task {scene_id}: {e}\n"
                log_file.write(err_msg)
                print(f"[worker {worker_id}] {err_msg}")
            finally:
                with process_lock:
                    if worker_id in active_processes:
                        del active_processes[worker_id]
                task_queue.task_done()

print(f"Running {len(tasks)} tasks with {args.num_workers} workers.")

threads = []
for i in range(args.num_workers):
    t = threading.Thread(target=worker, args=(i, port_start + i))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
