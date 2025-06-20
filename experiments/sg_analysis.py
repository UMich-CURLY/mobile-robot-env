import json
import yaml
import types
import open_clip
import torch
import torch.multiprocessing as mp
import cv2
import numpy as np
import matplotlib as mpl
from scipy.spatial import KDTree
import peek
import tqdm
from functools import partial

import sys
sys.path.append('..')
sys.path.append('.')
from utils.db_utils import get_df, get_data, connect_db, DB
from utils.plotly_utils import *
from utils.vis import *
from utils.predict_scenegraph import PredictSceneGraph
from utils.imagine_nav_planner import ImagineNavPlanner
from utils.scene_graph_utils import update_region
from experiments.test_scenegraph_offline import evaluate_sg
from constants import *

# Set up multiprocessing method
mp.set_start_method('spawn', force=True)

dump_folder = '/root/Projects/sg-vln/dump/prediction_fix_region_4o_mini_sample101_may14'
output_folder = f'{dump_folder}/objectnav-dino'

# load results
print(f'Loading results from {output_folder}/result.db')
args = types.SimpleNamespace(**json.load(open(f'{dump_folder}/args.json')))
results = get_df(f'{output_folder}/result.db', 'result')
print(f'Loaded {len(results)} results')
print(f'Current success rate: {results.tail(1)["success"].values[0]/len(results):.2%}')
print(f'Current SPL: {results["spl"].mean():.2f}')

# load agent modules
print(f'Loading agent modules')
device = torch.device("cuda")
args = types.SimpleNamespace(**json.load(open(f'{dump_folder}/args.json')))
with open(f'{dump_folder}/{args.exp_config}') as f:
    exp_config = yaml.safe_load(f)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", "laion2b_s32b_b79k"
)
clip_model = clip_model.to(device).half()
clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
clip_model_list = (clip_model, clip_preprocess, clip_tokenizer)
imagine_nav_planner = ImagineNavPlanner(args, exp_config, clip_model_list)
scene_graph = imagine_nav_planner.scene_graph

# access db
import os
import pathlib
with DB(f'{output_folder}/result.db') as con:
    table = con.table('result')
    print(table)
# episode infos
# steps_df = get_df(f'{output_folder}/result.db', 'result', select=['count_steps', 'episode', 'target', 'habitat_success', 'switch_upstair_count', 'switch_downstair_count'])
steps_df = get_df(f'{output_folder}/result.db', 'result', select=['count_steps', 'episode', 'target', 'habitat_success', 'switch_upstair_count', 'switch_downstair_count'])
# print(steps_df.head(30))
# step infos
sample_episode_label = steps_df['episode'].values[0]
with DB(f'{output_folder}/steps/{sample_episode_label}.db') as con:
    table = con.table('step_data')
    print(table)

# evaluate prediction vs caption

from utils.scene_graph_utils import update_region, detect_match
def get_sg_data(episode_label, step=None):
    if step is None:
        filter = lambda x: (x['episode_label']==episode_label) & (x['step']%5==0)
    else:
        filter = lambda x: (x['episode_label']==episode_label)
    try:
        data = get_data(
            f'{output_folder}/steps/{episode_label}.db',
            'step_data',
            filter=filter,
            select=[
                'timestamp',
                'step',
                'episode_label',
                'cate_object',
                'origins_grid',
                'current_grid_pose',
                'camera_position_tensor',
                'global_scene_graph_pickle',
                'gt_scenegraph',
                'predicted_global_scene_graph_pickle',
                'global_bev_rgb_map_tensor',
            ]
        )
    except Exception as e:
        print(e)
        return None
    # data = data[np.argmax([x['timestamp'] for x in data])]
    return data

def check_step(step):
    keys = ['gt_scenegraph', 'predicted_global_scene_graph', 'global_scene_graph']
    for key in keys:
        if key not in step or step[key] is None:
            return False
    return True

args = types.SimpleNamespace(**json.load(open(f'{dump_folder}/args.json')))

count_episode = 0
valid_steps = []
for i in range(len(results)):
    episode_label = results['episode'].iloc[i]
    print(f'Processing episode {episode_label}')
    data = get_sg_data(episode_label)
    if data is None:
        # print(f'No data for episode {episode_label}')
        continue
    count_episode += 1
    # find the last step with valid sg
    for step in data[::-1]:
        if check_step(step):
            valid_steps.append(step)
            break
    if len(valid_steps) >= 5:
        break
print(f'Loaded {count_episode} episodes')
print(f'Loaded {len(valid_steps)} sg')

def process_step(step, clip_model_list, args):
    """Process a single step and return the evaluation scores."""
    grid_size = args.map_resolution
    map_size = args.map_size_cm/grid_size
    origins_grid = step['origins_grid']
    gt_sg = step['gt_scenegraph']
    pred_sg = step['predicted_global_scene_graph']
    obs_sg = step['global_scene_graph']
    camera_position = step['camera_position'][:3, 3]
    floor_avg_heights = [floor['floor_avg_height'] for floor in gt_sg['floors']]
    floor_id = np.argmin(np.abs(np.array(floor_avg_heights) - camera_position[1]))
    
    for room in gt_sg['floors']:
        for region in room['regions']:
            region['caption'] = gt_region_captions_map[region['caption']]
    
    obs_regions = [update_region(region, grid_size, origins_grid) for room in obs_sg['rooms'] for region in room['regions']]
    pred_regions = [update_region(region, grid_size, origins_grid) for room in pred_sg['rooms'] for region in room['regions']]
    gt_regions = [update_region(region, grid_size, origins_grid) for region in gt_sg['floors'][floor_id]['regions']]
    obs_objects = [obj for region in obs_regions for obj in region['objects']]
    pred_objects = [obj for region in pred_regions for obj in region['objects']]
    gt_objects = [obj for region in gt_regions for obj in region['objects']]
    
    obs_matches, obs_scores = evaluate_sg(
        clip_model_list=clip_model_list,
        obs_regions=obs_regions,
        gt_regions=gt_regions,
        obs_objects=obs_objects,
        gt_objects=gt_objects,
        knn_region=3,
        knn_object=5,
        max_object_dist=100.0/grid_size,
    )
    
    pred_matches, pred_scores = evaluate_sg(
        clip_model_list=clip_model_list,
        obs_regions=pred_regions,
        gt_regions=gt_regions,
        obs_objects=pred_objects,
        gt_objects=gt_objects,
        knn_region=3,
        knn_object=5,
        max_object_dist=100.0/grid_size,
    )
    
    return {
        'episode_label': step['episode_label'],
        'step': step['step'],
        'obs_scores': obs_scores,
        'pred_scores': pred_scores
    }

def worker(i, steps, clip_model_list, args_dict, result_queue):
    """Worker process function to handle evaluation tasks."""
    print(f"Worker {i} started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = types.SimpleNamespace(**args_dict)
    
    for step in steps:
        print(f'Worker {i} processing episode {step["episode_label"]} step {step["step"]}')
        result = process_step(step, clip_model_list, args)
        result_queue.put(result)
    
    print(f"Worker {i} finished")

# Main multiprocessing logic
if __name__ == "__main__":
    # Number of processes to use
    num_processes = min(mp.cpu_count(), len(valid_steps))
    num_processes = 1
    print(f"Using {num_processes} processes")
    
    # Create a result queue
    result_queue = mp.Queue()
    
    # Divide steps among processes
    steps_per_process = [[] for _ in range(num_processes)]
    for i, step in enumerate(valid_steps):
        steps_per_process[i % num_processes].append(step)
    
    # Start worker processes
    processes = []
    args_dict = args.__dict__
    
    for i in range(num_processes):
        p = mp.Process(
            target=worker, 
            args=(i, steps_per_process[i], clip_model_list, args_dict, result_queue)
        )
        processes.append(p)
        p.start()
        print(f'Started process {i}')
    
    # Collect results
    all_results = []
    for _ in range(len(valid_steps)):
        all_results.append(result_queue.get())
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Combine results
    obs_scores_combined = {}
    pred_scores_combined = {}
    
    for result in all_results:
        for k, v in result['obs_scores'].items():
            obs_scores_combined.setdefault(k, []).append(v)
        for k, v in result['pred_scores'].items():
            pred_scores_combined.setdefault(k, []).append(v)
    
    # Calculate averages
    obs_avg_scores = {k: np.mean(v) for k, v in obs_scores_combined.items()}
    pred_avg_scores = {k: np.mean(v) for k, v in pred_scores_combined.items()}
    
    # Print results
    print("\nObservation Scores:")
    for k, v in obs_avg_scores.items():
        print(f"{k}: {v:.4f}")
    
    print("\nPrediction Scores:")
    for k, v in pred_avg_scores.items():
        print(f"{k}: {v:.4f}")
    
    peek(obs_avg_scores)
    peek(pred_avg_scores)