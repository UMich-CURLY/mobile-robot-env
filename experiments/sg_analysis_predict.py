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
import pickle
from utils.scene_graph_utils import update_region, detect_match
import os
import pathlib


def get_sg_data(episode_label, step=None):
    if step is None:
        filter = lambda x: (x['episode_label']==episode_label) & (x['step']%10==0)
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
                # fbe
                'traversible_map_tensor',
                'occupancy_map_tensor',
                'frontier_candidate_list',
            ]
        )
    except Exception as e:
        print(e)
        return None
    # data = data[np.argmax([x['timestamp'] for x in data])]
    return data

def check_step(step, llm):
    keys = ['gt_scenegraph', 'global_scene_graph', 'predicted_global_scene_graph']
    for key in keys:
        if key not in step or step[key] is None:
            return False
    return True

def sample_steps(step_list, n=5):
    step_list = [x for x in step_list if x['step']>=30]
    if len(step_list) <= n:
        return step_list
    else:
        # Calculate interval to get n evenly spaced samples
        interval = max(1, (len(step_list)-1) // (n-1))
        # Return evenly spaced samples using the interval
        return step_list[::interval][:n]

def process_step(step, clip_model_list, args):
    """Process a single step and return the evaluation scores."""
    grid_size = args.map_resolution
    map_size = args.map_size_cm/grid_size
    origins_grid = step['origins_grid']
    camera_position = step['camera_position'][:3, 3]
    target = step['cate_object']
    obs_sg = step['global_scene_graph']
    obs_regions = [update_region(region, grid_size, origins_grid) for room in obs_sg['rooms'] for region in room['regions']]
    obs_objects = [obj for region in obs_regions for obj in region['objects']]
    gt_sg = step['gt_scenegraph']
    floor_avg_heights = [floor['floor_avg_height'] for floor in gt_sg['floors']]
    floor_id = np.argmin(np.abs(np.array(floor_avg_heights) - camera_position[1]))
    for room in gt_sg['floors']:
        for region in room['regions']:
            region['caption'] = gt_region_captions_map[region['caption']]
    gt_regions = [update_region(region, grid_size, origins_grid) for region in gt_sg['floors'][floor_id]['regions']]
    gt_objects = [obj for region in gt_regions for obj in region['objects']]

    if args.scene_graph_prediction_llm != 'none':
        output_folder = f'{args.dump_folder}/{args.scene_graph_prediction_llm}'
        os.makedirs(output_folder, exist_ok=True)
        # check cache
        cache_file = f'{output_folder}/{step["episode_label"]}_{step["step"]}.pkl'
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            pred_sg = data['pred_sg']
        else:
            # fbe
            frontier_candidate_list = step['frontier_candidate_list']
            current_grid_pose = step['current_grid_pose']
            traversible_map = step['traversible_map']
            occupancy_map = step['occupancy_map']
            global_bev_rgb_map = step['global_bev_rgb_map']
            with open(f'{args.dump_folder}/{args.exp_config}') as f:
                exp_config = yaml.safe_load(f)
            
            imagine_nav_planner = ImagineNavPlanner(args, exp_config, clip_model_list)
            scene_graph = imagine_nav_planner.scene_graph
            imagine_nav_planner.origins_grid = origins_grid
                

            scene_graph.obs_sg = obs_sg
            scene_graph.grid_size = grid_size
            scene_graph.scene_graph = copy.deepcopy(obs_sg)
            imagine_nav_planner.set_obj_goal(target)
            imagine_nav_planner.set_step(step["step"], step['episode_label'])
            success = False
            while not success:
                try:
                    scores, best_frontier_id, exploration_scores = imagine_nav_planner.fbe(
                        frontier_candidate_list,
                        current_grid_pose,
                        traversible_map,
                        occupancy_map,
                        global_bev_rgb_map,
                        gt_sg['floors'][floor_id],
                        force_prediction=True
                    )
                    success = True
                except Exception as e:
                    print(f'Error: {e}')
                    continue
            pred_sg = imagine_nav_planner.predicted_global_scene_graph
        if pred_sg is None or pred_sg == {}:
            print("Error: predicted scene graph is None or empty")
            print('frontier_candidate_list: ', len(frontier_candidate_list))
            print('with_prediction: ', len(imagine_nav_planner.with_prediction))
            raise Exception('Error: predicted scene graph is None or empty')
        pred_regions = [update_region(region, grid_size, origins_grid) for room in pred_sg['rooms'] for region in room['regions']]
        pred_objects = [obj for region in pred_regions for obj in region['objects']]
        if not os.path.exists(cache_file):
            data = {
                # 'prompt': scene_graph._llm_prompt,
                # 'response': scene_graph._response,
                'pred_sg': imagine_nav_planner.predicted_global_scene_graph
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
    else:
        pred_sg = step['predicted_global_scene_graph']
        pred_regions = [update_region(region, grid_size, origins_grid) for room in pred_sg['rooms'] for region in room['regions']]
        pred_objects = [obj for region in pred_regions for obj in region['objects']]

    _, obs_scores3 = evaluate_sg(
        clip_model_list=clip_model_list,
        obs_regions=obs_regions,
        gt_regions=gt_regions,
        obs_objects=obs_objects,
        gt_objects=gt_objects,
        knn_region=3,
        knn_object=5,
        max_object_dist=100.0/grid_size,
        topk=3
    )

    _, obs_scores1 = evaluate_sg(
        clip_model_list=clip_model_list,
        obs_regions=obs_regions,
        gt_regions=gt_regions,
        obs_objects=obs_objects,
        gt_objects=gt_objects,
        knn_region=3,
        knn_object=5,
        max_object_dist=100.0/grid_size,
        topk=1
    )

    obs_scores = {
        'recall1': obs_scores1['region_recall_relaxed'],
        'recall3': obs_scores3['region_recall_relaxed'],
        'precision1': obs_scores1['region_precision_relaxed'],
        'precision3': obs_scores3['region_precision_relaxed'],
    }
    
    _, pred_scores3 = evaluate_sg(
        clip_model_list=clip_model_list,
        obs_regions=pred_regions,
        gt_regions=gt_regions,
        obs_objects=pred_objects,
        gt_objects=gt_objects,
        knn_region=3,
        knn_object=5,
        max_object_dist=100.0/grid_size,
        topk=3
    )
    
    _, pred_scores1 = evaluate_sg(
        clip_model_list=clip_model_list,
        obs_regions=pred_regions,
        gt_regions=gt_regions,
        obs_objects=pred_objects,
        gt_objects=gt_objects,
        knn_region=3,
        knn_object=5,
        max_object_dist=100.0/grid_size,
        topk=1
    )

    pred_scores = {
        'recall1': pred_scores1['region_recall_relaxed'],
        'recall3': pred_scores3['region_recall_relaxed'],
        'precision1': pred_scores1['region_precision_relaxed'],
        'precision3': pred_scores3['region_precision_relaxed'],
    }


    return {
        'episode_label': step['episode_label'],
        'step': step['step'],
        'obs_scores': obs_scores,
        'pred_scores': pred_scores
    }

def worker(i, steps, args_dict, result_queue):
    """Worker process function to handle evaluation tasks."""
    args = types.SimpleNamespace(**args_dict)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(device).half()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    clip_model_list = (clip_model, clip_preprocess, clip_tokenizer)

    print(f"Worker {i} started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.scene_graph_prediction_llm == 'gpt-4o':
        os.environ["GPT_API_KEY"] = "sk-Sg58hfwiUGtMak37De998909A76c44E88f7dF36730Dd29B4"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(i//8)
    elif args.scene_graph_prediction_llm == 'llama3.2-vision':
        port = args.ollama_port_start + i
        os.environ["OLLAMA_HOST"] = f'http://127.0.0.1:{port}'

    for step in steps:
        print(f'Worker {i} processing episode {step["episode_label"]} step {step["step"]}')
        result = process_step(step, clip_model_list, args)
        if result is None:
            continue
        result_queue.put(result)
    
    print(f"Worker {i} finished")

# Main multiprocessing logic
if __name__ == "__main__":

    # Set up multiprocessing method
    mp.set_start_method('spawn', force=True)

    dump_folder = '/root/Projects/sg-vln/dump/prediction_fix_region_4o_mini_sample101_may14'
    output_folder = f'{dump_folder}/objectnav-dino'

    # input_folder = '/root/Projects/sg-vln/dump/prediction_fix_region_4o_mini_sample101_may14/objectnav-dino'

    # load results
    print(f'Loading results from {output_folder}/result.db')
    args = types.SimpleNamespace(**json.load(open(f'{dump_folder}/args.json')))
    results = get_df(f'{output_folder}/result.db', 'result')
    # results = get_df(f'{input_folder}/result.db', 'result')
    print(f'Loaded {len(results)} results')
    print(f'Current success rate: {results.tail(1)["success"].values[0]/len(results):.2%}')
    print(f'Current SPL: {results["spl"].mean():.2f}')

    # load agent modules
    print(f'Loading agent modules')
    device = torch.device("cuda")
    args = types.SimpleNamespace(**json.load(open(f'{dump_folder}/args.json')))

    # access db
    # with DB(f'{output_folder}/result.db') as con:
    #     table = con.table('result')
    #     print(table)
    # episode infos
    # steps_df = get_df(f'{output_folder}/result.db', 'result', select=['count_steps', 'episode', 'target', 'habitat_success', 'switch_upstair_count', 'switch_downstair_count'])
    # steps_df = get_df(f'{output_folder}/result.db', 'result', select=['count_steps', 'episode', 'target', 'habitat_success', 'switch_upstair_count', 'switch_downstair_count'])
    # print(steps_df.head(30))
    # step infos
    # sample_episode_label = steps_df['episode'].values[0]
    # with DB(f'{output_folder}/steps/{sample_episode_label}.db') as con:
    #     table = con.table('step_data')
    #     print(table)

    # evaluate prediction vs caption


    args = types.SimpleNamespace(**json.load(open(f'{dump_folder}/args.json')))
    args.scene_graph_prediction_llm = sys.argv[1]
    args.ollama_port_start  = 13000
    print(f'Using {args.scene_graph_prediction_llm} as scene graph prediction llm')

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
        samples = [x for x in data if check_step(x, args.scene_graph_prediction_llm)]
        samples = sample_steps(samples, n=5)
        print(f'Episode {episode_label} has {len(samples)} valid steps, {[x["step"] for x in samples]}')
        valid_steps.extend(samples)
        if len(valid_steps) >= int(sys.argv[3]):
            break
    print(f'Loaded {count_episode} episodes')
    print(f'Loaded {len(valid_steps)} sg')

    # Number of processes to use
    # num_processes = min(mp.cpu_count(), len(valid_steps))
    num_processes = int(sys.argv[2])
    print(f"Using {num_processes} processes")
    
    # Create a result queue
    result_queue = mp.Queue()
    
    # Divide steps among processes
    steps_per_process = [[] for _ in range(num_processes)]
    for i, step in enumerate(valid_steps):
        steps_per_process[i % num_processes].append(step)


    from multiprocessing import Process, Queue, Event
    import subprocess

    def read_output(process, status):
        while True:
            output = process.stdout.readline()
            if b"Listening" in output:
                print("[OLLAMA] ollama serve started successfully")
                status.put(1)
            elif b"Error" in output:
                print("[OLLAMA] ", output.decode().strip())
                status.put(0)

    def run_ollama_serve(port, gpu_id):
        os.environ["OLLAMA_HOST"] = f'http://127.0.0.1:{port}'
        os_env = os.environ.copy()
        os_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            bufsize=-1,
            env=os_env,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
        )
        # status = Queue()
        # Process(target=read_output, args=(ollama_process, status), daemon=True).start()
        # try:
        #     if not status.get(True, timeout=3):
        #         raise Exception("[OLLAMA] Error starting ollama serve")
        # except Exception as e:
        #     print("[OLLAMA] Timeout waiting for ollama serve to start")
        #     raise
        return ollama_process
    
    # Start worker processes
    processes = []
    args_dict = args.__dict__
    args_dict['dump_folder'] = dump_folder
    for i in range(num_processes):
        if args.scene_graph_prediction_llm == 'llama3.2-vision':
            port = args.ollama_port_start + i
            run_ollama_serve(port, i%8)
        p = mp.Process(
            target=worker, 
            args=(i, steps_per_process[i], args_dict, result_queue)
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
    episode_labels = []
    
    for result in all_results:
        for k, v in result['obs_scores'].items():
            obs_scores_combined.setdefault(k, []).append(v)
        for k, v in result['pred_scores'].items():
            pred_scores_combined.setdefault(k, []).append(v)
        episode_labels.append(result['episode_label']+'_'+str(result['step']))
    
    # Calculate averages
    obs_avg_scores = {k: np.mean(v).item() for k, v in obs_scores_combined.items()}
    pred_avg_scores = {k: np.mean(v).item() for k, v in pred_scores_combined.items()}
    
    # Print results
    print("\nObservation Scores:")
    for k, v in obs_avg_scores.items():
        print(f"{k}: {v:.4f}")
    
    print("\nPrediction Scores:")
    for k, v in pred_avg_scores.items():
        print(f"{k}: {v:.4f}")
    
    peek(obs_avg_scores)
    peek(pred_avg_scores)

    import yaml
    with open(f'{output_folder}/sg_scores_{args.scene_graph_prediction_llm}_{sys.argv[3]}.yaml', 'w') as f:
        yaml.dump({
            'obs_avg_scores': obs_avg_scores,
            'pred_avg_scores': pred_avg_scores,
            'count': len(obs_scores_combined['region_recall_relaxed']),
        }, f)

    # save pickle
    with open(f'{output_folder}/sg_scores_{args.scene_graph_prediction_llm}_{sys.argv[3]}.pkl', 'wb') as f:
        pickle.dump({
            'obs_avg_scores': obs_avg_scores,
            'pred_avg_scores': pred_avg_scores,
            'count': len(obs_scores_combined['region_recall_relaxed']),
            'episode_labels': episode_labels,
            'obs_scores_combined': obs_scores_combined,
            'pred_scores_combined': pred_scores_combined
        }, f)
