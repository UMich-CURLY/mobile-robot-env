import os
from utils.predict_scenegraph import PredictSceneGraph
from utils.scene_graph_utils import get_gt_scenegraph
from copy import deepcopy
import json
import open_clip
import numpy as np
import copy
import pickle
import time
from utils.vln_logger import vln_logger
from utils.db_utils import get_df, get_data, connect_db, DB
from utils.utils_llm import construct_query, query_llm, format_graph, format_response, load_json, text2value, save_json
from utils.scene_graph_utils import update_region, detect_match, plot_matches, plot_region, detect_regions
from utils.plotly_utils import save_image
from utils.vis import remove_image_border
from utils.vis_scenegraph import visualize_BEV
from utils.db_utils import generate_schema, insert_data
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import pickle
from utils.raycast import seperate_map, get_visible_unknown
from sklearn.cluster import DBSCAN
import plotly.express as px


def parse_args():
    parser = argparse.ArgumentParser(description="SceneGraph Processing with CLIP + VLM")

    # Basic experiment arguments
    # parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment')
    # parser.add_argument('--llm_name', type=str, default='gpt-4', help='Name of the LLM used')
    # parser.add_argument('--vlm_name', type=str, default='clip', help='Name of the vision-language model')
    # parser.add_argument('--mode', type=str, default='prediction', choices=['prediction', 'generation'], help='Operation mode')

    # # Paths
    # parser.add_argument('--scene_data_path', type=str, required=True, help='Path to scene data (pickle/json/csv)')
    # parser.add_argument('--log_path', type=str, default='metrics.log', help='Path to log output')
    # parser.add_argument('--avg_log_path', type=str, default='metrics_avg.log', help='Path to average metrics output')

    # Optional runtime settings
    parser.add_argument('--split_l', type=int, default=0, help='Start Episode inference')
    parser.add_argument('--split_r', type=int, default=100, help='End Episode inference')

    return parser.parse_args()


def get_db_files(folder_path):
    """Return a list of files ending with .db in the given folder."""
    db_files = [f for f in os.listdir(folder_path) if f.endswith('.db') and 'scene' not in f]
    return db_files


def save_to_pickle(data, filename):
    """Save data to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

def load_from_pickle(filename):
    """Load data from a pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    # print(f"Data loaded from {filename}")
    return data

class MetricLogger:
    def __init__(self, log_path=None, avg_log_path=None):
        self.log_path = log_path
        self.avg_log_path = avg_log_path
        self.metric_sums = {}
        self.metric_counts = {}

    def log(self, metrics: dict):
        # Update totals and counts
        for key, value in metrics.items():
            if key not in ['esisode_label', 'step']:
                self.metric_sums[key] = self.metric_sums.get(key, 0.0) + value
                self.metric_counts[key] = self.metric_counts.get(key, 0) + 1

        # Format and print the current metrics
        line = ", ".join([
            f"{k}: {v:.2f}" if isinstance(v, (float, int)) else f"{k}: {v}"
            for k, v in metrics.items()
        ])

        print("Cur ->", line)

        # Save to metric log file
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(line + '\n')

    def get_averages(self):
        return {
            k: self.metric_sums[k] / self.metric_counts[k]
            for k in self.metric_sums if k not in ['esisode_label', 'step']
        }

    def write_averages(self):
        averages = self.get_averages()
        avg_line = ", ".join([f"{k}: {v:.2f}" for k, v in averages.items()])
        print("Avg ->", avg_line)

        if self.avg_log_path:
            with open(self.avg_log_path, 'a') as f:
                f.write(avg_line + '\n')


class OfflineAgentData():
    def __init__(self, db_path, split_l=None, split_r=None):
        # self.db_path = f'{db_path}/step_data.db'
        self.db_path = db_path
        # self.data = self.load_data()
        # self.episodes_list = sorted(set(self.data['episode_label']))
        self.episodes_list = self.load_data()
        if split_l is not None and split_r is not None:
            self.episodes_list = self.episodes_list[split_l:split_r]
        self._idx = 0  # Iterator index

    def load_data(self):
        # List all the keys in db
        # with DB(self.db_path) as con:
        #     table = con.table('step_data')
        #     print(table)
        start = time.time()
        # steps_df = get_df(self.db_path, 'step_data', select=['step', 'timestamp', 'scene_id', 'episode_label', 'episode_id'])
        db_files = get_db_files(self.db_path)
        print("Data loaded in {:.2f} seconds".format(time.time() - start))
        # return steps_df
        return db_files
    
    def __len__(self):
        return len(self.episodes_list)

    def __iter__(self):
        self._idx = 0  # Reset index on new iteration
        return self

    def __next__(self):
        if self._idx >= len(self.episodes_list):
            raise StopIteration

        episode_id = self.episodes_list[self._idx]
        # self.episode_id = episode_id
        # self.episode_data = self.data[self.data['episode_label'] == episode_id].reset_index(drop=True)
        self.cur_db_path = os.path.join(self.db_path, episode_id)
        self.episode_data = get_df(self.cur_db_path, 'step_data', select=['step', 'timestamp', 'scene_id', 'episode_label', 'episode_id'])
        self.episode_id = episode_id.split('.')[0]
        self._idx += 1
        return self.episode_data
    
    def load_step_data(self, step, episode_label):
        start = time.time()
        steps_df = get_data(self.cur_db_path, 'step_data',
            filter=lambda x: (x['step'] == step) & (x['episode_label'] == episode_label),
            select=[
                'step',
                'timestamp',
                'scene_id',
                'cate_object',
                'origins_grid',
                'episode_label', 
                'episode_id',
                'current_grid_pose',
                'frontier_candidate_list',
                'camera_position_tensor',
                'global_scene_graph',
                'gt_scenegraph',
                'global_bev_rgb_map_tensor',
                'gradient_map_tensor',
                'occupancy_map_tensor',
        ])
        try:
            step_data = steps_df[np.argmax([x['timestamp'] for x in steps_df])]
            print("Step data loaded in {:.2f} seconds".format(time.time() - start))
            # step_data = steps_df.iloc[0]
            return step_data
        except:
            return None


def set_vln_logger(step_data, experiment_name):
    vln_logger.set_step(step_data['step'])
    vln_logger.set_experiment_name(experiment_name)
    vln_logger.set_scene(step_data['scene_id'])
    vln_logger.set_episode(step_data['episode_id'])
    vln_logger.set_dump_dir(f"./dump/{experiment_name}/")
    return



def process_gtsg_as_obssg(step_data, gt_scenegraph):
    camera_position = step_data['camera_position'][:3, 3]
    floor_avg_heights = [floor['floor_avg_height'] for floor in gt_scenegraph['floors']]
    floor_id = np.argmin(np.abs(np.array(floor_avg_heights) - camera_position[1]))
    gt_regions = gt_scenegraph['floors'][floor_id]
    gtsg_as_obssg = {'rooms': [gt_regions]}
    return gtsg_as_obssg

def regroup_scenegraph(step_data, scenegraph, knn_object=30, distance_threshold=50, wall_threshold=10, min_num=3, save_path=None):
    regions, objects = get_regions_objects(step_data, scenegraph)
    origins_grid = step_data['origins_grid']
    step = step_data['step']
    grid_size = 5
    gradient_map = step_data['gradient_map']
    wall_map = (gradient_map>1.2).astype(np.uint8)
    objects_copy = copy.deepcopy(objects)
    vis_fig = px.imshow(wall_map[::-1])
    new_regions = detect_regions(objects_copy, wall_map, knn_object=knn_object, distance_threshold=distance_threshold, wall_threshold=wall_threshold, min_num=min_num, vis_fig=vis_fig)
    vis_fig.update_xaxes(visible=False)
    vis_fig.update_yaxes(visible=False)
    vis_fig.write_image(f"{save_path}/{step}_regroup.png", scale=4)
    
    for region in new_regions:
        update_region(region, grid_size, origins_grid)
    scenegraph_regouped = {
            "rooms": [
                {
                    "caption": "",
                    "regions": new_regions
                }
            ]
        }
    return scenegraph_regouped

def get_regions_objects(step_data, scenegraph):
    # print('processing gt regions')
    origins_grid = step_data['origins_grid']
    grid_size = 5
    
    if 'floors' in scenegraph:
        camera_position = step_data['camera_position'][:3, 3]
        floor_avg_heights = [floor['floor_avg_height'] for floor in scenegraph['floors']]
        floor_id = np.argmin(np.abs(np.array(floor_avg_heights) - camera_position[1]))
        objects = []
        regions = scenegraph['floors'][floor_id]['regions']
        for region in regions:
            update_region(region, grid_size, origins_grid)
            objects.extend(region.get('objects'))
    else:
        # print('processing obs regions')
        regions = []
        objects = []
        for room in scenegraph.get('rooms'):
            if len(room.get('regions')) == 0:
                continue
            for region in room.get('regions'):
                update_region(region, grid_size, origins_grid)
                regions.append(region)
                objects.extend(region.get('objects'))
    return regions, objects

def generate_evalformat(step_data, observed_scenegraph, gt_scenegraph):
    obs_regions, obs_objects = get_regions_objects(step_data, observed_scenegraph)
    gt_regions, gt_objects = get_regions_objects(step_data, gt_scenegraph)
    return obs_regions, gt_regions, obs_objects, gt_objects

def vis_matches(save_path, matches, global_bev_rgb_map, obs_regions, gt_regions):
    def remove_boarder(image):
        remove_image_border(image, black_bg=False)
        remove_image_border(image, black_bg=True)
        return image
    obs_vis = global_bev_rgb_map[::-1].copy()
    gt_vis = global_bev_rgb_map[::-1].copy()
    map_size = global_bev_rgb_map.shape[0]
    obs_vis = plot_region(obs_vis, obs_regions, map_size)
    gt_vis = plot_region(gt_vis, gt_regions, map_size)
    save_image(f'{save_path}_region_recall_relaxed.png', remove_boarder(plot_matches(obs_vis, gt_vis, matches['region_recall_relaxed'])))
    save_image(f'{save_path}_region_precision_relaxed.png', remove_boarder(plot_matches(obs_vis, gt_vis, matches['region_precision_relaxed'], reversed=True)))
    save_image(f'{save_path}_region_recall_strict.png', remove_boarder(plot_matches(obs_vis, gt_vis, matches['region_recall_strict'])))
    save_image(f'{save_path}_region_precision_strict.png', remove_boarder(plot_matches(obs_vis, gt_vis, matches['region_precision_strict'], reversed=True)))
    save_image(f'{save_path}_object_recall.png', remove_boarder(plot_matches(obs_vis, gt_vis, matches['object_recall'])))
    save_image(f'{save_path}_object_precision.png', remove_boarder(plot_matches(obs_vis, gt_vis, matches['object_precision'], reversed=True)))

def eval_scenegraph(clip_model_list, obs_regions, gt_regions, obs_objects, gt_objects, knn_region=3, knn_object=5, grid_size=5, topk=2):
    max_object_dist = 100.0/grid_size #cm

    region_recall_relaxed_matches, region_recall_relaxed = detect_match(
        clip_model_list=clip_model_list,
        keys=obs_regions,
        queries=gt_regions,
        knn=knn_region,
        overlap_relaxed=True,
        corr_score=None,
        topk=topk
    )
    region_recall_strict_matches, region_recall_strict = detect_match(
        clip_model_list=clip_model_list,
        keys=obs_regions,
        queries=gt_regions,
        knn=knn_region,
        overlap_relaxed=False,
        corr_score=None,
        topk=topk
    )
    region_precision_relaxed_matches, region_precision_relaxed = detect_match(
        clip_model_list=clip_model_list,
        keys=gt_regions,
        queries=obs_regions,
        knn=knn_region,
        overlap_relaxed=True,
        corr_score=None,
        topk=topk
    )
    region_precision_strict_matches, region_precision_strict = detect_match(
        clip_model_list=clip_model_list,
        keys=gt_regions,
        queries=obs_regions,
        knn=knn_region,
        overlap_relaxed=False,
        corr_score=None,
        topk=topk
    )
    object_recall_matches, object_recall = detect_match(
        clip_model_list=clip_model_list,
        keys=obs_objects,
        queries=gt_objects,
        knn=knn_object,
        distance=max_object_dist,
        overlap_relaxed=None,
        corr_score=0.9,
    )
    object_precision_matches, object_precision = detect_match(
        clip_model_list=clip_model_list,
        keys=gt_objects,
        queries=obs_objects,
        knn=knn_object,
        distance=max_object_dist,
        overlap_relaxed=None,
        corr_score=0.9,
    )

    matches = {
        "region_recall_strict": region_recall_strict_matches,
        "region_recall_relaxed": region_recall_relaxed_matches,
        "region_precision_strict": region_precision_strict_matches,
        "region_precision_relaxed": region_precision_relaxed_matches,
        "object_recall": object_recall_matches,
        "object_precision": object_precision_matches,
    }

    # Log one step of metrics
    metric = {
        "region_recall_strict": region_recall_strict,
        "region_recall_relaxed": region_recall_relaxed,
        "region_precision_strict": region_precision_strict,
        "region_precision_relaxed": region_precision_relaxed,
        "object_recall": object_recall,
        "object_precision": object_precision,
    }
    return matches, metric



def evaluate_sg(
    clip_model_list,
    obs_regions,
    gt_regions,
    obs_objects,
    gt_objects,
    knn_region = 3,
    knn_object = 5,
    max_object_dist = 100.0/5.0,
    corr_score=0.1,
    topk=3
):
    ### another version of evaluate_sg
    matches = {}
    scores = {}
    metric_name = 'region_recall_relaxed'
    matches[metric_name], scores[metric_name] = detect_match(
        clip_model_list=clip_model_list,
        keys=obs_regions,
        queries=gt_regions,
        knn=knn_region,
        overlap_relaxed=True,
        corr_score=corr_score,
        topk=topk,
    )
    # metric_name = 'region_recall_strict'
    # matches[metric_name], scores[metric_name] = detect_match(
    #     clip_model_list=clip_model_list,
    #     keys=obs_regions,
    #     queries=gt_regions,
    #     knn=knn_region,
    #     overlap_relaxed=False,
    #     corr_score=corr_score,
    # )
    metric_name = 'region_precision_relaxed'
    matches[metric_name], scores[metric_name] = detect_match(
        clip_model_list=clip_model_list,
        keys=gt_regions,
        queries=obs_regions,
        knn=knn_region,
        overlap_relaxed=True,
        corr_score=corr_score,
        topk=topk,
    )
    # metric_name = 'region_precision_strict'
    # matches[metric_name], scores[metric_name] = detect_match(
    #     clip_model_list=clip_model_list,
    #     keys=gt_regions,
    #     queries=obs_regions,
    #     knn=knn_region,
    #     overlap_relaxed=False,
    #     corr_score=corr_score,
    # )
    # metric_name = 'object_recall'
    # matches[metric_name], scores[metric_name] = detect_match(
    #     clip_model_list=clip_model_list,
    #     keys=obs_objects,
    #     queries=gt_objects,
    #     knn=knn_object,
    #     distance=max_object_dist,
    #     overlap_relaxed=None,
    #     corr_score=0.9,
    # )
    # metric_name = 'object_precision'
    # matches[metric_name], scores[metric_name] = detect_match(
    #     clip_model_list=clip_model_list,
    #     keys=gt_objects,
    #     queries=obs_objects,
    #     knn=knn_object,
    #     distance=max_object_dist,
    #     overlap_relaxed=None,
    #     corr_score=0.9,
    # )
    return matches, scores

def get_mask_center(mask):
    """
    Compute the center (centroid) of a binary mask.
    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s or 0s and 255s)
        
    Returns:
        (y, x): Tuple of center coordinates
    """
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    # Get coordinates of non-zero pixels
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None  # Empty mask

    center_y = np.mean(ys)
    center_x = np.mean(xs)
    return [center_y, center_x]

def get_unknown_cluster_centers(unknown_centers, eps=20, min_samples=1):
    """
    Group unknown centers into nearby regions and return the center of each cluster.

    Args:
        unknown_centers (np.ndarray): Array of shape (N, 2) with (y, x) coordinates.
        eps (float): Distance threshold for clustering.
        min_samples (int): Minimum number of points to form a cluster.

    Returns:
        list of tuple: List of (y, x) coordinates representing each cluster's center.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(unknown_centers)
    labels = clustering.labels_

    cluster_centers = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # skip noise
        cluster_points = unknown_centers[labels == cluster_id]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center.astype(int))
    cluster_centers = np.array(cluster_centers).reshape(-1, 2)
    return cluster_centers

def frontier_to_unknown(frontier_locations, occupancy_map, radius=64):
    unknown = (occupancy_map == 0) * 1.0
    obstacle = (occupancy_map == 1) * 1.0
    free = (occupancy_map == 2) * 1.0
    # unknown, obstacle, free = seperate_map(occupancy_map)
    unknown_centers = []
    count = 1.0 * (radius*2) ** 2
    unknown_grids = []
    for i, loc in enumerate(frontier_locations):
        mask = get_visible_unknown(loc,600,obstacle,unknown,radius)
        center = get_mask_center(mask)
        if center is None:
            continue
        unknown_centers.append(center)
        unknown_grids.append(np.count_nonzero(mask)/count)
    
    unknown_idx = [i for i, count in enumerate(unknown_grids) if count >= 0.1]
    print("Unknown grids: ", unknown_grids)
    if len(unknown_idx) == 0:
        return []
    unknown_centers = np.array(unknown_centers)
    if len(unknown_idx) > 1:
        unknown_centers = get_unknown_cluster_centers(unknown_centers[unknown_idx])
    else:
        unknown_centers = unknown_centers[unknown_idx]
    return unknown_centers

def process_step_data(i, episode_label, scene_data, clip_model_list, experiment_name, llm_name, vlm_name, vlm, mode, db_saved=None):
    # load step data
    step_data = scene_data.load_step_data(i, episode_label)
    if step_data is None:
        return None

    # set vln logger and create dump directory
    vln_logger.info(f"Processing step {i} of episode {step_data['episode_label']}")
    set_vln_logger(step_data, experiment_name)
    dump_dir = f"./dump/{experiment_name}/"
    llm_name = '_'.join(llm_name.split(':'))
    save_path = f"{dump_dir}bev_{llm_name}/{episode_label}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pkl_datapath = f"{save_path}/{i}.pkl".replace('prediction', 'generation').replace('_unknowncenter', '')
    if not os.path.exists(pkl_datapath):
        return None
    
    # load scene graph from file
    target = step_data['cate_object']
    observed_scenegraph_file = step_data['global_scene_graph']
    gt_scenegraph_file = step_data['gt_scenegraph']
    agent_location = step_data['current_grid_pose']
    frontier_locations = np.array(step_data['frontier_candidate_list'])
    global_bev_rgb_map = np.ascontiguousarray(step_data['global_bev_rgb_map'])[::-1].astype(np.uint8)
    occupancy_map = np.ascontiguousarray(step_data['occupancy_map']).astype(np.uint8)
    if observed_scenegraph_file is None or gt_scenegraph_file is None:
        return None

    # prepare scene graph for generation or prediction
    # filtered_observed_scenegraph = filter_scenegraph(observed_scenegraph_file)
    observed_scenegraph = PredictSceneGraph(target=target, clip_model_list=clip_model_list, json_file=observed_scenegraph_file)
    gt_scenegraph = PredictSceneGraph(target=target, clip_model_list=clip_model_list, json_file=gt_scenegraph_file)
    if mode == 'prediction':
        generated_scenegraph = load_from_pickle(pkl_datapath)
        if 'predicted_scenegraph' in generated_scenegraph:
            scenegraph_regrouped = observed_scenegraph.assign_ids(generated_scenegraph['predicted_scenegraph'])
        else:
            return None
    else:
        # gt_scenegraph_dict = process_gtsg_as_obssg(step_data, deepcopy(gt_scenegraph.scene_graph))
        scenegraph_regrouped = regroup_scenegraph(step_data, copy.deepcopy(observed_scenegraph.scene_graph), knn_object=30, distance_threshold=50, wall_threshold=10, min_num=3, save_path=save_path)
    if mode == 'prediction':
        # scene graph prediction
        # vln_logger.info(f"Predicting scene graph...")
        start = time.time()
        # processed_scenegraph = observed_scenegraph.predict_global_scenegraph(dump_dir, global_bev_rgb_map, observed_scenegraph.scene_graph, frontier_locations, agent_location, llm_name, vlm)
        unknown_centers = frontier_to_unknown(frontier_locations, occupancy_map)
        if len(unknown_centers) > 0:
            # processed_scenegraph = observed_scenegraph.predict_global_scenegraph_TopK(dump_dir, global_bev_rgb_map, scenegraph_regrouped, unknown_centers, agent_location, llm_name, vlm)
            processed_scenegraph = observed_scenegraph.predict_global_scenegraph_TopK_WithNumber(dump_dir, global_bev_rgb_map, scenegraph_regrouped, unknown_centers, agent_location, llm_name, vlm)
            # processed_scenegraph = observed_scenegraph.predict_global_scenegraph_TopK(dump_dir, global_bev_rgb_map, scenegraph_regrouped, frontier_locations, agent_location, llm_name, vlm)
        else:
            return None
        print("Prediction {:.2f} seconds".format(time.time() - start))
        if processed_scenegraph is None:
            return None
    else:
        # scene graph generation
        # vln_logger.info(f"Generating scene graph...")
        # image = np.array(step_data['rgb_image']).reshape(step_data['rgb_image_shape']) if vlm else None
        start = time.time()
        # processed_scenegraph = observed_scenegraph.generate_region_caption(observed_scenegraph.scene_graph, target, llm_name, vlm_name)
        # processed_scenegraph = observed_scenegraph.generate_region_caption_withconf(observed_scenegraph.scene_graph, target, llm_name, vlm_name)
        # processed_scenegraph = observed_scenegraph.generate_region_caption_global(observed_scenegraph.scene_graph, target, llm_name, vlm_name)
        # processed_scenegraph = observed_scenegraph.generate_region_caption_TopK(observed_scenegraph.scene_graph, target, llm_name, vlm_name)
        # processed_scenegraph = observed_scenegraph.generate_region_caption_TopK(gt_scenegraph_dict, target, llm_name, vlm_name)
        processed_scenegraph = observed_scenegraph.generate_region_caption_TopK(scenegraph_regrouped, target, llm_name, vlm_name)
        if processed_scenegraph is None:
            return None
        # observed_scenegraph.plot_and_save(dump_dir, llm_name, global_bev_rgb_map, frontier_locations, agent_location, processed_scenegraph, None)
        bev_image_path = f"{save_path}/{i}_occ_map.png"
        bev_image = visualize_BEV(global_bev_rgb_map, processed_scenegraph, bev_image_path, target_locations=frontier_locations, agent_location=agent_location)
        print("Generation {:.2f} seconds".format(time.time() - start))

    # evaluate scene graph
    # accuracy = observed_scenegraph.evaluate_scenegraph(processed_scenegraph, gt_scenegraph)
    obs_regions, gt_regions, obs_objects, gt_objects = generate_evalformat(step_data, processed_scenegraph, gt_scenegraph.scene_graph)
    matches, metric = eval_scenegraph(clip_model_list, obs_regions, gt_regions, obs_objects, gt_objects, topk=3)
    vis_matches(f"./dump/{experiment_name}/bev_{llm_name}/{episode_label}/{i}", matches, global_bev_rgb_map, obs_regions, gt_regions)
    metric.update({'esisode_label': step_data['episode_label'], 'step': i})
    data = {
        'step': i,
        'origins_grid': step_data['origins_grid'],
        'camera_position': step_data['camera_position'],
        'episode_label': step_data['episode_label'],
        'target': target,
        'gt_scenegraph': gt_scenegraph.scene_graph,
        'observed_scenegraph': observed_scenegraph.scene_graph,
    }
    if mode == 'generation':
        data.update({'generated_scenegraph': processed_scenegraph})
    else:
        data.update({'predicted_scenegraph': processed_scenegraph})
    
    save_to_pickle(data, f"{save_path}/{i}.pkl")
    # insert_data(db_saved, 'step_data', data)
    # print("Step data inserted to db: ", data)
    return metric


def filter_scenegraph(scenegraph):
    # Filter the scene graph to remove unnecessary nodes and edges
    filtered_object = ['balustrade']
    for room in scenegraph["rooms"]:
        regions = room.get("regions", [])
        for i, region in enumerate(regions):
            objects = region.get("objects", [])
            # Filter out objects that are not relevant
            filtered_objects = [obj for obj in objects if obj.get("caption") not in filtered_object]
            region["objects"] = filtered_objects
    return scenegraph

def test_process_scenegraph(metric_logger, scene_data, clip_model_list, experiment_name, llm_name, vlm_name, vlm, mode='prediction', db_saved=None):
    dump_dir = f"./dump/{experiment_name}/"
    llm_name = '_'.join(llm_name.split(':'))
    # Process each episode in parallel
    count = 0
    for episode_data in tqdm(scene_data, desc="Episodes"):
        episode_label = episode_data['episode_label'].values[0]
        steps = max(scene_data.episode_data['step'].values)

        # Submit a job to process each step in parallel
        # i = (steps - 1) // 5 * 5 # For gt scene graph
        for i in tqdm(range(50, steps, 30), desc=f"Steps in {episode_label}", leave=False):
            pkl_datapath = f"{dump_dir}bev_{llm_name}/{episode_label}/{i}.pkl"
            if os.path.exists(pkl_datapath):
                continue
            metric = process_step_data(i, episode_label, scene_data, clip_model_list, experiment_name, llm_name, vlm_name, vlm, mode, db_saved)
            if metric is not None:
                metric_logger.log(metric)
                metric_logger.write_averages()
        # count += 1
        # if count > 10:
        #     break
    return metric_logger

# def worker_fn(args):
#     """Unpack arguments and call the step processing function."""
#     return process_step_data(*args)

# def process_scenegraph(metric_logger, scene_data, clip_model_list, experiment_name, llm_name, vlm_name, vlm, mode='prediction'):
#     task_args = []

#     # Prepare tasks
#     for episode_data in tqdm(scene_data, desc="Preparing Episodes"):
#         episode_label = episode_data['episode_label'].values[0]
#         steps = len(episode_data)

#         for i in range(30, steps, 20):
#             args = (
#                 i,
#                 episode_label,
#                 scene_data, 
#                 clip_model_list,
#                 experiment_name,
#                 llm_name,
#                 vlm_name,
#                 vlm,
#                 mode
#             )
#             task_args.append(args)

#     # Use multiprocessing Pool to process tasks in parallel
#     with multiprocessing.Pool(processes=8) as pool:
#         for metric in tqdm(pool.imap_unordered(worker_fn, task_args), total=len(task_args), desc="Processing Steps"):
#             if metric is not None:
#                 metric_logger.log(metric)
#                 metric_logger.write_averages()
#     return metric_logger


def process_scenegraph(metric_logger, scene_data, clip_model_list, experiment_name, llm_name, vlm_name, vlm, mode='prediction'):
    futures = []

    # ThreadPoolExecutor is safe for GPU models (shared context)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for episode_data in tqdm(scene_data, desc="Episodes"):
            episode_label = episode_data['episode_label'].values[0]
            steps = len(episode_data)

            for i in range(30, steps, 30):
                future = executor.submit(
                    process_step_data,
                    i, episode_label, scene_data,
                    clip_model_list, experiment_name,
                    llm_name, vlm_name, vlm, mode
                )
                futures.append(future)

        # As tasks finish, collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Steps"):
            metric = future.result()
            if metric is not None:
                metric_logger.log(metric)
                metric_logger.write_averages()
    return metric_logger

def test(db_path, experiment_name, metric_logger, mode='prediction', split_l=None, split_r=None, db_saved=None):
    llm_name = 'gpt'
    vlm_name = 'llama3.2-vision'
    vlm = True

    # load clip model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    clip_model = clip_model.to(device='cuda')
    clip_model_list = (clip_model, clip_preprocess, clip_tokenizer)

    # load data
    scene_data = OfflineAgentData(db_path, split_l, split_r)
    metric_logger = test_process_scenegraph(metric_logger, scene_data, clip_model_list, experiment_name, llm_name, vlm_name, vlm, mode, db_saved)
    print("Averages: ", metric_logger.get_averages())


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    # load db data
    os.environ["GPT_API_KEY"] = "sk-Sg58hfwiUGtMak37De998909A76c44E88f7dF36730Dd29B4"
    # db_path = '/root/Projects/junzhe/Projects/sg-vln/dump/offline_analysis_sample100_apr23_copy/objectnav-dino'
    # db_path = '/root/Projects/junzhe/Projects/sg-vln/dump/offline_analysis_sample100_apr24/objectnav-dino/steps'
    # db_path = '/root/Projects/junzhe/Projects/sg-vln/dump/offline_analysis_sample100_apr26/objectnav-dino/steps'
    db_path = './dump/steps'
    # db_path = './db_path'

    ########## Save results in DB ##########
    # save_path = f'{db_path}/scenegraph_generation_debug.db'
    # os.remove(save_path) if os.path.exists(save_path) else None
    # db_saved = connect_db(save_path)
    # print('connected to db')
    # data = {
    #     'step': 0,
    #     'episode_label': 'episode_0',
    #     'target': 'bed',
    # }
    # schema = generate_schema(data)
    # db_saved.create_table('step_data', schema=schema, overwrite=True)
    # # insert_data(db_saved, 'step_data', data)


    print(f"Loading data from db {db_path}")
    args = parse_args()
    experiment_name = 'llm_regrouped_unknowncenter_obs_prediction_top3'
    log_name = 'obs_scenegraph_unknowncenter_prediction_metrics'
    metric_logger = MetricLogger(log_path=f"{log_name}.log", avg_log_path=f"{log_name}_avg.log")

    # print("Test generation...")
    # test(db_path, experiment_name, metric_logger, mode='generation', split_l=args.split_l, split_r=args.split_r, db_saved=None)
    print("Test prediction...")
    test(db_path, experiment_name, metric_logger, mode='prediction', split_l=args.split_l, split_r=args.split_r, db_saved=None)
    # db_saved.disconnect()

