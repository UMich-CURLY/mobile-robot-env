from test_scenegraph_offline import load_from_pickle
from utils.db_utils import get_df, get_data, connect_db, DB
from test_scenegraph_offline import eval_scenegraph, generate_evalformat, MetricLogger
from utils.scene_graph_utils import detect_match, get_text_feat
import numpy as np
import open_clip
import os
import pickle
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

def get_pkl_files(folder_path):
    """Return a list of files ending with .pkl in the given folder."""
    db_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    return db_files

def parse_args():
    parser = argparse.ArgumentParser(description="SceneGraph Processing with CLIP + VLM")
    # Optional runtime settings
    parser.add_argument('--split_l', type=int, default=0, help='Start Episode inference')
    parser.add_argument('--split_r', type=int, default=100, help='End Episode inference')

    return parser.parse_args()


def eval_topk(result_path, log_name, split_l=0, split_r=100, mode='generation'):
    key = 'generated_scenegraph' if mode == 'generation' else 'predicted_scenegraph'
    # load clip model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    clip_model = clip_model.to(device='cuda')
    clip_model_list = (clip_model, clip_preprocess, clip_tokenizer)

    metric_logger = []
    for topk in range(2):
        metric_logger.append(MetricLogger(log_path=f"{log_name}_{topk}.log", avg_log_path=f"{log_name}_avg_{topk}.log"))

    episode_labels = os.listdir(result_path)[split_l:split_r]
    for episode_label in tqdm(episode_labels, desc="Episodes"):
        step_files = get_pkl_files(os.path.join(result_path, episode_label))
        step_files = sorted(step_files)
        for step_file in tqdm(step_files, desc=f"Steps in {episode_label}", leave=False):
            scenegraph = load_from_pickle(os.path.join(result_path, episode_label, step_file))
            obs_regions, gt_regions, obs_objects, gt_objects = generate_evalformat(scenegraph, scenegraph[key], scenegraph['gt_scenegraph'])

            for topk in range(1, 3):
                matches, metric = eval_scenegraph(clip_model_list, obs_regions, gt_regions, obs_objects, gt_objects, topk=topk)
                if metric is not None:
                    metric_logger[topk-1].log(metric)
                    metric_logger[topk-1].write_averages()
    return

def get_conf_matrix(result_path, log_name, split_l=0, split_r=100, mode='generation'):
    key = 'generated_scenegraph' if mode == 'generation' else 'predicted_scenegraph'
    # load clip model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    clip_model = clip_model.to(device='cuda')
    clip_model_list = (clip_model, clip_preprocess, clip_tokenizer)

    metric_logger = MetricLogger(log_path=f"{log_name}_confmat.log", avg_log_path=f"{log_name}_confmat_avg.log")
    episode_labels = os.listdir(result_path)[split_l:split_r]
    all_caption_true = []
    all_caption_pred = []
    for episode_label in tqdm(episode_labels, desc="Episodes"):
        step_files = get_pkl_files(os.path.join(result_path, episode_label))
        step_files = sorted(step_files)
        
        for step_file in tqdm(step_files, desc=f"Steps in {episode_label}", leave=False):
            scenegraph = load_from_pickle(os.path.join(result_path, episode_label, step_file))
            obs_regions, gt_regions, obs_objects, gt_objects = generate_evalformat(scenegraph, scenegraph[key], scenegraph['gt_scenegraph'])
            # matches, metric = eval_scenegraph(clip_model_list, obs_regions, gt_regions, obs_objects, gt_objects, topk=3)
            region_matches, region_recall = detect_match(
                clip_model_list=clip_model_list,
                keys=gt_regions,
                queries=obs_regions,
                knn=3,
                overlap_relaxed=True,
                corr_score=None,
                topk=3
            )
            y_true = [match['k']['caption'] for match in region_matches]
            y_pred = [match['captions'][match['sim_scores'].index(max(match['sim_scores']))] for match in region_matches]
            all_caption_true.extend(y_true)
            all_caption_pred.extend(y_pred)
    
    # Generate confusion matrix
    labels = sorted(set(all_caption_true + all_caption_pred))  # List of all unique captions
    conf_matrix = confusion_matrix(all_caption_true, all_caption_pred, labels=labels)

    return conf_matrix, labels

def group_labels_by_roomlist(labels, clip_model_list, room_list, threshold=0.7):
    clip_model, clip_preprocess, clip_tokenizer = clip_model_list
    device = clip_model.device

    with torch.no_grad():
        # Encode rooms
        room_inputs = clip_tokenizer(room_list).to(device)
        room_embeddings = clip_model.encode_text(room_inputs)
        room_embeddings = room_embeddings / room_embeddings.norm(dim=-1, keepdim=True)

        # Encode labels
        label_inputs = clip_tokenizer(labels).to(device)
        label_embeddings = clip_model.encode_text(label_inputs)
        label_embeddings = label_embeddings / label_embeddings.norm(dim=-1, keepdim=True)

    room_embeddings = room_embeddings.cpu().numpy()
    label_embeddings = label_embeddings.cpu().numpy()

    label_to_group = {}
    extended_room_list = list(room_list)  # Make a copy because we may add new rooms

    for i, label in enumerate(labels):
        sims = np.dot(room_embeddings, label_embeddings[i])
        max_idx = np.argmax(sims)
        max_sim = sims[max_idx]

        if max_sim >= threshold:
            assigned_room = extended_room_list[max_idx]
        else:
            # Treat as a new room
            assigned_room = label
            extended_room_list.append(label)

            # Update room embeddings
            room_embeddings = np.vstack([room_embeddings, label_embeddings[i]])

        label_to_group[label] = assigned_room

    return label_to_group

def remap_conf_matrix(conf_matrix, labels, label_to_group):
    # Map old indices to new group indices
    group_labels = sorted(set(label_to_group.values()))
    group_idx = {label: idx for idx, label in enumerate(group_labels)}

    new_cm = np.zeros((len(group_labels), len(group_labels)), dtype=int)

    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            group_i = label_to_group[label_i]
            group_j = label_to_group[label_j]
            new_i = group_idx[group_i]
            new_j = group_idx[group_j]
            new_cm[new_i, new_j] += conf_matrix[i, j]

    return new_cm, group_labels

if __name__ == '__main__':
    args = parse_args()
    # experiment_name = 'llm_gt_generation_top3'
    # log_name = 'gt_scenegraph_generation_metrics'

    # experiment_name = 'llm_regrouped_obs_generation_top3'
    # log_name = 'obs_scenegraph_generation_metrics'
    experiment_name = 'llm_regrouped_obs_prediction_top3'
    log_name = 'obs_scenegraph_prediction_metrics'
    mode = 'prediction'
    result_path = f'./dump/{experiment_name}/bev_gpt'
    eval_topk(result_path, log_name, split_l=args.split_l, split_r=args.split_r, mode=mode)
    # conf_matrix, labels = get_conf_matrix(result_path, log_name, split_l=args.split_l, split_r=args.split_r)
    # # Display confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    # fig, ax = plt.subplots(figsize=(10, 10))  # Adjust size if many labels
    # disp.plot(ax=ax, xticks_rotation='vertical')
    # plt.title("Confusion Matrix of Region Captions")
    # plt.savefig('ConfusionMatrix.png')

    # import ipdb; ipdb.set_trace()