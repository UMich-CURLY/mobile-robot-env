import base64
import math
from collections import Counter
from io import BytesIO
from pathlib import Path, PosixPath
import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
import copy
from time import time
import os

import clip
from sklearn.cluster import DBSCAN

from utils.compute_similarities import compute_spatial_similarities, compute_visual_similarities
from utils.scene_graph_utils import text2value, calculate_entropy, get_gt_bbox_and_caption
from utils.utils_llm import format_response, query_llm, load_json, construct_query
from utils.compute_similarities import merge_detections_to_objects
from utils.slam_classes import MapObjectList
from utils.mapping import filter_objects, gobs_to_detection_list, vote_class_name
from utils.timing_global_tool import PERCEPTION_TIME_EPISODE, SCENE_GRAPH_TIME_EPISODE
from utils.vln_logger import vln_logger
from constants import categories_21

# from conceptgraph.llava.llava_model import LLaVaChat

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]


class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.center = None
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []

# TODO: multy: do we want a house node to be the root node now?

# TODO: if we want to store info, then needed, otherwise only GroupNode and Node are needed
class FloorNode():
    def __init__(self, caption):
        self.caption = caption
        self.center = None
        self.height_min = None
        self.height_max = None
        self.height_avg = None # average height of all the objects in this floor
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []

class GroupNode():
    def __init__(self, caption=''):
        self.caption = caption
        self.exploration_level = 0
        # self.corr_score = 0
        self.center = None
        self.center_node = None
        self.nodes = []
        self.image_idx = None
        self.edges = set()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompts = load_json(os.path.join(current_dir, 'phase.json'))
        self.group_caption_prompt = 'Given the objects [{}] as context, infer the most likely intermediate semantic concept (e.g., furniture type, functional area). Only answer the intermediate semantic concept! #Answer: '
        self._last_caption_object_count = 0

    def __eq__(self, other):
        # Check if the group is actually the same by checking all the nodes
        if not isinstance(other, GroupNode):
            raise TypeError('Can only compare two GroupNode') 
        # iterate through all nodes and compare if all of them are the same
        for node in self.nodes:
            if node not in other.nodes:
                return False
        for node in other.nodes:
            if node not in self.nodes:
                return False
        return True
        
    def __lt__(self, other):
        return self.corr_score < other.corr_score
    
    def update_graph(self):
        self.center = np.array([node.center for node in self.nodes]).mean(axis=0)
        min_distance = np.inf
        for node in self.nodes:
            distance = np.linalg.norm(np.array(node.center) - np.array(self.center))
            if distance < min_distance:
                min_distance = distance
                self.center_node = node
            self.edges.update(node.edges)
    
    def update_image_idx(self):
        nodes = self.nodes
        image_idx1 = nodes[0].object["image_idx"]
        for node2 in nodes[1:]:
            image_idx2 = node2.object["image_idx"]
            image_idx1 = set(image_idx1) | set(image_idx2)
        if len(image_idx1) == 0:
            return None
        conf_list = []
        for idx in image_idx1:
            conf_all = 0.0
            for node in nodes:
                if idx not in node.object["image_idx"]:
                    continue
                conf = node.object["conf"][node.object["image_idx"].index(idx)]
                conf_all += conf
            conf_list.append(conf_all)
        # print(f"[SGNavMemory] Get image from idx: {idx_max}")
        self.image_idx = list(image_idx1)[np.argmax(np.array(conf_list))]

    
    def get_group_caption(self, nodes, image_history, model_name, default_caption='indoor environment'):
        vln_logger.increase_function_call_count('get_group_caption')
        nodes_text = ', '.join(list(set([node.caption for node in nodes])))
        # prompt = self.group_caption_prompt.format(nodes_text)
        prompt = construct_query(self.prompts["RegionCaptionGenerator"], objects=nodes_text)
        image = image_history[self.image_idx]
        max_try_times = 3
        try:
            for i in range(max_try_times):
                if image is not None and len(image.shape) == 3 and image.shape[0]*image.shape[1] > 0:
                    response = query_llm(prompt, llm_name=model_name, image=image)
                else:
                    response = query_llm(prompt, llm_name=model_name)
                response = format_response(response)
                if len(response) != '' or len(response) < 30:
                    break
        except:
            response = default_caption
        print(f"[SGNavMemory] Region caption: {response}")
        return response

    def graph_to_text(self, nodes, edges):
        nodes_text = ', '.join([node.caption for node in nodes])
        edges_text = ', '.join([f"{edge.node1.caption} {edge.relation} {edge.node2.caption}" for edge in edges])
        return f"Nodes: {nodes_text}. Edges: {edges_text}."


class ObjectNode():
    def __init__(self):
        self.is_new_node = True
        self.caption = None
        self.object = None
        self.reason = None
        self.center = None
        self.room_node = None
        self.exploration_level = 0
        self.distance = 2
        self.score = 0.5
        self.edges = set()
        # TODO: multy: add posterior image feature, and token distribution.
        # Original raw features are stored in self.object dictionary.
        self.image_feature_posterior = None
        self.token_distribution_posterior = None

    # def __eq__(self, other):
    #     if not isinstance(other, ObjectNode):
    #         raise TypeError('Can only compare two ObjectNode') 
    #     return self.caption == other.caption and self.object == other.object
    
    def __lt__(self, other):
        return self.score < other.score

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def set_caption(self, new_caption):
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True
        self.caption = new_caption
        self.reason = None
        self.distance = 2
        self.score = 0.5
        self.exploration_level = 0
        self.edges.clear()
    
    def set_object(self, object):
        self.object = object
        self.object['node'] = self
        self.num_detections = object['num_detections']
    
    def set_center(self, center):
        self.center = center
    
    def get_vis_sim_score(self, goal_clip_feat):
        vis_feat = self.object['clip_ft']
        # vis_sim_score = torch.tensor(vis_feat).cuda() @ goal_clip_feat.T
        vis_sim_score = torch.cosine_similarity(torch.tensor(vis_feat).unsqueeze(0).cuda(), goal_clip_feat, dim=-1)
        self.vis_sim_score = vis_sim_score.item()
        return
    
    def update_image_feature(self, img_feats):
        pass
    
    def update_token_distribution(self, token_logits):
        pass


class Edge():
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.relation = None

    def set_relation(self, relation):
        self.relation = relation

    def delete(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def text(self):
        text = '({}, {}, {})'.format(self.node1.caption, self.node2.caption, self.relation)
        return text

class SGNavMemory():
    def __init__(self, args, config, map_resolution, map_size_cm, map_size, camera_matrix, clip_model_list=None, is_navigation=True, scenegraph_type='both', vlm_name='llama3.2-vision') -> None:
        
        self.args = args
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_navigation = is_navigation
        self.scenegraph_type = scenegraph_type
        self.set_cfg()
        
        ### ------ static variable ------ ###
        self.vlm_name = vlm_name
        self.bg_classes = ["wall", "floor", "ceiling"]
        self.rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
        self.threshold_list = {'bathtub': 3, 'bed': 3, 'cabinet': 2, 'chair': 1, 'chest_of_drawers': 3, 'clothes': 2, 'counter': 1, 'cushion': 3, 'fireplace': 3, 'gym_equipment': 2, 'picture': 3, 'plant': 3, 'seating': 0, 'shower': 2, 'sink': 2, 'sofa': 2, 'stool': 2, 'table': 1, 'toilet': 3, 'towel': 2, 'tv_monitor': 0}

        object_captions = '. '.join(categories_21) +'.'
        self.classes = copy.deepcopy(object_captions).strip('.').split('. ') # object category list
        
        ### ------ map related ------ ###
        self.map_resolution = map_resolution
        self.map_size_cm = map_size_cm
        self.map_size = map_size
        self.camera_matrix = camera_matrix

        ### ------ init variables ------ ###
        self.reset()
    
        ### ------ init llm prompts ------ ###
        self.init_llm_prompts()
        self.region_count = 0
        self.caption_list = {}
        
        ### ------ init clip ------ ###
        if clip_model_list is None:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
            self.clip_model.to(device='cuda')
            self.clip_tokenizer = clip.tokenize
        else:
            self.clip_model, self.clip_preprocess, self.clip_tokenizer = clip_model_list
        
        ### ------ never used or don't know how to use ------ ###
        # self.num_of_goal = torch.zeros(full_w, full_h).int()
        # self.classes = ['item']
        # self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        # self.found_goal_times_threshold = 1
        # self.score_source = score_source
        # self.reasoning = 'both'
        # self.PSL_infer = 'one_hot'
        # self.reason_visualization = ''
        
        print("[SGNavMemory] Memory initialized")
    
    def init_llm_prompts(self):
        self.prompt_edge_proposal = '''
Provide the most possible single spatial relationship for each of the following object pairs. Answer with only one relationship per pair, and separate each answer with a newline character. Do not response superfluous text.
Example 1:
Input:
Object pair(s):
(cabinet, chair)
Output:
next to

Example 2:
Input:
Object pair(s):
(table, lamp)
(bed, nightstand)
Output:
on
next to

Now input is: 
Object pair(s):
        '''
        self.prompt_relation = 'What is the spatial relationship between the {} and the {} in the image? You can only answer a word or phrase that describes a spatial relationship.'
        self.prompt_discriminate_relation = 'In the image, do {} and {} satisfy the relationship of {}? Only answer "yes" or "no".'
        self.prompt_room_predict = 'Which room is the most likely to have the [{}] in: [{}]. Only answer the room.'
        self.prompt_graph_corr_0 = 'What is the probability of A and B appearing together. [A:{}], [B:{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_graph_corr_1 = 'What else do you need to know to determine the probability of A and B appearing together? [A:{}], [B:{}]. Please output a short question (output only one sentence with no additional text).'
        self.prompt_graph_corr_2 = 'Here is the objects and relationships near A: [{}] You answer the following question with a short sentence based on this information. Question: {}'
        self.prompt_graph_corr_3 = 'The probability of A and B appearing together is about {}. Based on the dialog: [{}], re-determine the probability of A and B appearing together. A:[{}], B:[{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'

    def set_cfg(self):
        cfg = {'dataset_config': PosixPath('tools/replica.yaml'), 'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680, 'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}', 'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': 'cuda', 'use_iou': True, 'spatial_sim_type': 'overlap', 'visual_sim_type': 'clip', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5, 'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95, 'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95, 'max_bbox_area_ratio': 0.5, 'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.025, 'dbscan_remove_noise': True, 'dbscan_eps': 0.1, 'dbscan_min_points': 10, 'obj_min_points': 0, 'obj_min_detections': 3, 'merge_overlap_thresh': 0.7, 'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1, 'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub', 'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True, 'render_camera_path': 'replica_room0.json', 'max_num_points': 512}
        cfg = DictConfig(cfg)
        if self.is_navigation:
            cfg.sim_threshold = 0.8
            cfg.sim_threshold_spatial = 0.01
        self.cfg = cfg
    
    def reset(self):
        self.full_w = self.map_size
        self.full_h = self.map_size
        self.objects = MapObjectList(device=self.device)
        self.nodes = []
        self.edge_text = ''
        self.edge_list = []
        self.group_nodes = []
        self.init_room_nodes()
        # self.reason = ''

    def set_obj_goal(self, obj_goal):
        self.obj_goal = obj_goal
        self.goal_clip_feat = self.compute_goal_clip_features(obj_goal)

    def set_navigate_steps(self, navigate_steps):
        self.navigate_steps = navigate_steps
    
    def set_gt_semantic_annotation(self, semantic_annotation):
        self.semantic_annotation = semantic_annotation

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        edges = list(edges)
        return edges
    
    def get_text(self):
        text_node = ', '.join([node.caption for node in self.nodes])
        text_edge = ' '.join([' '.join([f"({edge.node1.caption}-->{edge.relation}-->{edge.node2.caption})" for edge in list(node.edges)]) for node in self.nodes])
        return text_node, text_edge
    
    def init_room_nodes(self):
        room_nodes = []
        for caption in self.rooms:
            room_node = RoomNode(caption)
            room_nodes.append(room_node)
        self.room_nodes = room_nodes
        
    def compute_goal_clip_features(self, obj_goal):
        with torch.no_grad():
            tokenized_text = self.clip_tokenizer([obj_goal]).to("cuda")
            text_feat = self.clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
        return text_feat
    
    def get_pose_matrix(self, observations):
        x = self.map_size_cm / 100.0 / 2.0 + observations['gps'][0]
        y = self.map_size_cm / 100.0 / 2.0 - observations['gps'][1]
        t = (observations['compass'] - np.pi / 2)[0] # input degrees and meters
        pose_matrix = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return pose_matrix

    def update_node(self, mapping_results):
        room_map = np.zeros_like(mapping_results.occupancy_map)
        origins_grid = mapping_results.origins_grid
        # update nodes
        for i, node in enumerate(self.nodes):
            caption_ori = node.caption
            caption_new = node.object['caption']
            if caption_ori != caption_new:
                node.set_caption(caption_new)
            if node.num_detections != node.object['num_detections']:
                node.num_detections = node.object['num_detections'] # meaning the node has been updated
                node.get_vis_sim_score(self.goal_clip_feat) # also update the clip similarity
        # add new nodes
        new_objects = list(filter(lambda object: 'node' not in object, self.objects))
        for new_object in new_objects:
            new_node = ObjectNode()
            caption = new_object['caption']
            new_node.set_caption(caption)
            new_node.set_object(new_object)
            new_node.get_vis_sim_score(self.goal_clip_feat)
            self.nodes.append(new_node)
        # get node.center and node.room
        for node in self.nodes: 
            points = np.asarray(node.object['pcd'].points)
            center = points.mean(axis=0)
            x = int(center[0] * 100 / self.map_resolution + origins_grid[0])
            y = int(center[2] * 100 / self.map_resolution + origins_grid[1])
            # y = self.map_size - 1 - y
            node.set_center([x, y])
            # TODO: multy the code below should be replaced by floor?
            if 0 <= x < self.map_size and 0 <= y < self.map_size and hasattr(self, 'room_map'):
                if sum(room_map[0, :, y, x]!=0).item() == 0:
                    room_label = 0
                else:
                    room_label = torch.where(room_map[0, :, y, x]!=0)[0][0].item()
            else:
                room_label = 0
            if node.room_node is not self.room_nodes[room_label]:
                if node.room_node is not None:
                    node.room_node.nodes.discard(node)
                node.room_node = self.room_nodes[room_label]
                node.room_node.nodes.add(node)
                centers = np.array([object_node.center for object_node in node.room_node.nodes])
                # print('update room: ', node.room_node.caption)
                node.room_node.center = centers.mean(axis=0)

    def update_edge(self, mapping_results):
        occupancy_map = mapping_results.occupancy_map
        fbe_free_map = np.zeros_like(occupancy_map)
        fbe_free_map[occupancy_map == 2] = 1

        
        old_nodes = []
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False
            else:
                old_nodes.append(node)
        if len(new_nodes) == 0:
            return
        # create the edge between new_node and old_node
        new_edges = []
        for i, new_node in enumerate(new_nodes):
            for j, old_node in enumerate(old_nodes):
                new_edge = Edge(new_node, old_node)
                new_edges.append(new_edge)
        # create the edge between new_node
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]):
                new_edge = Edge(new_node1, new_node2)
                new_edges.append(new_edge)
        # get all new_edges
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        for new_edge in new_edges:
            image = self.get_joint_image(new_edge.node1, new_edge.node2)
            if image is not None:
                prompt = self.prompt_relation.format(new_edge.node1.caption, new_edge.node2.caption)
                response = self.get_vlm_response(prompt=prompt, image=image)
                response = response.replace('.', '').lower()
                new_edge.set_relation(response)
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        # get all relation proposals
        if len(new_edges) > 0:
            node_pairs = []
            for new_edge in new_edges:
                node_pairs.append(new_edge.node1.caption)
                node_pairs.append(new_edge.node2.caption)
            prompt = self.prompt_edge_proposal + '\n({}, {})' * len(new_edges)
            prompt = prompt.format(*node_pairs)
            relations = self.get_llm_response(prompt=prompt)
            relations = relations.split('\n')
            # print('Pairs: ', node_pairs)
            # print('Response: ', relations)
            if len(relations) == len(new_edges):
                for i, relation in enumerate(relations):
                    new_edges[i].set_relation(relation)
            self.free_map = fbe_free_map > 0.5
            for i, new_edge in enumerate(new_edges):
                if new_edge.relation == None or not self.discriminate_relation(new_edge):
                    new_edge.delete()
    
    def edge_post(self, response, edges):
        node_pairs = [[edge.node1.caption, edge.node2.caption] for edge in edges]
        relation_list = ['' for _ in edges]
        rel_node_pairs = []
        rel_list = []
        print('Response: ', response.split('\n'))
        for relation in response.split('\n'):
            if '-' in relation:
                relation_split = relation.split(' - ')
                if len(relation_split) == 2:
                    cur_node_pair, cur_relation = relation_split
                    cur_node_pair = cur_node_pair.strip('(').strip(')').split(',')
                    cur_node_pair = [x.strip() for x in cur_node_pair]
                    rel_node_pairs.append(cur_node_pair)
                    rel_list.append(cur_relation)
        for i, node_pair in enumerate(node_pairs):
            for j, rel_node_pair in enumerate(rel_node_pairs):
                if node_pair == rel_node_pair:
                    relation_list[i] = rel_list[j]
        print('Pairs: ', node_pairs)
        print('Relations: ', relation_list)
        return relation_list
    
    def update_group(self, perception_results):
        # print('------------ update group ------------')
        image_history = perception_results.image_history
        default_caption = 'indoor environment'
        
        for room_node in self.room_nodes:
            if len(room_node.nodes) > 0:
                # generate new group nodes
                new_group_nodes = []
                object_nodes = list(room_node.nodes)
                centers = [object_node.center for object_node in object_nodes]
                centers = np.array(centers)
                dbscan = DBSCAN(eps=32, min_samples=1)  
                clusters = dbscan.fit_predict(centers)  
                for i in range(clusters.max() + 1):
                    group_node = GroupNode()
                    indices = np.where(clusters == i)[0]
                    for index in indices:
                        group_node.nodes.append(object_nodes[index])
                    group_node.update_graph()
                    group_node.update_image_idx()
                    new_group_nodes.append(group_node)
                    
                # check if group nodes have already been updated
                last_group_nodes = room_node.group_nodes
                for old_group_node in last_group_nodes:
                    # check if nodes are the same
                    if old_group_node in new_group_nodes:
                        new_index = new_group_nodes.index(old_group_node)
                        new_group_node = new_group_nodes[new_index]
                        old_group_node.update_image_idx()

                        # generate caption
                        caption_required = np.array([node.object['updated_at'] == self.navigate_steps for node in new_group_node.nodes]).any()
                        model_name = self.vlm_name
                        if self.config["utility"]["hierarchical"]["region"]:
                            if caption_required:
                                old_group_node.caption = old_group_node.get_group_caption(self.nodes, image_history, model_name, default_caption=default_caption)
                        else:
                            self.caption = default_caption

                        new_group_nodes.remove(new_group_node)
                        new_group_nodes.append(old_group_node)

                # if self.args.debug:
                # for group_node in room_node.group_nodes:
                #     vln_logger.info(f"Prev GroupNode: {group_node.caption}, with objects: {[node.caption for node in group_node.nodes]}", extra={'module': "SGNavMemory"})
                room_node.group_nodes = new_group_nodes
                # if self.args.debug:
                # for group_node in room_node.group_nodes:
                #     vln_logger.info(f"GroupNode: {group_node.caption}, with objects: {[node.caption for node in group_node.nodes]}", extra={'module': "SGNavMemory"})
                # print('----------------------------------------')

    
    def update_scenegraph(self, objects, mapping_results, perception_results):
        """
        Update the scene graph and save only the current navigation step.
        """
        if len(objects)>0:
            scene_graph_start = time()
            self.objects = objects
            vote_class_name(self.objects)
            self.update_node(mapping_results)
            self.update_group(perception_results)
            if self.scenegraph_type == 'both':
                print("[SGNavMemory] Using scenegraph = both is not desired for our current implementation. Please use `--scenegraph node` option")
                self.update_edge()
            # SCENE_GRAPH_TIME_EPISODE[-1] += time() - scene_graph_start


    def get_joint_image(self, node1, node2):
        image_idx1 = node1.object["image_idx"]
        image_idx2 = node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        if len(image_idx) == 0:
            return None
        conf_max = -np.inf
        # get joint images of the two nodes
        for idx in image_idx:
            conf1 = node1.object["conf"][image_idx1.index(idx)]
            conf2 = node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        image = self.segment2d_results[idx_max]["image_rgb"]
        return image

    def discriminate_relation(self, edge):
        image = self.get_joint_image(edge.node1, edge.node2)
        if image is not None:
            response = self.get_vlm_response(self.prompt_discriminate_relation.format(edge.node1.caption, edge.node2.caption, edge.relation), image)
            print('VLM discriminate_relation response: ',response )
            if 'yes' in response.lower():
                return True
            else:
                return False
        else:
            if edge.node1.room_node != edge.node2.room_node:
                return False
            x1, y1 = edge.node1.center
            x2, y2 = edge.node2.center
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance > self.map_size // 40:
                return False
            alpha = math.atan2(y2 - y1, x2 - x1)  
            sin_2alpha = 2 * math.sin(alpha) * math.cos(alpha)
            if not -0.05 < sin_2alpha < 0.05:
                return False
            n = 3
            for i in range(1, n):
                x = int(x1 + (x2 - x1) * i / n)
                y = int(y1 + (y2 - y1) * i / n)
                if not self.free_map[y, x]:
                    return False
            return True

    def graph_corr(self, goal, graph):
        prompt = self.prompt_graph_corr_0.format(graph.center_node.caption, goal)
        response_0 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_1.format(graph.center_node.caption, goal)
        response_1 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_2.format(graph.caption, response_1)
        response_2 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_3.format(response_0, response_1 + response_2, graph.center_node.caption, goal)
        response_3 = self.get_llm_response(prompt=prompt)
        corr_score = text2value(response_3)
        return corr_score
                
    # def insert_goal(self, goal=None):
    #     if goal is None:
    #         goal = self.obj_goal
    #     self.update_group()
    #     room_node_text = ''
    #     for room_node in self.room_nodes:
    #         if len(room_node.group_nodes) > 0:
    #             room_node_text = room_node_text + room_node.caption + ','
    #     # room_node_text[-2] = '.'
    #     if room_node_text == '':
    #         return None
    #     prompt = self.prompt_room_predict.format(goal, room_node_text)
    #     response = self.get_llm_response(prompt=prompt)
    #     response = response.lower()
    #     print('LLM room response: ', response)
    #     predict_room_node = None
    #     for room_node in self.room_nodes:
    #         if len(room_node.group_nodes) > 0 and room_node.caption.lower() in response:
    #             predict_room_node = room_node
    #     if predict_room_node is None:
    #         return None
    #     for group_node in predict_room_node.group_nodes:
    #         corr_score = self.graph_corr(goal, group_node)
    #         group_node.corr_score = corr_score
    #     sorted_group_nodes = sorted(predict_room_node.group_nodes)
    #     self.mid_term_goal = sorted_group_nodes[-1].center
    #     print('Group node: ', sorted_group_nodes[-1].caption.lower(), sorted_group_nodes[-1].corr_score)
    #     print('Mid term goal: ', self.mid_term_goal)
    #     return self.mid_term_goal

    # def crop_image_and_mask(self, image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    #     """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
        
    #     image = np.array(image)
    #     # Verify initial dimensions
    #     if image.shape[:2] != mask.shape:
    #         print("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
    #         return None, None

    #     # Define the cropping coordinates
    #     x1 = max(0, x1 - padding)
    #     y1 = max(0, y1 - padding)
    #     x2 = min(image.shape[1], x2 + padding)
    #     y2 = min(image.shape[0], y2 + padding)
    #     # round the coordinates to integers
    #     x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    #     # Crop the image and the mask
    #     image_crop = image[y1:y2, x1:x2]
    #     mask_crop = mask[y1:y2, x1:x2]

    #     # Verify cropped dimensions
    #     if image_crop.shape[:2] != mask_crop.shape:
    #         print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
    #         return None, None
        
    #     # convert the image back to a pil image
    #     image_crop = Image.fromarray(image_crop)

    #     return image_crop, mask_crop

    # def process_cfg(self, cfg: DictConfig):
        # cfg.dataset_root = Path(cfg.dataset_root)
        # cfg.dataset_config = Path(cfg.dataset_config)

        # if cfg.dataset_config.name != "multiscan.yaml":
        #     # For datasets whose depth and RGB have the same resolution
        #     # Set the desired image heights and width from the dataset config
        #     dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        #     if cfg.image_height is None:
        #         cfg.image_height = dataset_cfg.camera_params.image_height
        #     if cfg.image_width is None:
        #         cfg.image_width = dataset_cfg.camera_params.image_width
        #     print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
        # else:
        #     # For dataset whose depth and RGB have different resolutions
        #     assert cfg.image_height is not None and cfg.image_width is not None, \
        #         "For multiscan dataset, image height and width must be specified"

        # return cfg