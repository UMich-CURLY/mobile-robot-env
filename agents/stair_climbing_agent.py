#!/usr/bin/env python3
import math
import time
import os

import torch
import open3d as o3d
from multiprocessing import Process, Queue

from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from PIL import Image
import yaml
import quaternion
from yacs.config import CfgNode as CN
import logging

import numpy as np
import cv2
from skimage import measure
import skimage.morphology
from collections import Counter
import matplotlib.pyplot as plt

import open_clip
from utils.fmm_planner import FMMPlanner
from utils.mapping import (
    merge_obj2_into_obj1, 
    denoise_objects,
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
    get_camera_K
)
from utils.model_utils import compute_clip_features
from utils.slam_classes import DetectionList, MapObjectList
from utils.detection_segmentation import Object_Detection_and_Segmentation
from utils.compute_similarities import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
    color_by_clip_sim,
    cal_clip_sim
)
from constants import color_palette, category_to_id, GRADIENT_THRESHOLD,  OBSTACLE_HEIGHT, LOCAL_GRID_RANGE, ACTUAL_CAMERA_HEIGHT, FLOOR_HEIGHT #, category_to_id_replica
from utils.vis import init_vis_image, draw_line, vis_result_fast, add_img, add_text, get_pyplot_as_numpy_img
from utils.explored_map_utils import (
    build_full_scene_pcd,
    detect_frontier,
)
import utils.pose as pu

# import keyboard

# Disable torch gradient computation
torch.set_grad_enabled(False)
   
   

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
UP_KEY="q"
DOWN_KEY="e"
FINISH="f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class Stair_Climbing_Agent(Agent):
    def __init__(self, args, follower=None) -> None:
        
        # ------------------------------------------------------------------
        ##### Initialize basic config
        # ------------------------------------------------------------------
        self.args = args
        self.episode_n = 0
        print("init agent")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        self.device = "cuda:{}".format(self.args.gpu_id)

        self.dump_dir = "{}/{}/".format(args.dump_location, args.exp_name)

        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # ------------------------------------------------------------------
        ##### Initialize the perception model
        # ------------------------------------------------------------------
        self.classes=["potted plant", "table", "chair", "couch", "bed", "toilet", "tv", "stairs", "balustrade", "rail", "stool", "camera"]
        
        self.obj_det_seg = Object_Detection_and_Segmentation(self.args, self.classes, self.device)
        
        # Initialize the CLIP model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        self.clip_model = self.clip_model.to(self.device).half()

        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        
        self.annotated_image = None


        # 3D mapping
        self.camera_K = get_camera_K(
            self.args.frame_width, self.args.frame_height, self.args.hfov)

        fcfg = open('configs/mapping_base.yaml')
        self.cfg = CN.load_cfg(fcfg)
        self.init_map_and_pose()
        # ------------------------------------------------------------------
            
        # ------------------------------------------------------------------
        ##### Initialize navigation
        # ------------------------------------------------------------------
        if follower != None:
            self.follower = follower
            
        self.text_queries = 'chair'
        
        self.turn_angle = args.turn_angle
        self.init_map_and_navigation_param()
        
        ### for vidoe saving ###
        self.vis_frames = []
        # ------------------------------------------------------------------
        
        # debug parameters
        self.use_keyboard = args.keyboard_actor

    def reset(self, episode_label) -> None:
        self.episode_n += 1
        self.episode_label = episode_label
        self.init_map_and_pose()
        self.init_map_and_navigation_param()
        ### for vidoe saving ###
        self.vis_frames = []
        self.another_floor = True
        self.upstair_flag = self.args.stair_aim == 1
        self.downstair_flag = self.args.stair_aim == 0
        self.start_climbing = False
        self.climb_success = False
        self.found_stair_goal = False
        self.last_frontier_goal = None
        
    def init_map_and_pose(self):
        # local map
        self.map_size = self.args.map_size_cm // self.args.map_resolution
        self.map_real_halfsize  = self.args.map_size_cm / 100.0 / 2.0
        self.local_w, self.local_h = self.map_size, self.map_size
        
        self.explored_map = np.zeros((self.local_w, self.local_h))
        self.obstacle_map = np.zeros((self.local_w, self.local_h))
        self.visited_vis = np.zeros((self.local_w, self.local_h))
        self.goal_map = np.zeros((self.local_w, self.local_h))
        self.similarity_obj_map = np.zeros((self.local_w, self.local_h))
        self.similarity_img_map = np.zeros((self.local_w, self.local_h))
        self.collision_map = np.zeros((self.local_w, self.local_h))
        
        self.last_grid_pose = [self.map_size/2, self.map_size/2]
        self.last_real_pose = [0, 0]
        self.origins_grid = [self.map_size/2, self.map_size/2]
        self.origins_real = [0.0, 0.0]
        self.col_width = 1
   
    def reset_map(self, camera_position):
        self.explored_map = np.zeros((self.local_w, self.local_h))
        self.obstacle_map = np.zeros((self.local_w, self.local_h))
        self.visited_vis = np.zeros((self.local_w, self.local_h))
        self.similarity_obj_map = np.zeros((self.local_w, self.local_h))
        self.similarity_img_map = np.zeros((self.local_w, self.local_h))
        self.collision_map = np.zeros((self.local_w, self.local_h))
        pcd = self.remove_diff_floor_points_cell(self.point_sum, camera_position)
        self.update_map(pcd, camera_position, self.args.map_height_cm / 100.0 /2.0)
        
    def move_map_and_pose(self, shift, axis):
        
        self.explored_map = pu.roll_array(self.explored_map, shift, axis) # need
        self.obstacle_map = pu.roll_array(self.obstacle_map, shift, axis) # need
        self.visited_vis = pu.roll_array(self.visited_vis, shift, axis)
        self.goal_map = pu.roll_array(self.goal_map, shift, axis)
        self.similarity_obj_map = pu.roll_array(self.similarity_obj_map, shift, axis)
        self.similarity_img_map = pu.roll_array(self.similarity_img_map, shift, axis)
        self.collision_map = pu.roll_array(self.collision_map, shift, axis) # need
        
        self.last_grid_pose = pu.roll_pose(self.last_grid_pose, shift, axis)
        self.origins_grid = pu.roll_pose(self.origins_grid, shift, axis)
        self.origins_real = pu.roll_pose(self.origins_real, -shift * self.args.map_resolution / 100.0, axis)

        
    def init_map_and_navigation_param(self):
        
        # 3D mapping
        self.point_sum = o3d.geometry.PointCloud()
        self.init_sim_position = None
        self.init_sim_rotation = None
        self.init_agent_position = None
        self.Open3D_traj = []
        self.objects = MapObjectList(device=self.device)
        self.nearest_point = None
        self.current_grid_pose = None
        
        self.relative_angle = 0
        self.eve_angle = 0

        # navigation
        self.l_step = 0
        
        self.no_frontiers_count = 0
        self.curr_frontier_count = 0
        self.greedy_stop_count = 0
        self.replan_count = 0

        self.is_running = True
        self.found_goal = False
        self.last_action = 0
        
        self.upstair_flag = False
        self.downstair_flag = False
        self.another_floor = False
        self.start_climbing = False
        self.found_stair_goal = False
        self.current_floor_height = 0
        # mainly for visualize and debug
        self.new_floor_pcd_num = None
        self.height_diff = None
        self.climbing_step = 0
        
    
    def act(self, observations, sim_info, send_queue = None, receive_queue = None):
        time_step_info = 'Mapping time (s): \n'

        preprocess_s_time = time.time()
        # ------------------------------------------------------------------
        ##### At first step, get the object name and init the visualization
        # ------------------------------------------------------------------
        env, scene_name, episode_id, agent_state = sim_info['env'], sim_info['scene_name'], sim_info['episode_id'], sim_info['agent_state']
        if self.l_step == 0:
            self.init_sim_position = agent_state.sensor_states["depth"].position
            self.init_agent_position = agent_state.position
            self.init_sim_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)

            cate_object = category_to_id[observations['objectgoal'][0]]
            # cate_object = category_to_id_mp3d[observations['objectgoal'][0]]
            # cate_object = category_to_id_replica[observations['objectgoal'][0]]
            if cate_object == 'sofa':
                cate_object = 'couch'
            if cate_object == 'tv_monitor':
                cate_object = 'tv'
            
            if cate_object == 'tv' or cate_object == 'plant' or cate_object == 'chair' or cate_object == 'toilet':
                self.text_queries = "looks like a " + cate_object
            else:
                self.text_queries = cate_object
        
        # ------------------------------------------------------------------
        ##### Preprocess the observation
        # ------------------------------------------------------------------
        image_rgb = observations['rgb']
        depth = observations['depth']
        image = transform_rgb_bgr(image_rgb) 
        self.annotated_image = image
        
        get_results, detections = self.obj_det_seg.detect(image, image_rgb, self.classes) 
        
        clip_s_time = time.time()
        image_crops, image_feats, current_image_feats = compute_clip_features(
            image_rgb, detections, self.clip_model, self.clip_preprocess, self.device)
        
        clip_e_time = time.time()
        # print('clip: %.3f秒'%(clip_e_time - clip_s_time)) 
        
        if get_results:
            results = {
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": self.classes,
                "image_crops": image_crops,
                "image_feats": image_feats
                # "text_feats": text_feats
            }

        else:
            results = None

   
        preprocess_e_time = time.time()
        time_step_info += 'Preprocess time:%.3fs\n'%(preprocess_e_time - preprocess_s_time)
        
        # ------------------------------------------------------------------
        ##### Object Set building
        # ------------------------------------------------------------------
        v_time = time.time()

        cfg = self.cfg
        depth = self._preprocess_depth(depth)
        
        camera_matrix_T = self.get_transform_matrix(agent_state)
        camera_position = camera_matrix_T[:3, 3]
        self.Open3D_traj.append(camera_matrix_T)
        self.relative_angle = round(np.arctan2(camera_matrix_T[2][0], camera_matrix_T[0][0])* 57.29577951308232 + 180)
        # print("self.relative_angle: ", self.relative_angle)
        
        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = image_rgb,
            depth_array = depth,
            cam_K = self.camera_K,
            idx = self.l_step,
            gobs = results,
            trans_pose = camera_matrix_T,
            class_names = self.classes,
        ) # point clouds in global frame

        
        obj_time = time.time()
        objv_time = obj_time - v_time
        # print('build objects: %.3f秒'%objv_time) 
        time_step_info += 'build objects time:%.3fs\n'%(objv_time)

        if len(fg_detection_list) > 0 and len(self.objects) > 0 :
            spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
            visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
            agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)

            # Threshold sims according to cfg. Set to negative infinity if below threshold
            agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf')

            self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, agg_sim)

        # Perform post-processing periodically if told so
        if cfg.denoise_interval > 0 and (self.l_step+1) % cfg.denoise_interval == 0:
            self.objects = denoise_objects(cfg, self.objects)
        if cfg.merge_interval > 0 and (self.l_step+1) % cfg.merge_interval == 0:
            self.objects = merge_objects(cfg, self.objects)
        if cfg.filter_interval > 0 and (self.l_step+1) % cfg.filter_interval == 0:
            self.objects = filter_objects(cfg, self.objects)
            
        sim_time = time.time()
        sim_obj_time = sim_time - obj_time 
        # print('calculate merge: %.3f秒'%sim_obj_time) 
        time_step_info += 'calculate merge time:%.3fs\n'%(sim_obj_time)


        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])
        
        # ------------------------------------------------------------------
        ##### 2D Obstacle Map
        # ------------------------------------------------------------------
        f_map_time = time.time()
        
        local_grid_pose = [camera_position[0]*100/self.args.map_resolution + int(self.origins_grid[0]), 
                      camera_position[2]*100/self.args.map_resolution + int(self.origins_grid[1])] # x pointing up, z pointing right in image
        pose_x = int(local_grid_pose[0]) if int(local_grid_pose[0]) < self.map_size-1 else self.map_size-1
        pose_y = int(local_grid_pose[1]) if int(local_grid_pose[1]) < self.map_size-1 else self.map_size-1
        
        # Adjust the centriod of the map when the robot move to the edge of the map
        if pose_x < 100:
            self.move_map_and_pose(shift = 100, axis=0)
            pose_x += 100
        elif pose_x > self.map_size - 100:
            self.move_map_and_pose(shift = -100, axis=0)
            pose_x -= 100
        elif pose_y < 100:
            self.move_map_and_pose(shift = 100, axis=1)
            pose_y += 100
        elif pose_y > self.map_size - 100:
            self.move_map_and_pose(shift = -100, axis=1)
            pose_y -= 100
        
        self.current_grid_pose = [pose_x, pose_y]
        
        # visualize trajectory
        self.visited_vis = draw_line(self.last_grid_pose, self.current_grid_pose, self.visited_vis)
        self.last_grid_pose = self.current_grid_pose
        
        # Collision check
        self.collision_check(camera_position)
        full_scene_pcd = build_full_scene_pcd(depth, image_rgb, self.args.hfov)
        
        
        # build 3D pc map
        full_scene_pcd.transform(camera_matrix_T)
        self.point_sum += self.remove_full_points_cell(full_scene_pcd, camera_position) # filter out higher points
        
        self.update_map(full_scene_pcd, camera_position, self.args.map_height_cm / 100.0 /2.0)

        target_score, target_edge_map, target_point_list = detect_frontier(self.explored_map, self.obstacle_map, self.current_grid_pose, threshold_point=8)
        
        # ------------------------------------------------------------------
        ##### Calculate the similarities 
        # ------------------------------------------------------------------
            
        if not send_queue.empty():
            text_input, self.is_running = send_queue.get()
            if text_input is not None:
                self.text_queries = text_input
            # print("self.text_queries: ", self.text_queries)

        clip_time = time.time()
        candidate_objects = []
        clip_candidate_objects = []
        candidate_id = []
        upstair_candidate_objects = []
        downstair_candidate_objects = []
        all_stair_candidates = [] # TODO: for test stair map
        similarity_threshold = 0.27
        if "toilet" in self.text_queries:
            similarity_threshold = 0.28
        if "chair" in self.text_queries or "plant" in self.text_queries:
            similarity_threshold = 0.26
        similarities = None
        if len(self.objects) > 0:
            self.objects, similarities = color_by_clip_sim(self.text_queries, 
                                                        self.objects, 
                                                        self.clip_model, 
                                                        self.clip_tokenizer)
            
            similarities = similarities.cpu().numpy()
            
            self.objects, stairs_similarities = color_by_clip_sim('stairs', 
                                                        self.objects, 
                                                        self.clip_model, 
                                                        self.clip_tokenizer,
                                                        color_set = False)
            if "couch" in self.text_queries:
                candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
                                if similarities[i] > similarity_threshold +0.015 and self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] in self.text_queries]
            else:
                # candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
                #                 if (similarities[i] > similarity_threshold +0.015 and \
                #                     ((max(self.objects[i]['conf']) > 0.85 and self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] == self.text_queries) or (max(self.objects[i]['conf']) < 0.85))) \
                #                     or \
                #                     (max(self.objects[i]['conf']) > 0.88 and self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] == self.text_queries)]
                candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
                                if ((similarities[i] > similarity_threshold +0.01 and \
                                    self.objects[i]['num_detections'] > 2) \
                                    or \
                                    max(self.objects[i]['conf']) > 0.85) and \
                                    (self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] in self.text_queries and max(self.objects[i]['conf']) > 0.5)]
            
            clip_candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
                                if (similarities[i] > similarity_threshold) and self.objects[i]['num_detections'] > 1 and \
                                    self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] in self.text_queries]
            # candidate_id = [i for i in range(len(self.objects)) 
            #                     if (similarities[i] > similarity_threshold) and self.objects[i]['num_detections'] > 1 and \
            #                         self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] in self.text_queries] 
            
            for i in range(len(self.objects)):
                if stairs_similarities[i] > 0.24: # and self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] == "stairs":
                    all_stair_candidates.append(self.objects[i]['pcd']) # TODO: store all pcd of the stairs
                    
            # for i in range(len(self.objects)):
            #     if stairs_similarities[i] > 0.24 and self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] == "stairs":
            #         all_stair_candidates.append(self.objects[i]['pcd']) # TODO: store all pcd of the stairs
                    # if self.downstair_flag:
                    #     if self.objects[i]['bbox'].min_bound[1] > camera_position[1] - 1.0:
                    #         max_index = np.argmax(np.asarray(self.objects[i]['pcd'].points)[:, 1])
                    #         upstair_candidate_objects.append(self.objects[i]['pcd'].points[max_index])
                    #     else:
                    #         min_index = np.argmin(np.asarray(self.objects[i]['pcd'].points)[:, 1])
                    #         downstair_candidate_objects.append(self.objects[i]['pcd'].points[min_index])
                    # else:
                    #     if self.objects[i]['bbox'].max_bound[1] > camera_position[1] - 0.8:
                    #         # NOTE multy: what does this if statment choice mean? determine how many stair goes up and how many goes down?
                    #         max_index = np.argmax(np.asarray(self.objects[i]['pcd'].points)[:, 1])
                    #         upstair_candidate_objects.append(self.objects[i]['pcd'].points[max_index])
                    #     else:
                    #         min_index = np.argmin(np.asarray(self.objects[i]['pcd'].points)[:, 1])
                    #         downstair_candidate_objects.append(self.objects[i]['pcd'].points[min_index])
                   
                if similarities[i] > 0.24:
                    self.similarity_obj_map[self.object_map_building(self.objects[i]['pcd'])] = similarities[i]
                # TODO: use same method to build map for stairs 
            
            
        image_clip_sim = cal_clip_sim("something near the " + self.text_queries, current_image_feats, self.clip_model, self.clip_tokenizer)
        self.similarity_img_map[self.obstacle_map==1] = image_clip_sim.cpu().numpy()
            
        # ------------------------------------------------------------------
        ##### TODO: Handle stiar map
        # ------------------------------------------------------------------
        # stair_goal_ind, stiar_obstacle_map, self.goal_map = self.stair_climbing_map_draft(all_stair_candidates, camera_position)
        stair_goal_ind, stiar_obstacle_map, self.goal_map = self.stair_climbing_policy(all_stair_candidates, camera_position)
        
        # ------------------------------------------------------------------
        ##### Determine if go to another floor
        # ------------------------------------------------------------------
        # if len(upstair_candidate_objects) > 0:
        #     if self.found_goal == False:
        #         self.goal_map = np.zeros((self.local_w, self.local_h))
        #     max_conf = [max(obj['conf']) for obj in upstair_candidate_objects]
        #     max_index = np.argmax(max_conf)
            
        #     # try to find the stair center
        #     self.goal_map[self.object_map_building(upstair_candidate_objects[max_index]['pcd'])] = 1
        #     self.nearest_point = self.find_nearest_point_cloud(upstair_candidate_objects[max_index]['pcd'], camera_position)
        #     self.found_goal = True
        # else:
        #     self.found_goal = False
            
        if self.another_floor:
            new_floor_pcd = self.detect_plane_points_cell(full_scene_pcd, camera_position)
            print("Number of new floor points:", len(new_floor_pcd.points))
            
            # if we are high enough and get enough points, we are probably on another floor
            # floor hight must be higher than camera height
            self.height_diff = camera_position[1] - self.current_floor_height
            height_criteria = self.height_diff > FLOOR_HEIGHT if self.upstair_flag else self.height_diff < -FLOOR_HEIGHT
            new_floor_thress = 1500 if self.upstair_flag else 1200
            if len(new_floor_pcd.points) > 1500 and height_criteria:
                    ground_pcd = self.plane_segmentation_xy(new_floor_pcd)
                    print(len(ground_pcd.points))
                    
                    if len(ground_pcd.points) > 1500:
                        self.another_floor = False
                        self.start_climbing = False
                        self.uupstair_flag = False
                        self.downstair_flag = False
                        self.climb_success = True
                        self.current_floor_height = camera_position[1]
                        self.reset_map(camera_position)
                        
                        objects_to_keep = []
                        for obj in self.objects:
                            if abs(obj['bbox'].get_center()[1] - camera_position[1])  < 1.0 or 'stairs' in obj['class_name']:
                                objects_to_keep.append(obj)
                        self.objects = MapObjectList(objects_to_keep)

        if stair_goal_ind is None and len(target_point_list) > 0:
            # greedily find the closest frontier to explore
            # max_ind = np.argmax(target_score)
            # closest_frontier = target_point_list[max_ind]
            if self.last_frontier_goal is None or self.curr_frontier_count > 5:
                distance_to_frontier = np.linalg.norm(np.array(self.current_grid_pose) - np.array(target_point_list), axis=1)
                closest_frontier = target_point_list[np.argmin(distance_to_frontier)]
                self.curr_frontier_count = 0 # reset counter
            else:
                closest_frontier = self.last_frontier_goal
            self.goal_map[closest_frontier[0], closest_frontier[1]] = 1
            self.last_frontier_goal = closest_frontier
            self.curr_frontier_count += 1
        elif stair_goal_ind is None and len(target_point_list) == 0:  
            print("Goal to stair not found, using random goal")
            goal_pose_x = int(np.random.rand() * self.map_size)
            goal_pose_y = int(np.random.rand() * self.map_size)
            self.goal_map[goal_pose_x, goal_pose_y] = 1
                
        # self.has_goal = (np.sum(self.goal_map) > 0)
        # ------------------------------------------------------------------
        ##### visualize map
        # ------------------------------------------------------------------
        vis_image = None
        if self.args.print_images or self.args.visualize:
            self.annotated_image  = vis_result_fast(image, detections, self.classes)
            vis_image = self._visualize(self.obstacle_map, self.explored_map, target_edge_map, self.goal_map, self.text_queries)
            
            # self.save_rgbd_image(image, depth)
        
        
        if not self.another_floor:
            # input("Reached another floor, look right? Press to enter next episode")
            return HabitatSimActions.STOP
            
        # ------------------------------------------------------------------
        ##### TODO: Taking action
        # ------------------------------------------------------------------
        action = None
        if self.use_keyboard:
            action = self.keyboard_act() # TODO: overwrite action by keyboard actor
            print("z_angle: ", self.eve_angle)
        
        if action == 'p' or not self.use_keyboard:
            # heuristic determined action, look down to see the map, and turn around to see more
            if self.l_step < 12:
                action = HabitatSimActions.TURN_RIGHT
            elif self.l_step < 12 + 3:
                action = HabitatSimActions.LOOK_DOWN
            elif self.l_step < 12 + 6:
                action = HabitatSimActions.LOOK_UP
            else:
                ################################
                ##### plan action to goal ######
                ################################
                self.stg, self.stop, plan_path = self._get_stg(stiar_obstacle_map, self.current_grid_pose, np.copy(self.goal_map))
                action = self.ffm_act()
                # record planned path
                plan_path = np.array(plan_path) 
                plan_path_x = (plan_path[:, 0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
                plan_path_y = plan_path[:, 0] * 0
                plan_path_z = (plan_path[:, 1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
                plan_path = np.stack((plan_path_x, plan_path_y, plan_path_z), axis=-1)
            
            for i in HabitatSimActions:
                    if HabitatSimActions[i] == action:
                        print("Stair action:", i, action)
                        break
        
        self.l_step += 1
        self.last_action = action
        
        if self.another_floor and (self.found_stair_goal or self.start_climbing):
            self.climbing_step += 1
        
        return action

    # def act(self, observations: Observations, agent_state, send_queue=None, receive_queue=None):
    #     time_step_info = 'Mapping time (s): \n'

    #     preprocess_s_time = time.time()
    #     # ------------------------------------------------------------------
    #     ##### At first step, get the object name and init the visualization
    #     # ------------------------------------------------------------------
    #     if self.l_step == 0:
    #         self.init_sim_position = agent_state.sensor_states["depth"].position
    #         self.init_agent_position = agent_state.position
    #         self.init_sim_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)

    #         cate_object = category_to_id[observations['objectgoal'][0]]
    #         # cate_object = category_to_id_replica[observations['objectgoal'][0]]
    #         if cate_object == 'sofa':
    #             cate_object = 'couch'
    #         if cate_object == 'tv_monitor':
    #             cate_object = 'tv'
            
    #         if cate_object == 'tv' or cate_object == 'plant' or cate_object == 'chair' or cate_object == 'toilet':
    #             self.text_queries = "looks like a " + cate_object
    #         else:
    #             self.text_queries = cate_object

    #     # print("current position: ", agent_state.sensor_states["depth"].position)
    #     # ------------------------------------------------------------------
    #     ##### Preprocess the observation
    #     # ------------------------------------------------------------------
            
    #     image_rgb = observations['rgb']
    #     depth = observations['depth']
    #     image = transform_rgb_bgr(image_rgb) 
    #     self.annotated_image = image
        
    #     get_results, detections = self.obj_det_seg.detect(image, image_rgb, self.classes) 
        
    #     clip_s_time = time.time()
    #     image_crops, image_feats, current_image_feats = compute_clip_features(
    #         image_rgb, detections, self.clip_model, self.clip_preprocess, self.device)
        
    #     clip_e_time = time.time()
    #     # print('clip: %.3f秒'%(clip_e_time - clip_s_time)) 
        
    #     if get_results:
    #         results = {
    #             "xyxy": detections.xyxy,
    #             "confidence": detections.confidence,
    #             "class_id": detections.class_id,
    #             "mask": detections.mask,
    #             "classes": self.classes,
    #             "image_crops": image_crops,
    #             "image_feats": image_feats
    #             # "text_feats": text_feats
    #         }

    #     else:
    #         results = None

   
    #     preprocess_e_time = time.time()
    #     time_step_info += 'Preprocess time:%.3fs\n'%(preprocess_e_time - preprocess_s_time)

    #     # ------------------------------------------------------------------
    #     ##### Object Set building
    #     # ------------------------------------------------------------------
    #     v_time = time.time()

    #     cfg = self.cfg
    #     depth = self._preprocess_depth(depth)
        
    #     camera_matrix_T = self.get_transform_matrix(agent_state)
    #     camera_position = camera_matrix_T[:3, 3]
    #     self.Open3D_traj.append(camera_matrix_T)
    #     self.relative_angle = round(np.arctan2(camera_matrix_T[2][0], camera_matrix_T[0][0])* 57.29577951308232 + 180)
    #     # print("self.relative_angle: ", self.relative_angle)
        
    #     fg_detection_list, bg_detection_list = gobs_to_detection_list(
    #         cfg = self.cfg,
    #         image = image_rgb,
    #         depth_array = depth,
    #         cam_K = self.camera_K,
    #         idx = self.l_step,
    #         gobs = results,
    #         trans_pose = camera_matrix_T,
    #         class_names = self.classes,
    #     ) # point clouds in global frame

        
    #     obj_time = time.time()
    #     objv_time = obj_time - v_time
    #     # print('build objects: %.3f秒'%objv_time) 
    #     time_step_info += 'build objects time:%.3fs\n'%(objv_time)

    #     if len(fg_detection_list) > 0 and len(self.objects) > 0 :
    #         spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
    #         visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
    #         agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)

    #         # Threshold sims according to cfg. Set to negative infinity if below threshold
    #         agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf')

    #         self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, agg_sim)

    #     # Perform post-processing periodically if told so
    #     if cfg.denoise_interval > 0 and (self.l_step+1) % cfg.denoise_interval == 0:
    #         self.objects = denoise_objects(cfg, self.objects)
    #     if cfg.merge_interval > 0 and (self.l_step+1) % cfg.merge_interval == 0:
    #         self.objects = merge_objects(cfg, self.objects)
    #     if cfg.filter_interval > 0 and (self.l_step+1) % cfg.filter_interval == 0:
    #         self.objects = filter_objects(cfg, self.objects)
            
    #     sim_time = time.time()
    #     sim_obj_time = sim_time - obj_time 
    #     # print('calculate merge: %.3f秒'%sim_obj_time) 
    #     time_step_info += 'calculate merge time:%.3fs\n'%(sim_obj_time)


    #     if len(self.objects) == 0:
    #         # Add all detections to the map
    #         for i in range(len(fg_detection_list)):
    #             self.objects.append(fg_detection_list[i])

    #     # ------------------------------------------------------------------
        
    #     # ------------------------------------------------------------------
    #     ##### 2D Obstacle Map
    #     # ------------------------------------------------------------------
    #     f_map_time = time.time()
        
    #     local_grid_pose = [camera_position[0]*100/self.args.map_resolution + int(self.origins_grid[0]), 
    #                   camera_position[2]*100/self.args.map_resolution + int(self.origins_grid[1])] # x pointing up, z pointing right in image
    #     pose_x = int(local_grid_pose[0]) if int(local_grid_pose[0]) < self.map_size-1 else self.map_size-1
    #     pose_y = int(local_grid_pose[1]) if int(local_grid_pose[1]) < self.map_size-1 else self.map_size-1
        
    #     # Adjust the centriod of the map when the robot move to the edge of the map
    #     if pose_x < 100:
    #         self.move_map_and_pose(shift = 100, axis=0)
    #         pose_x += 100
    #     elif pose_x > self.map_size - 100:
    #         self.move_map_and_pose(shift = -100, axis=0)
    #         pose_x -= 100
    #     elif pose_y < 100:
    #         self.move_map_and_pose(shift = 100, axis=1)
    #         pose_y += 100
    #     elif pose_y > self.map_size - 100:
    #         self.move_map_and_pose(shift = -100, axis=1)
    #         pose_y -= 100
        
    #     self.current_grid_pose = [pose_x, pose_y]
        
    #     # visualize trajectory
    #     self.visited_vis = draw_line(self.last_grid_pose, self.current_grid_pose, self.visited_vis)
    #     self.last_grid_pose = self.current_grid_pose
        
    #     # Collision check
    #     self.collision_check(camera_position)
    #     full_scene_pcd = build_full_scene_pcd(depth, image_rgb, self.args.hfov)
        
        
    #     # build 3D pc map
    #     full_scene_pcd.transform(camera_matrix_T)
    #     self.point_sum += self.remove_full_points_cell(full_scene_pcd, camera_position) # filter out higher points
        
    #     obs_i_values, obs_j_values = self.update_map(full_scene_pcd, camera_position, self.args.map_height_cm / 100.0 /2.0)


    #     target_score, target_edge_map, target_point_list = detect_frontier(self.explored_map, self.obstacle_map, self.current_grid_pose, threshold_point=8)
         
    #     v_map_time = time.time()
    #     # print('voxel map: %.3f秒'%(v_map_time - f_map_time)) 
        
    #     # ------------------------------------------------------------------
        
        
    #     # ------------------------------------------------------------------
    #     ##### Calculate the similarities 
    #     # ------------------------------------------------------------------
            
    #     if send_queue is not None and not send_queue.empty():
    #         text_input, self.is_running = send_queue.get()
    #         if text_input is not None:
    #             self.text_queries = text_input
    #         # print("self.text_queries: ", self.text_queries)

    #     clip_time = time.time()
    #     candidate_objects = []
    #     clip_candidate_objects = []
    #     candidate_id = []
    #     upstair_candidate_objects = []
    #     downstair_candidate_objects = []
    #     all_stair_candidates = [] # TODO: for test stair map
    #     similarity_threshold = 0.27
    #     if "toilet" in self.text_queries:
    #         similarity_threshold = 0.28
    #     if "chair" in self.text_queries or "plant" in self.text_queries:
    #         similarity_threshold = 0.26
    #     similarities = None
    #     if len(self.objects) > 0:
    #         self.objects, similarities = color_by_clip_sim(self.text_queries, 
    #                                                     self.objects, 
    #                                                     self.clip_model, 
    #                                                     self.clip_tokenizer)
            
    #         similarities = similarities.cpu().numpy()
            
    #         self.objects, stairs_similarities = color_by_clip_sim('stairs', 
    #                                                     self.objects, 
    #                                                     self.clip_model, 
    #                                                     self.clip_tokenizer,
    #                                                     color_set = False)
    #         if "couch" in self.text_queries:
    #             candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
    #                             if similarities[i] > similarity_threshold +0.015 and self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] in self.text_queries]
    #         else:
    #             # candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
    #             #                 if (similarities[i] > similarity_threshold +0.015 and \
    #             #                     ((max(self.objects[i]['conf']) > 0.85 and self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] == self.text_queries) or (max(self.objects[i]['conf']) < 0.85))) \
    #             #                     or \
    #             #                     (max(self.objects[i]['conf']) > 0.88 and self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] == self.text_queries)]
    #             candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
    #                             if ((similarities[i] > similarity_threshold +0.01 and \
    #                                 self.objects[i]['num_detections'] > 2) \
    #                                 or \
    #                                 max(self.objects[i]['conf']) > 0.85) and \
    #                                 (self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] in self.text_queries and max(self.objects[i]['conf']) > 0.5)]
            
    #         clip_candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
    #                             if (similarities[i] > similarity_threshold) and self.objects[i]['num_detections'] > 1 and \
    #                                 self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] in self.text_queries]
    #         # candidate_id = [i for i in range(len(self.objects)) 
    #         #                     if (similarities[i] > similarity_threshold) and self.objects[i]['num_detections'] > 1 and \
    #         #                         self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] in self.text_queries] 
            
    #         for i in range(len(self.objects)):
    #             if stairs_similarities[i] > 0.24 and self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] == "stairs":
    #                 all_stair_candidates.append(self.objects[i]['pcd']) # TODO: store all pcd of the stairs
    #                 if self.downstair_flag:
    #                     if self.objects[i]['bbox'].min_bound[1] > camera_position[1] - 1.0:
    #                         max_index = np.argmax(np.asarray(self.objects[i]['pcd'].points)[:, 1])
    #                         upstair_candidate_objects.append(self.objects[i]['pcd'].points[max_index])
    #                     else:
    #                         min_index = np.argmin(np.asarray(self.objects[i]['pcd'].points)[:, 1])
    #                         downstair_candidate_objects.append(self.objects[i]['pcd'].points[min_index])
    #                 else:
    #                     if self.objects[i]['bbox'].max_bound[1] > camera_position[1] - 0.8:
    #                         # NOTE multy: what does this if statment choice mean? determine how many stair goes up and how many goes down?
    #                         max_index = np.argmax(np.asarray(self.objects[i]['pcd'].points)[:, 1])
    #                         upstair_candidate_objects.append(self.objects[i]['pcd'].points[max_index])
    #                     else:
    #                         min_index = np.argmin(np.asarray(self.objects[i]['pcd'].points)[:, 1])
    #                         downstair_candidate_objects.append(self.objects[i]['pcd'].points[min_index])
                   
    #             if similarities[i] > 0.24:
    #                 self.similarity_obj_map[self.object_map_building(self.objects[i]['pcd'])] = similarities[i]
    #             # TODO: use same method to build map for stairs 
            
            
    #     image_clip_sim = cal_clip_sim("something near the " + self.text_queries, current_image_feats, self.clip_model, self.clip_tokenizer)
    #     self.similarity_img_map[obs_i_values, obs_j_values] = image_clip_sim.cpu().numpy()
        
    #     f_clip_time = time.time()
    #     # print('calculate clip sim: %.3f秒'%(f_clip_time - clip_time)) 
    #     time_step_info += 'calculate clip sim time:%.3fs\n'%(f_clip_time - clip_time)

    #     # ------------------------------------------------------------------
        

    #     # ------------------------------------------------------------------
    #     ##### frontier selection and determine goal location
    #     # ------------------------------------------------------------------

    #     # Note junzhe:
    #     # 1. check if goal is found in candidate_objects
    #     # 2. or, check if goal is found in clip_candidate_objects
    #     # 3. or, set goal to frontier with max similarity
    #     # 4. or, set goal to the object with max similarity,
    #     #        go to another floor if no frontier for 5 steps
    #     # 5. or, if there is no objects at all, walk randomly

    #     if len(candidate_objects) > 0:
    #         if self.found_goal == False:
    #             self.goal_map = np.zeros((self.local_w, self.local_h))
    #         max_conf = [max(obj['conf']) for obj in candidate_objects]
    #         max_index = np.argmax(max_conf)
    #         self.goal_map[self.object_map_building(candidate_objects[max_index]['pcd'])] = 1
    #         self.nearest_point = self.find_nearest_point_cloud(candidate_objects[max_index]['pcd'], camera_position)
    #         self.found_goal = True
    #     else:
    #         self.found_goal = False
        
    #     # NOTE junzhe: confidence of the object "being" the target or the object being "near" the target
    #     similarity_map = np.max(np.stack([self.similarity_obj_map, self.similarity_img_map]), axis=0)
    #     self.save_similarity_map(similarity_map)
    #     if not self.found_goal:
    #         stg = None
    #         if np.sum(self.goal_map) == 1:
    #             f_pos = np.argwhere(self.goal_map == 1)
    #             stg = f_pos[0]
            
    #         self.goal_map = np.zeros((self.local_w, self.local_h))
            

    #         if len(clip_candidate_objects) > 0 and \
    #             self.objects[np.argmax(similarities)]['num_detections'] < 25 :
    #             self.goal_map = np.zeros((self.local_w, self.local_h))
    #             max_conf = [max(obj['conf']) for obj in clip_candidate_objects]
    #             max_index = np.argmax(max_conf)
    #             self.goal_map[self.object_map_building(clip_candidate_objects[max_index]['pcd'])] = 1
    #             self.nearest_point = self.find_nearest_point_cloud(clip_candidate_objects[max_index]['pcd'], camera_position)
                
    #             if self.greedy_stop_count > 25:
    #                 self.objects[np.argmax(similarities)]['num_detections'] = 25

    #         elif len(target_point_list) > 0:
    #             self.no_frontiers_count = 0
    #             simi_max_score = []
    #             stair_score = []
    #             for i in range(len(target_point_list)):
    #                 fmb = self.get_frontier_boundaries((target_point_list[i][0], 
    #                                                 target_point_list[i][1]),
    #                                                 (self.local_w/12, self.local_h/12),
    #                                                 (self.local_w, self.local_h))
                    
    #                 cropped_sim_map = similarity_map[fmb[0]:fmb[1], fmb[2]:fmb[3]]
    #                 simi_max_score.append(np.max(cropped_sim_map))

    #             # print("simi_max_score: ", simi_max_score)
    #             global_item = 0
    #             if len(simi_max_score) > 0:
    #                 if max(simi_max_score) > 0.22:
    #                     # print(simi_max_score)
    #                     global_item = simi_max_score.index(max(simi_max_score))

    #             # TODO: indicating the score on the map
    #             if np.array_equal(stg , target_point_list[global_item]) and target_score[global_item] < 30:
    #                 self.curr_frontier_count += 1
    #             else:
    #                 self.curr_frontier_count = 0
                    
    #             if self.curr_frontier_count > 20 or self.replan_count > 20 or self.greedy_stop_count > 20:
    #                 self.obstacle_map[target_edge_map == global_item+1] = 1
    #                 self.curr_frontier_count = 0
    #                 self.replan_count = 0
    #                 self.greedy_stop_count = 0
           
    #             self.goal_map[target_point_list[global_item][0], target_point_list[global_item][1]] = 1
    #         elif len(self.objects) > 0:
    #             self.no_frontiers_count += 1
    #             self.goal_map = np.zeros((self.local_w, self.local_h))
    #             max_index = np.argmax(similarities)
    #             self.goal_map[self.object_map_building(self.objects[max_index]['pcd'])] = 1
    #             self.nearest_point = self.find_nearest_point_cloud(self.objects[max_index]['pcd'], camera_position)
            
    #             if self.no_frontiers_count > 5:
    #                 if max(similarities) > 0.25 and ("plant" in self.text_queries or "chair" in self.text_queries or "tv" in self.text_queries):
    #                     self.found_goal = True
    #                 else:
    #                     self.another_floor = True
    #                     if not self.upstair_flag and not self.downstair_flag:
    #                         if len(upstair_candidate_objects) > 0 and len(downstair_candidate_objects) == 0:
    #                             self.upstair_flag = True
    #                         elif len(upstair_candidate_objects) == 0 and len(downstair_candidate_objects) > 0:
    #                             self.downstair_flag = True
    #                         elif len(upstair_candidate_objects) > 0 and len(downstair_candidate_objects) > 0:
    #                             if self.text_queries == "bed":
    #                                 self.upstair_flag = True
    #                             else:
    #                                 self.downstair_flag = True
    #         else:
    #             goal_pose_x = int(np.random.rand() * self.map_size)
    #             goal_pose_y = int(np.random.rand() * self.map_size)
    #             self.goal_map[goal_pose_x, goal_pose_y] = 1
        
    #     # note junzhe: go to the goal if found, otherwise walk randomly
    #     if np.sum(self.goal_map) == 1:
    #         f_pos = np.argwhere(self.goal_map == 1)
    #         stg = f_pos[0]
    #         x = (stg[0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
    #         y = camera_position[1]
    #         z = (stg[1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
    #     elif np.sum(self.goal_map) == 0:
    #         # stg = self._get_closed_goal(self.obstacle_map, self.last_grid_pose, find_big_connect(self.goal_map))
    #         self.found_goal == False
    #         goal_pose_x = int(np.random.rand() * self.map_size)
    #         goal_pose_y = int(np.random.rand() * self.map_size)
    #         self.goal_map[goal_pose_x, goal_pose_y] = 1
    #         f_pos = np.argwhere(self.goal_map == 1)
    #         stg = f_pos[0]
            
    #         x = (stg[0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
    #         y = camera_position[1]
    #         z = (stg[1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
    #     else: # > 1           
    #         x = self.nearest_point[0]
    #         y = self.nearest_point[1]
    #         z = self.nearest_point[2]

    #     Open3d_goal_pose = [x, y, z]
        
    #     # note junzhe: find highest or lowest stair
    #     if self.another_floor and not self.found_goal:
    #         if self.upstair_flag and len(upstair_candidate_objects) > 0:
    #             highest_stair = max(upstair_candidate_objects, key=lambda coord: coord[1]) 
    #             Open3d_goal_pose = [highest_stair[0], highest_stair[1]+0.8, highest_stair[2]]
    #             # TODO: use same method to build map for stairs 
    #             print("upstair_flag!!")
    #         elif self.downstair_flag and len(downstair_candidate_objects) > 0:
    #             lowest_stair = min(downstair_candidate_objects, key=lambda coord: coord[1]) 
    #             Open3d_goal_pose = [lowest_stair[0], lowest_stair[1]+0.8, lowest_stair[2]]
    #             # TODO: use same method to build map for stairs 
    #             print("downstair_flag!!")
    #         else:
    #             self.reset_map(camera_position)
                
    #             new_floor_pcd = self.detect_plane_points_cell(full_scene_pcd, camera_position)
    #             if len(new_floor_pcd.points) > 1500:
    #                 ground_pcd = self.plane_segmentation_xy(new_floor_pcd)
    #                 print(len(ground_pcd.points))
    #                 if len(ground_pcd.points) > 1500:
    #                     self.another_floor = False
                        
    #                     objects_to_keep = []
    #                     for obj in self.objects:
    #                         if abs(obj['bbox'].get_center()[1] - camera_position[1])  < 1.0 or 'stairs' in obj['class_name']:
    #                             objects_to_keep.append(obj)
    #                     self.objects = MapObjectList(objects_to_keep)

                
    #     # ------------------------------------------------------------------
    #     ##### Path planning and action selection
    #     # ------------------------------------------------------------------
    #     Rx = np.array([[0, 0, -1],
    #                 [0, 1, 0],
    #                 [1, 0, 0]])
    #     R_habitat2open3d = self.init_sim_rotation @ Rx.T
    #     self.habitat_goal_pose = np.dot(R_habitat2open3d, Open3d_goal_pose) + self.init_agent_position
    #     habitat_final_pose = self.habitat_goal_pose.astype(np.float32)

    #     # note junzhe: cheat here, search_navigable_path returns shortest path according to the mesh
    #     plan_path = []
    #     plan_path = self.search_navigable_path(
    #         habitat_final_pose
    #     )
    #     if len(plan_path) > 1 and self.another_floor:
    #         vector = plan_path[-1] - plan_path[-2]
    #         one_more_point = (vector / np.linalg.norm(vector)) * 0.5 + plan_path[-1]
    #         one_more_point[1] = plan_path[-1][1]
    #         # if one_more_point[1] >= self.Open3d_goal_pose_old[1]:
    #         # print("init agent pos:", self.init_agent_position)
    #         # print("goal prev:", self.habitat_goal_pose)
    #         self.habitat_goal_pose = one_more_point
    #         # print("goal new:", self.habitat_goal_pose)
    #         plan_path = self.search_navigable_path(
    #             one_more_point
    #         )
    #             # self.Open3d_goal_pose_old = self.habitat_goal_pose
    #         # else:
    #         #     plan_path = self.search_navigable_path(
    #         #         self.Open3d_goal_pose_old
    #         #     )
    #         #     self.habitat_goal_pose = self.Open3d_goal_pose_old
                
    #     if len(plan_path) > 1 and not self.args.fmm_planner:
    #         plan_path = np.dot(R_habitat2open3d.T, (np.array(plan_path) - self.init_agent_position).T).T
    #         action = self.greedy_follower_act(plan_path)
    #     else:
    #         # plan a path by fmm
    #         self.stg, self.stop, plan_path = self._get_stg(self.obstacle_map, self.current_grid_pose, np.copy(self.goal_map))
    #         plan_path = np.array(plan_path) 
    #         plan_path_x = (plan_path[:, 0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
    #         plan_path_y = plan_path[:, 0] * 0
    #         plan_path_z = (plan_path[:, 1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0

    #         plan_path = np.stack((plan_path_x, plan_path_y, plan_path_z), axis=-1)

    #         if not self.use_keyboard:
    #             action = self.ffm_act()

    #     # ------------------------------------------------------------------
    #     ##### TODO: Handel stiar map
    #     # ------------------------------------------------------------------
    #     # stair_goal_ind, stiar_obstacle_map, stair_goal_map = self.stair_climbing_map_draft(all_stair_candidates, camera_position)
    #     stair_goal_ind, stiar_obstacle_map, stair_goal_map = self.stair_climbing_policy(all_stair_candidates, camera_position)
        
       
    #     # action = self.keyboard_act()
    #     vis_image = None
    #     if self.args.print_images or self.args.visualize:
    #         self.annotated_image  = vis_result_fast(image, detections, self.classes)
    #         vis_image = self._visualize(self.obstacle_map, self.explored_map, target_edge_map, self.goal_map, self.text_queries)
            
    #         self.save_rgbd_image(image, depth)

    #     dd_map_time = time.time()
    #     time_step_info += '2d map building time:%.3fs\n'%(dd_map_time - f_map_time)

    #     if len(self.objects) > 0:
    #         time_step_info += 'max similarity: %3fs\n'%(np.max(similarities))
    #     # ------------------------------------------------------------------
    #     ##### Send to Open3D visualization thread
    #     # ------------------------------------------------------------------
    #     if self.args.visualize:
    #         if cfg.filter_interval > 0 and (self.l_step+1) % cfg.filter_interval == 0:
    #             self.point_sum.voxel_down_sample(0.05)
    #         if receive_queue is not None:
    #             receive_queue.put([image_rgb, 
    #                                 depth, 
    #                                 self.annotated_image , 
    #                                 self.objects.to_serializable(), 
    #                                 np.asarray(self.point_sum.points), 
    #                                 np.asarray(self.point_sum.colors), 
    #                                 self.Open3D_traj,
    #                                 self.episode_n,
    #                                 self.episode_label,
    #                                 plan_path,
    #                                 transform_rgb_bgr(vis_image),
    #                                 Open3d_goal_pose,
    #                                 time_step_info, 
    #                                 candidate_id]
    #                                 )

    #     if self.use_keyboard:
    #         action = self.keyboard_act(action) # TODO: overwrite action by keyboard actor
    #         print("z_angle: ", self.eve_angle)
        
    #     # if action == 'p':
    #     #     if stair_goal_ind is not None:
    #     #         ################################
    #     #         ##### plan action to goal ######
    #     #         ################################
    #     #         self.stg, self.stop, plan_path = self._get_stg(stiar_obstacle_map, self.current_grid_pose, np.copy(stair_goal_map))
    #     #         plan_path = np.array(plan_path) 
    #     #         plan_path_x = (plan_path[:, 0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
    #     #         plan_path_y = plan_path[:, 0] * 0
    #     #         plan_path_z = (plan_path[:, 1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
    #     #         plan_path = np.stack((plan_path_x, plan_path_y, plan_path_z), axis=-1)

    #     #         action = self.ffm_act()
    #     #         for i in HabitatSimActions:
    #     #             if HabitatSimActions[i] == action:
    #     #                 print("Stair action:", i, action)
    #     #                 break
    #     #     else:
    #     #         print("Goal to stair not found")
    #     #         action = None
            
    #     self.l_step += 1
    #     self.last_action = action
        

    #     # transfer_time = time.time()
    #     # time_step_info += 'transfer data time:%.3fs\n'%(transfer_time - dd_map_time)

    #     # cv2.imshow("episode_n {}".format(self.episode_label), self.annotated_image)
    #     # cv2.waitKey(1)
    #     # print(time_step_info)

    #     return action
    
    def keyboard_actor(self, observations: Observations, agent_state, send_queue= Queue(), receive_queue= Queue()):
        # ------------------------------------------------------------------
        ##### At first step, get the object name and init the visualization
        # ------------------------------------------------------------------
        if self.l_step == 0:
            self.init_sim_position = agent_state.sensor_states["depth"].position
            self.init_agent_position = agent_state.position
            self.init_sim_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)

            cate_object = category_to_id[observations['objectgoal'][0]]
            # cate_object = category_to_id_replica[observations['objectgoal'][0]]
            if cate_object == 'sofa':
                cate_object = 'couch'
            if cate_object == 'tv_monitor':
                cate_object = 'tv'
            
            if cate_object == 'tv' or cate_object == 'plant' or cate_object == 'chair' or cate_object == 'toilet':
                self.text_queries = "looks like a " + cate_object
            else:
                self.text_queries = cate_object
                
        # ------------------------------------------------------------------
        ##### Preprocess the observation
        # ------------------------------------------------------------------
        image_rgb = observations['rgb']
        depth = observations['depth']
        image = transform_rgb_bgr(image_rgb) 
        self.annotated_image = image
        
        # classes for climing stairs
        self.classes=["stairs", "floor", "other"]
        get_results, detections = self.obj_det_seg.detect(image, image_rgb, self.classes) 
        
        vis_image = None
        if self.args.print_images or self.args.visualize:
            self.annotated_image  = vis_result_fast(image, detections, self.classes)
            # save only annotated image
            ep_dir = '{}episodes/{}/annotated/'.format(
                self.dump_dir, self.episode_label)
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)
            fn = ep_dir + 'Annotated-{}.png'.format(self.l_step)
            cv2.imwrite(fn, self.annotated_image)
            cv2.imwrite(self.dump_dir + "current_state.png", self.annotated_image)
            
            self.vis_frames.append(cv2.cvtColor(self.annotated_image, cv2.COLOR_BGR2RGB))
            # vis_image = self._visualize(self.obstacle_map, self.explored_map, target_edge_map, self.goal_map, self.text_queries)
            
            self.save_rgbd_image(image, depth)
        
        action = self.keyboard_act()
        
        return action
    
    
    def search_navigable_path(self, original_point, offset = 0.1):
        
        plan_path = self.follower.get_path_points(
            original_point
        )
        
        if len(plan_path) > 1:
            return plan_path
  
        # Possible changes to each coordinate
        deltas = [-offset, offset]

        # Generate surrounding points using nested loops
        for dx in deltas:
            for dy in deltas:
                for dz in deltas:
                    new_point = (original_point[0] + dx, original_point[1] + dy, original_point[2] + dz)
                    plan_path = self.follower.get_path_points(
                        new_point
                    )
                    if len(plan_path) > 1:
                        self.habitat_goal_pose = new_point
                        
                        return plan_path
              
        return plan_path  

    def keyboard_act(self):
        # ------------------------------------------------------------------
        ##### Update long-term goal if target object is found
        ##### Otherwise, use the LLM to select the goal
        # ------------------------------------------------------------------

        # keystroke = cv2.waitKey(0)
        # # keystroke = input("Enter a key: ")
        # action = None
        # if keystroke == ord(FORWARD_KEY):
        #     action = HabitatSimActions.MOVE_FORWARD
        #     print("action: FORWARD")
        # elif keystroke == ord(LEFT_KEY):
        #     action = HabitatSimActions.TURN_LEFT
        #     print("action: LEFT")
        # elif keystroke == ord(RIGHT_KEY):
        #     action = HabitatSimActions.TURN_RIGHT
        #     print("action: RIGHT")
        # elif keystroke == ord(UP_KEY):
        #     action = HabitatSimActions.LOOK_UP
        #     print("action: UP")
        #     self.eve_angle += 30
        # elif keystroke == ord(DOWN_KEY):
        #     action = HabitatSimActions.LOOK_DOWN
        #     print("action: DOWN")
        #     self.eve_angle -= 30
        # elif keystroke == ord(FINISH):
        #     action = HabitatSimActions.STOP
        #     print("action: FINISH")
        # else:
        #     print("INVALID KEY")
        
        # keystroke = keyboard.read_key()
        keystroke = input("Enter a key: ")
        print(keystroke)
        action = None
        if keystroke == FORWARD_KEY:
            action = HabitatSimActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == LEFT_KEY:
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == RIGHT_KEY:
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == UP_KEY:
            action = HabitatSimActions.LOOK_UP
            print("action: UP")
            self.eve_angle += 30
        elif keystroke == DOWN_KEY:
            action = HabitatSimActions.LOOK_DOWN
            print("action: DOWN")
            self.eve_angle -= 30
        elif keystroke == FINISH:
            action = HabitatSimActions.STOP
            print("action: FINISH")
        elif keystroke == "p":
            # do not overwrite planning action
            action = 'p'
            print("use planned action")
        else:
            print("INVALID KEY")
        
        # self.l_step += 1
        return action


    
    def greedy_follower_act(self, plan_path):
        
        action_s_time = time.time()

        if self.is_running == False:
            return None   

        action = self.follower.get_next_action(
            self.habitat_goal_pose
        )

        if not self.found_goal and action == 0:
            self.greedy_stop_count += 1
            action = 2
        else:
            self.greedy_stop_count = 0
        
        distance = np.linalg.norm(plan_path[0] - plan_path[1])
        high_diff = plan_path[1][1] - plan_path[0][1]
        angle_goal = math.degrees(math.asin(high_diff/distance))

        angle_agent = (360 - self.relative_angle) % 360.0
        eve_start_x = int(5 * math.sin(math.radians(angle_agent)) + self.current_grid_pose[0])
        eve_start_y = int(5 * math.cos(math.radians(angle_agent)) + self.current_grid_pose[1])
        eve_start_x = min(max(0, eve_start_x), self.map_size - 1)
        eve_start_y = min(max(0, eve_start_y), self.map_size - 1)
        
        
        if not self.use_keyboard:
            if self.another_floor and self.downstair_flag:
                if self.eve_angle > -20:
                    action = 5
                    self.eve_angle -= 30
                elif self.eve_angle < -40:
                    action = 4
                    self.eve_angle += 30
            else:
                if (self.explored_map[eve_start_x, eve_start_y] == 0 or (angle_goal - self.eve_angle) < -self.args.turn_angle/2 ) and self.eve_angle > -90:
                    action = 5
                    self.eve_angle -= 30
                elif self.explored_map[eve_start_x, eve_start_y] == 1 and (angle_goal - self.eve_angle) > self.args.turn_angle/2 and self.eve_angle < 0:
                    action = 4
                    self.eve_angle += 30
            
        action_e_time = time.time()

        # print('acton: %.3f秒'%(action_e_time - action_s_time)) 
        self.l_step += 1
        return action
    
    def ffm_act(self):
        if self.is_running == False:
            return None
        
        action_s_time = time.time()
        # # heuristic greedy action, look down to see the map, and turn around to see more
        # if self.l_step < 3:
        #     action = HabitatSimActions.LOOK_DOWN
        # elif self.l_step < 6:
        #     action = HabitatSimActions.LOOK_UP
        # elif self.l_step < 6 + 12:
        #     action = HabitatSimActions.TURN_RIGHT
        # fmm planner
        if self.stop and self.found_goal and not self.use_keyboard:
            action = 0
        else:
            (stg_x, stg_y) = self.stg
            angle_st_goal = math.degrees(math.atan2(stg_x - self.current_grid_pose[0],
                                                    stg_y - self.current_grid_pose[1]))
            angle_agent = (360 - self.relative_angle) % 360.0
            if angle_agent > 180:
                angle_agent -= 360
            # angle_agent = 360 - self.relative_angle
            relative_angle = angle_agent - angle_st_goal
            if relative_angle > 180:
                relative_angle -= 360
            if relative_angle < -180:
                relative_angle += 360

            eve_start_x = int(5 * math.sin(math.radians(self.relative_angle)) + self.current_grid_pose[0])
            eve_start_y = int(5 * math.cos(math.radians(self.relative_angle)) + self.current_grid_pose[1])
            eve_start_x = min(max(0, eve_start_x), self.map_size - 1)
            eve_start_y = min(max(0, eve_start_y), self.map_size - 1)
            # if eve_start_x >= self.map_size: eve_start_x = self.map_size-1
            # if eve_start_y >= self.map_size: eve_start_y = self.map_size-1 
            # if eve_start_x < 0: eve_start_x = 0 
            # if eve_start_y < 0: eve_start_y = 0 
            camera_holding_angle = 0
            # if found stair goal, then look down and go 
            if self.found_stair_goal and self.upstair_flag:
                camera_holding_angle = -30
            if self.another_floor and self.downstair_flag:
                camera_holding_angle = -30
            if self.start_climbing and self.downstair_flag:
                camera_holding_angle = -60
                
            if self.explored_map[eve_start_x, eve_start_y] == 0 and self.eve_angle > -90:
                action = 5
                self.eve_angle -= 30
            elif self.explored_map[eve_start_x, eve_start_y] == 1 and self.eve_angle < camera_holding_angle:
                action = 4
                self.eve_angle += 30
            elif self.found_stair_goal and self.eve_angle > camera_holding_angle:
                action = 5
                self.eve_angle -= 30
            elif self.found_stair_goal and self.eve_angle < camera_holding_angle:
                action = 4
                self.eve_angle += 30
            elif relative_angle > self.args.turn_angle:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle:
                action = 2  # Left
            # elif relative_angle > self.args.turn_angle / 2.:
            #     action = 7  # Right
            # elif relative_angle < -self.args.turn_angle / 2.:
            #     action = 6  # Left
            else:
                action = 1
        # self.l_step += 1
        action_e_time = time.time()
        # print('action: %.3f秒'%(action_e_time - action_s_time)) 
        return action

    def _get_stg(self, grid, start, goal):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = [0, self.local_w, 0, self.local_h] 

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        # print("grid: ", grid.shape)

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        selem = skimage.morphology.disk(3)
        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            selem) != True
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        traversible[cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], kernel) == 1] = 1
        traversible[cv2.dilate(self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2], kernel) == 1] = 0
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        if ("plant" in self.text_queries or "tv" in self.text_queries) and \
            np.sum(self.goal_map) > 1:
            selem = skimage.morphology.disk(15)
        else: 
            selem = skimage.morphology.disk(5)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        path = []
        path.append(start)

        state = [start[0] - x1, start[1] - y1]
        stg_x, stg_y, replan, stop_f = planner.get_short_term_goal(state)
        stg_x, stg_y = stg_x + x1 , stg_y + y1 
        for i in range(10):
            state = [stg_x - x1 , stg_y - y1 ]
            stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
            stg_x, stg_y = stg_x + x1 , stg_y + y1 
            
            path.append([stg_x, stg_y])
            if stop:
                break

        if replan:
            self.replan_count += 1
            # print("false: ", self.replan_count)
        else:
            self.replan_count = 0

        return (path[1][0], path[1][1]), stop_f, path
    
    def _get_closed_goal(self, grid, start, goal):
        """Get short-term goal"""

        x1, y1, = 0, 0
        x2, y2 = grid.shape
        # grid = np.zeros((self.map_size, self.map_size))
        start_map = np.zeros((self.map_size, self.map_size))
        start_map[start[0], start[1]] = 1

        # print("grid: ", grid.shape)

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        selem = skimage.morphology.disk(3)
        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            selem) != True
        # traversible = np.zeros((self.map_size, self.map_size))
        
        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)
        start_map = add_boundary(start_map, value=0)

        selem = skimage.morphology.disk(5)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        start_map = skimage.morphology.binary_dilation(
            start_map, selem) != True
        start_map = 1 - start_map * 1.
        traversible[goal == 1] = 1
        planner = FMMPlanner(traversible)
        planner.set_multi_goal(start_map)

        # mask = traversible

        dist_map = planner.fmm_dist * goal
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(np.argmin(dist_map), dist_map.shape)

        return goal

    def collision_check(self, camera_position):
        current_real_pose = [camera_position[0], camera_position[2]]
        
        if self.last_action == 1 and self.l_step > 0:
            x1, y1 = self.last_real_pose
            x2, y2 = current_real_pose
            t1 = 360-self.relative_angle
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < self.args.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = y1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = x1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / self.args.map_resolution)+ int(self.origins_grid[0]), \
                            int(c * 100 / self.args.map_resolution)+ int(self.origins_grid[1])
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1
                        # print("collision!")
                        
        self.last_real_pose = current_real_pose   


    def _preprocess_depth(self, depth, min_d=0.5, max_d=5.0):
        # print("depth origin: ", depth.shape)
        depth = depth[:, :, 0] * 1
        # print(np.max(depth))
        # print(np.min(depth))
        # for i in range(depth.shape[1]):
        #     depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        depth = depth * max_d 

        return depth

    def update_map(self, point_sum, camera_position, height_diff):

        explored_map = np.zeros((self.local_w, self.local_h))
        obstacle_map = np.zeros((self.local_w, self.local_h))

        # height range (z is down in Open3D)
        z_min = camera_position[1] - height_diff
        z_max = camera_position[1] + height_diff
        z_stair = camera_position[1] - self.args.map_height_cm / 100.0 

        points = np.asarray(point_sum.points)

        mask_obstacle = (((points[:, 1] >= z_min) & (points[:, 1] <= z_max)) | \
                    (points[:, 1] <= z_stair)) & \
                    (points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                    (points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                    (points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                    (points[:, 2] <= self.origins_real[1] + self.map_real_halfsize)

        points_obstacle = points[mask_obstacle]

        mask_explored = (points[:, 1] <= z_max) & \
                    (points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                    (points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                    (points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                    (points[:, 2] <= self.origins_real[1] + self.map_real_halfsize )

        points_explored = points[mask_explored]

        # 计算二维地图的索引ww
        obs_i_values = np.floor((points_obstacle[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        obs_j_values = np.floor((points_obstacle[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])

        obstacle_map[obs_i_values, obs_j_values] = 1
        self.obstacle_map[obs_i_values, obs_j_values] = 1
        
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        self.obstacle_map[cv2.dilate(self.visited_vis, kernel) == 1] = 0 # mark robot path as free space

        exp_i_values = np.floor((points_explored[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        exp_j_values = np.floor((points_explored[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])

        explored_map[exp_i_values, exp_j_values] = 1
        self.explored_map[exp_i_values, exp_j_values] = 1

        diff_ob_ex = explored_map - obstacle_map

        if np.abs(self.eve_angle) < 10 and self.last_action != 4 and self.last_action != 5:
            self.obstacle_map[diff_ob_ex == 1] = 0 # TODO: multy: what is this for?
            
        return obs_i_values, obs_j_values

    
    def get_transform_matrix(self, agent_state):
        """
        transform the habitat-lab space to Open3D space (initial pose in habitat)
        habitat-lab space need to rotate camera from x,y,z to  x, -y, -z
        Returns Pose_diff, R_diff change of the agent relative to the initial timestep
        """
        camera_position = agent_state.sensor_states["depth"].position
        camera_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)

        h_camera_matrix = np.eye(4)
        h_camera_matrix[:3, :3] = camera_rotation
        h_camera_matrix[:3, 3] = camera_position

        habitat_camera_self = np.eye(4)
        habitat_camera_self[:3, :3] = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
        habitat_camera_self_aj = np.eye(4)
        habitat_camera_self_aj[:3, :3] = np.array([[0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0]])
        
        R_habitat2open3d = np.eye(4)
        R_habitat2open3d[:3, :3] = self.init_sim_rotation
        R_habitat2open3d[:3, 3] = self.init_sim_position

        camera_pose = habitat_camera_self_aj @ np.linalg.inv(R_habitat2open3d) @ h_camera_matrix
        O_camera_matrix = habitat_camera_self_aj @ np.linalg.inv(R_habitat2open3d) @ h_camera_matrix @ habitat_camera_self


        return O_camera_matrix
    
    def find_nearest_point_cloud(self, point_cloud, target_point):
        # 创建 KDTree
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)

        # 查找离目标点最近的点
        [k, idx, _] = kdtree.search_knn_vector_3d(target_point, 1)
        nearest_point = np.asarray(point_cloud.points)[idx[0]]
        
        return nearest_point
    
    
    def object_map_building(self, point_sum):

        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        mask = (points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                (points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                (points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                (points[:, 2] <= self.origins_real[1] + self.map_real_halfsize)
                
        points_filtered = points[mask]
        colors_filtered = colors[mask]
 
        # 计算二维地图的索引ww
        i_values = np.floor((points_filtered[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        j_values = np.floor((points_filtered[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        
        return i_values, j_values

     
    def detect_plane_points_cell(self, point_sum, camera_position):
        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        # mask = (points[:, 1] <= camera_position[1] + 0.5 )
        mask = (points[:, 1] <= camera_position[1] -0.7) & (points[:, 1] > camera_position[1] - 1.0)

        points_filtered = points[mask]
        colors_filtered = colors[mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_filtered)
        pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
        
        self.new_floor_pcd_num = len(pcd.points)

        return pcd
    

    def remove_full_points_cell(self, point_sum, camera_position):
        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        # mask = (points[:, 1] <= camera_position[1] + 0.5 )
        mask = (points[:, 1] <= camera_position[1] + 0.5 )

        points_filtered = points[mask]
        colors_filtered = colors[mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_filtered)
        pcd.colors = o3d.utility.Vector3dVector(colors_filtered)

        return pcd
        
    def remove_diff_floor_points_cell(self, point_sum, camera_position):
        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        mask = (points[:, 1] <= camera_position[1]) & (points[:, 1] > camera_position[1] - 1.3)

        points_filtered = points[mask]
        colors_filtered = colors[mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_filtered)
        pcd.colors = o3d.utility.Vector3dVector(colors_filtered)

        return pcd


    def plane_segmentation_xy(self, pcd, distance_threshold=0.02, ransac_n=10, num_iterations=100):

        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                ransac_n=ransac_n,
                                                num_iterations=num_iterations)
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return inlier_cloud


    def get_frontier_boundaries(self, frontier_loc, frontier_sizes, map_sizes):
        loc_r, loc_c = frontier_loc
        local_w, local_h = frontier_sizes
        full_w, full_h = map_sizes

        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
 
        return [int(gx1), int(gx2), int(gy1), int(gy2)]
    
    def save_rgbd_image(self, rgb_image, depth):
        vis_image_rgb = np.ones((480, 1280, 3)).astype(np.uint8) * 255
        vis_image_rgb[0:480, 0:640] = rgb_image 
        # Normalize the depth values to the range 0-255
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply a colormap (e.g., COLORMAP_JET)
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        vis_image_rgb[0:480, 640:1280] = depth_color
        
        ep_dir = '{}episodes/{}/rgbd/'.format(
            self.dump_dir, self.episode_label)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        fn = ep_dir + 'Vis-{}.png'.format(self.l_step)
        cv2.imwrite(fn, vis_image_rgb)
        
    def save_similarity_map(self, map):
        depth_normalized = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply a colormap (e.g., COLORMAP_JET)
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        ep_dir = '{}episodes/{}/similarity/'.format(
            self.dump_dir, self.episode_label)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        fn = ep_dir + 'Vis-simi-{}.png'.format(self.l_step)
        cv2.imwrite(fn, depth_color)
    
    def _visualize(self, map_pred, exp_pred, map_edge, goal_map, text_queries):

        # start_x, start_y, start_o = pose

        sem_map = np.zeros((self.local_w, self.local_h))

        # no_cat_mask = sem_map == 20
        map_mask = map_pred == 1
        exp_mask = exp_pred == 1
        vis_mask = self.visited_vis == 1
        edge_mask = map_edge > 0

        # sem_map[no_cat_mask] = 0
        # m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[exp_mask] = 2

        # m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[map_mask] = 1

        sem_map[vis_mask] = 3
        sem_map[edge_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal_map, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
        # drawing circle around the goal
        if np.sum(goal_map) == 1:
            f_pos = np.argwhere(goal_map == 1)
            # fmb = get_frontier_boundaries((f_pos[0][0], f_pos[0][1]))
            # goal_fmb = skimage.draw.circle_perimeter(int((fmb[0]+fmb[1])/2), int((fmb[2]+fmb[3])/2), 23)
            goal_fmb = skimage.draw.circle_perimeter(f_pos[0][0], f_pos[0][1], int(self.map_size/16 -1))
            goal_fmb[0][goal_fmb[0] > self.map_size-1] = self.map_size-1
            goal_fmb[1][goal_fmb[1] > self.map_size-1] = self.map_size-1
            goal_fmb[0][goal_fmb[0] < 0] = 0
            goal_fmb[1][goal_fmb[1] < 0] = 0
            # goal_fmb[goal_fmb < 0] =0
            goal_mask[goal_fmb[0], goal_fmb[1]] = 1
            sem_map[goal_mask] = 4


        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8)) # original map, x down, z right 
        sem_map_vis = sem_map_vis.convert("RGB") 
        sem_map_vis = np.flipud(sem_map_vis) # flip upsdie down x up, z right 

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        vis_image = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)

       
        def get_contour_points(pos, origin, size=20):
            x, y, o = pos
            pt1 = (int(x) + origin[0],
                int(y) + origin[1])
            pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
                int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
            pt3 = (int(x + size * np.cos(o)) + origin[0],
                int(y + size * np.sin(o)) + origin[1])
            pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
                int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

            return np.array([pt1, pt2, pt3, pt4])

        pos = [self.last_grid_pose[1], int(self.map_size)-self.last_grid_pose[0], np.deg2rad(self.relative_angle)]
        agent_arrow = get_contour_points(pos, origin=(0, 0), size=10)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 2
        text = "Find {} ".format(text_queries)
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (480 - textsize[0]) // 2 + 30
        textY = (50 + textsize[1]) // 2
        vis_image_show = cv2.putText(vis_image, text, (textX, textY),
                                font, fontScale, color, thickness,
                                cv2.LINE_AA)

        
        vis_image_rgb = init_vis_image(text_queries, self.last_action)
        vis_image_rgb[50:530, 15:655] = self.annotated_image 
        vis_image_rgb[50:530, 670:1150] = vis_image
        # TODO: add aditional stair map for visual, stair map is created in stiar_climbing_map
        vis_image_rgb[580:530*2, 15:495] = self.global_height_image
        vis_image_rgb[580:530*2, 670:1150] = self.global_height_gradient_img
        # vis_image_rgb[580:530*2, 15:495] = self.height_image
        # vis_image_rgb[580:530*2, 670:1150] = self.height_gradient_img
        vis_image_rgb = add_img(vis_image_rgb, self.traversable_occupancy_map,  (1200, 50))
        vis_image_rgb = add_img(vis_image_rgb, self.global_bev_rgb_map, (1700, 50))
        vis_image_rgb = add_img(vis_image_rgb, self.stair_map_img, (1200, 540))
        vis_image_rgb = add_img(vis_image_rgb, self.filtered_stair_map_img, (1700, 540))
        vis_image_rgb = add_img(vis_image_rgb, self.connected_img, (1700, 540+600))
        
        # if self.another_floor:
        vis_image_rgb = add_text(vis_image_rgb, f"[Num Floor Points] {self.new_floor_pcd_num}", (850, 1100), (0, 0, 0))
        vis_image_rgb = add_text(vis_image_rgb, f"[Height Diff] {self.height_diff:.2f}", (1300, 1100), (0, 0, 0))
        vis_image_rgb = add_text(vis_image_rgb, f"[Climb Steps] {self.climbing_step}", (1650, 1100), (0, 0, 0))
        
        
        which_stair = None
        if self.upstair_flag:
            which_stair = "up"
        elif self.downstair_flag:
            which_stair = "down"
        
        if not self.found_stair_goal:
            vis_image_rgb = add_text(vis_image_rgb, "[Agent State] Finding stair entrance " + which_stair, (50, 1100), (0, 0, 0))
        elif self.another_floor and not self.start_climbing:
            vis_image_rgb = add_text(vis_image_rgb, "[Agent State] Going stair entrance " + which_stair, (50, 1100), (255, 0, 0))
        elif self.another_floor and self.start_climbing:
            vis_image_rgb = add_text(vis_image_rgb, "[Agent State] Climbing stair " + which_stair, (50, 1100), (0, 0, 255))
        else:
            vis_image_rgb = add_text(vis_image_rgb, "[Agent State] Exploring", (50, 1100), (0, 255, 0))
            
        ## save for video
        self.vis_frames.append(cv2.cvtColor(vis_image_rgb, cv2.COLOR_BGR2RGB))
        
        if self.args.print_images:
            ep_dir = '{}episodes/{}/eps_{}/'.format(
                self.dump_dir, self.args.rank, self.episode_label)
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)
            fn = ep_dir + 'Vis-{}.png'.format(self.l_step)
            cv2.imwrite(fn, vis_image_rgb)
            cv2.imwrite(self.dump_dir + "current_state.png", vis_image_rgb)

        # cv2.imshow("episode_n {}".format(self.episode_label), vis_image_show)
        # cv2.waitKey(1)

        return vis_image_show

    def stair_climbing_policy(self, stair_candidate_list, camera_position):
        '''
        
        NOTE: in map img, x up, z right, y out
        NOTE: there is a move_map_and_pose function that shifts the grid map origin
            when robot is moving to the edge of the map
        '''
        #################################################################
        ##### select valid points within height range (same floor) ######
        #################################################################
        print("Camera pose: ", camera_position)
        all_points = np.asarray(self.point_sum.points) 
        in_map_mask = (all_points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                    (all_points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                    (all_points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                    (all_points[:, 2] <= self.origins_real[1] + self.map_real_halfsize )
        all_points = all_points[in_map_mask] # in flipped image, x up, z right, y out
        if self.upstair_flag:
            height_min = camera_position[1] - 1.5 # 0.88 is habitat camera height, min value is 1 meters below the camera
            height_max = camera_position[1] + 0.3 # heuristically add a bit more height 
        if self.downstair_flag:
            height_min = camera_position[1] - 2 # 0.88 is habitat camera height, min value is 1 meters below the camera
            height_max = camera_position[1] # we need to see anything at the camera height as obstacle
        height_mask = (all_points[:, 1] >= height_min) & (all_points[:, 1] <= height_max)
        all_points = all_points[height_mask] # filter out points that does not belong to the current floor TODO: debug
        
        ##################################################
        ##### local region mask on grid for x and y ######
        ##################################################
        # create 2m range mask
        # x_min, x_max = camera_position[0] - LOCAL_REIGION_RANGE, camera_position[0] + LOCAL_REIGION_RANGE
        # z_min, z_max = camera_position[2] - LOCAL_REIGION_RANGE, camera_position[2] + LOCAL_REIGION_RANGE
        # i_min = np.floor(x_min * 100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        # i_max = np.floor(x_max * 100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        # j_min = np.floor(z_min * 100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        # j_max = np.floor(z_max * 100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        i_min = self.current_grid_pose[0] - LOCAL_GRID_RANGE
        i_max = self.current_grid_pose[0] + LOCAL_GRID_RANGE
        j_min = self.current_grid_pose[1] - LOCAL_GRID_RANGE
        j_max = self.current_grid_pose[1] + LOCAL_GRID_RANGE
        local_map_mask = np.zeros((self.local_w, self.local_h)).astype(bool)
        local_map_mask[i_min:i_max, j_min:j_max] = True # this is a mask to indicate the local region in the grid
        local_map_mask_goal_selection = np.zeros((self.local_w, self.local_h)).astype(bool)
        local_map_mask_goal_selection[i_min+LOCAL_GRID_RANGE//2:i_max-LOCAL_GRID_RANGE//2, j_min+LOCAL_GRID_RANGE//2:j_max-LOCAL_GRID_RANGE//2] = True # this is a mask to indicate the local region in the grid specifically for selecting stiar goal
        
        ###############################
        ##### global bev rgb map ######
        ###############################
        all_colors = np.asarray(self.point_sum.colors)
        all_colors = all_colors[in_map_mask]
        all_colors = all_colors[height_mask]
        self.global_bev_rgb_map = self.get_bev_rgb_map(all_points, all_colors)
        
        ##############################################
        ##### global height and height gradient ######
        ##############################################
        global_height_map, global_gradient_map = self.create_global_height_map(all_points, height_min)
        gradient_mask = global_gradient_map < GRADIENT_THRESHOLD # TODO: need to tune
        
        ####################################################
        ##### calculate new obstacle and explored map ######
        ####################################################
        # old method
        obstacle_mask = self.obstacle_map == 1
        # filling in small holes 
        kernel = np.ones((5, 5), dtype=np.uint8)
        # connected_traversable_mask = cv2.inRange(connected_traversable_mask,0.1,1)
        closed_explored_map = cv2.morphologyEx(self.explored_map.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        self.obstacle_map_multy, self.explored_map_multy = self.get_new_obstacle_explore_map(all_points, global_height_map, global_gradient_map, camera_position)
        explored_map = self.explored_map_multy
        obstacle_map = self.obstacle_map_multy
        
        ###############################################
        ##### get traversable considering stairs ######
        ###############################################
        traversable_mask = (explored_map == 1) & gradient_mask & (obstacle_map != 1) # get pixel where robot can stand
        connected_traversable_mask = self.get_connected_traversable_mask(traversable_mask) # only connected regions are traversable
        
        #############################################
        ##### semantic method to get stiar map ######
        #############################################
        
        # build a 2D map that indiates occupancy of stairs using semantics
        stair_map = self.create_stair_map(stair_candidate_list)
        # filter out invalid stairs by connectivity and gradient
        connected_traversable_stair_mask = connected_traversable_mask & stair_map.astype(bool) & (global_gradient_map > 0)
        
        ####################################
        ##### selecting goal position ######
        ####################################
        goal_ind = self.stair_goal_selection(connected_traversable_mask, connected_traversable_stair_mask, global_height_map, local_map_mask_goal_selection, camera_position, height_min)
        print("Goal index: ", goal_ind)
        
        ##########################################
        ##### detect frontier using new map ######
        ##########################################
        target_score, target_edge_map, target_point_list = detect_frontier(explored_map, obstacle_map, self.current_grid_pose, threshold_point=8)
        
        ################################################
        ##### visualize important maps and thigns ######
        ################################################
        
        # visualize global height map and gradient
        self.global_height_image = get_pyplot_as_numpy_img(global_height_map)
        self.global_height_gradient_img = get_pyplot_as_numpy_img(global_gradient_map, cmap='hot')
        
        # visualize connected_traversable_mask
        self.connected_img = get_pyplot_as_numpy_img(connected_traversable_mask.astype(np.uint8), cmap='binary') # visualize connected traversable mask
        
        # visualize stair_mask
        self.stair_map_img = get_pyplot_as_numpy_img(stair_map.astype(np.uint8), cmap='binary') # visualize stair map
        self.filtered_stair_map_img = get_pyplot_as_numpy_img(connected_traversable_stair_mask.astype(np.uint8), cmap='binary') # visualize filtered stair map
        
        # create new map: traversable occupancy map
        sem_map = np.zeros((self.local_w, self.local_h))
        
        visited_mask = self.visited_vis == 1
        sem_map[connected_traversable_mask] = 2 # use stiar to overwrite obstacle as free space
        sem_map[obstacle_map == 1] = 1
        sem_map[visited_mask] = 3
        sem_map[target_edge_map > 0] = 3
        
        new_obstacle_map = (sem_map == 1).astype(np.uint8)
        
        # indicating goal on goal map
        goal_map = np.zeros((self.local_w, self.local_h))
        if goal_ind is not None:
            #################################
            ##### visualize stair goal ######
            #################################
            goal_map[goal_ind[0], goal_ind[1]] = 1
            
            # selem = skimage.morphology.disk(4)
            selem = skimage.morphology.disk(1)
            goal_mat = 1 - skimage.morphology.binary_dilation(
            goal_map, selem) != True
            goal_mask = goal_mat == 1
            
            # draw larger circle to show goal
            goal_fmb = skimage.draw.circle_perimeter(goal_ind[0], goal_ind[1], int(self.map_size/16 -1))
            goal_fmb[0][goal_fmb[0] > self.map_size-1] = self.map_size-1
            goal_fmb[1][goal_fmb[1] > self.map_size-1] = self.map_size-1
            goal_fmb[0][goal_fmb[0] < 0] = 0
            goal_fmb[1][goal_fmb[1] < 0] = 0
            # goal_fmb[goal_fmb < 0] =0
            goal_mask[goal_fmb[0], goal_fmb[1]] = 1
            sem_map[goal_mask] = 4
        
        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8)) # original map, x down, z right 
        sem_map_vis = sem_map_vis.convert("RGB") 
        sem_map_vis = np.array(sem_map_vis)
        
        ##### only visual local map #####
        if self.start_climbing:
            sem_map_vis[~local_map_mask] = (0,0,0) # set non-local region to black, this is for visualization only
        
        sem_map_vis = np.flipud(sem_map_vis) # flip upsdie down x up, z right 
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        vis_image = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.traversable_occupancy_map = vis_image
        
        self.last_goal_ind = goal_ind
        
        return goal_ind, new_obstacle_map, goal_map
    
    # def generate_local_mask():
        
        
    def create_global_height_map(self, point_cloud, height_min):
        '''
        Create a height mask, starting frmo the min height
        
        Input:
            - point_cloud: numpy array of shape (N, 3) where N is the number 
                of points, and each point is (x, y, z)
            - height_min: the minimum height to consider for the height map 
                (e.g., camera height - 1.5)
        Output:
            - global_height_map: a 2D numpy array representing the height map, 
                where each cell corresponds to the height at that grid location
            - global_gradient_map: the gradient of the height map, 
                which can be used for further analysis or visualization (all 
                positive value)
        '''
        def generate_kernels():
            """
            Generate 4 different 3x3 kernels where each has one 1 and one -1 at opposite ends.

            Returns:
                np.ndarray: An array of shape (4, 3, 3) containing the four kernels.
            """
            kernels = np.zeros((4, 3, 3))  # Initialize 4 kernels with zeros
            
            positions = [((0, 0), (2, 2)),  # Top-left to bottom-right
                        ((0, 2), (2, 0)),  # Top-right to bottom-left
                        ((0, 1), (2, 1)),  # Top-center to bottom-center
                        ((1, 0), (1, 2))]  # Middle-left to middle-right

            for i, (pos1, pos2) in enumerate(positions):
                kernels[i, pos1[0], pos1[1]] = 1   # Set 1 at one end
                kernels[i, pos2[0], pos2[1]] = -1  # Set -1 at the opposite end
            
            return kernels
        # create global height map
        global_height_map = np.ones((self.local_w, self.local_h)) * (height_min)
        sorted_indices = np.argsort(point_cloud[:, 1]) # ascending order
        sorted_points = point_cloud[sorted_indices]
        i_values = np.floor((sorted_points[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        j_values = np.floor((sorted_points[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        global_height_map[i_values, j_values] = sorted_points[:, 1]
        
        # filling in holes in height map
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        # global_height_map = cv2.dilate(global_height_map, kernel)
        global_height_map = cv2.morphologyEx(global_height_map, cv2.MORPH_CLOSE, np.ones((5,5)))

        # hand desgined gradient kernel that has physical unit
        kernels = generate_kernels()
        filtered_img = []
        for kernel in kernels:
            img = cv2.filter2D(src=global_height_map, ddepth=-1, kernel=kernel)
            filtered_img.append(img)
        filtered_img = np.stack(filtered_img)  # Shape: (4, height, width)
        global_gradient = np.max(np.abs(filtered_img), axis=0)  # Take the max across the 4 kernels to get the final gradient map
        
        # cv gradient 
        # global_gradient = cv2.morphologyEx(global_height_map, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
        # calculate gradient as image
        # global_gradient = np.abs(cv2.Laplacian(global_height_map_norm,cv2.CV_64F))
        
        # filling in holes in gradient map, becuase for each edge, the gradient may genreates two highlighted edges
        kernel = np.ones((3, 3), dtype=np.uint8)
        global_gradient = cv2.morphologyEx(global_gradient, cv2.MORPH_CLOSE, kernel)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        # global_gradient = cv2.dilate(global_gradient, kernel)
        
        return global_height_map, global_gradient
    
    def get_new_obstacle_explore_map(self, all_points, global_height_map, global_gradient_map, camera_position):
        explored_map = np.zeros((self.local_w, self.local_h))
        obstacle_map = np.zeros((self.local_w, self.local_h))
        
        # explored area
        explored_i_values = np.floor((all_points[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        explored_j_values = np.floor((all_points[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        explored_map[explored_i_values, explored_j_values] = 1
        # remove small holes in side the region 
        kernel = np.ones((5, 5), dtype=np.uint8)
        explored_map = cv2.morphologyEx(explored_map.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        # explored_map = cv2.dilate(explored_map.astype(np.uint8), np.ones((3,3))) # expand the explored region a little bit 
        
        # obstacle map = explored region but gradient is large, height is also large
        floor_height = camera_position[1] - 1 # assume the floor height is 1 meter below the camera
        obstacle_map = (explored_map == 1) & (global_gradient_map > GRADIENT_THRESHOLD) & (global_height_map - floor_height > OBSTACLE_HEIGHT)
        obstacle_map = obstacle_map.astype(np.uint8)
        # dilate obstacle area for safe travel 
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        # obstacle_map = cv2.dilate(obstacle_map, kernel)
        # mark robot path as free space
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        obstacle_map[cv2.dilate(self.visited_vis, kernel) == 1] = 0 
        
        return obstacle_map, explored_map
    
    def get_connected_traversable_mask(self, traversable_mask):
        # use connectivity to check if the robot can go to the free space
        connected_traversable_mask = np.zeros((self.local_w, self.local_h)).astype(bool)
        thresh = traversable_mask.astype(np.uint8)
        # use erode to break losely connected regions
        thresh_dialated = cv2.erode(thresh, np.ones((5,5), np.uint8), iterations=1)
        print("After erode size: ", np.sum(thresh), np.sum(thresh_dialated))
        
        # find conected components
        numLabels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(thresh_dialated, 8, cv2.CV_32S)
        for label in range(1, numLabels+1):
            mask = labelImage == label
            if (mask[self.current_grid_pose[0]-1:self.current_grid_pose[0]+2, self.current_grid_pose[1]-1:self.current_grid_pose[1]+2]).any():
                # the robot is in the connected component in 3x3 kernel
                connected_traversable_mask = traversable_mask & mask
                break
            # else if find not connected path, we can't determine a goal point
        
        # dilate to get back the eroded region
        connected_traversable_mask = cv2.dilate(connected_traversable_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) 
    
        return connected_traversable_mask.astype(bool)
    
    def create_stair_map(self, stair_candidate_list):
        print("Number of stair candidates:", len(stair_candidate_list))
        stair_map = np.zeros((self.local_w, self.local_h))
        for stair_pcd in stair_candidate_list:
            stair_points = np.asarray(stair_pcd.points)
            in_map_mask_stair = (stair_points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                    (stair_points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                    (stair_points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                    (stair_points[:, 2] <= self.origins_real[1] + self.map_real_halfsize )
            stair_points = stair_points[in_map_mask_stair]
            
            
            # calc 2D map index
            stair_i_values = np.floor((stair_points[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
            stiar_j_values = np.floor((stair_points[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
            
            # mark stair occupancy 
            stair_map[stair_i_values, stiar_j_values] = 1    
            # stair_mask = stair_map.astype(bool)
        
        # expand search area
        kernel = np.ones((5, 5), dtype=np.uint8)
        stair_map = cv2.morphologyEx(stair_map.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return stair_map
    
    def stair_goal_selection(self, connected_traversable_mask, connected_traversable_stair_mask, 
                             global_height_map, local_map_mask, camera_position, height_min):
        goal_ind = None
        if connected_traversable_mask.any():
            if self.another_floor and self.last_goal_ind is not None:
                # check if you arrived at the stair entrance:
                distance_to_goal = np.linalg.norm(np.array(self.last_goal_ind) - np.array(self.current_grid_pose))
                if distance_to_goal < 5: # distance is smaller than 5 grid (25cm)
                    self.start_climbing = True
            
            # Generates two 2D arrays: row indices and col indices
            indices = np.indices((global_height_map.shape[0], global_height_map.shape[1]))  
            map_indices =  np.stack(indices, axis=-1)   # Stack them along the last dimension, eg. map_indices[2,5] = [2,5]
            
            if self.upstair_flag:
                height_mask = global_height_map > camera_position[1] - ACTUAL_CAMERA_HEIGHT + 0.1
            if self.downstair_flag:
                height_mask = global_height_map < camera_position[1] - ACTUAL_CAMERA_HEIGHT

            # traversible_height_map = global_height_map.copy()
            if self.start_climbing:
                valid_local_mask = local_map_mask & connected_traversable_mask & height_mask # TODO: slice into local map
                valid_indices = map_indices[valid_local_mask]
                valid_heights = global_height_map[valid_local_mask] # get the height values at valid indices
                sorted_ind = np.argsort(valid_heights) # ascending order of height values at valid indices
                
                if self.upstair_flag:
                    sorted_ind = sorted_ind[::-1] # reverse the order if going upstairs, so we can select the highest point first
                # If going down stairs, find the lowest point 
                
            else:
                # go to the most likely stair entrance
                valid_mask = connected_traversable_stair_mask & height_mask
                valid_indices = map_indices[valid_mask]
                valid_heights = global_height_map[valid_mask] # get the height values at valid indices
                sorted_ind = np.argsort(valid_heights) # ascending order of height values at valid indices
                
                if self.downstair_flag:
                    # for down stair, we want to go to the highest stair entrance point first
                    sorted_ind = sorted_ind[::-1]
                    
            # select the goal that for nearby (5,5) kernel all are traversible and in local region
            for i in sorted_ind:
                map_ind = valid_indices[i] # get the index of the valid height
                height_value = valid_heights[i]
                
                values = connected_traversable_mask[map_ind[0]-2:map_ind[0]+3, map_ind[1]-2:map_ind[1]+3]
                if values.all():
                    goal_ind = map_ind
                    break
                
        self.found_stair_goal = True if goal_ind is not None else False
        
        return goal_ind

    def get_bev_rgb_map(self, all_points, all_colors):
        global_rgb_map = np.zeros((self.local_w, self.local_h, 3))
        
        sorted_indices = np.argsort(all_points[:, 1]) # ascending order of height
        sorted_points = all_points[sorted_indices]
        sorted_colors = all_colors[sorted_indices]
        i_values = np.floor((sorted_points[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        j_values = np.floor((sorted_points[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        global_rgb_map[i_values, j_values] = sorted_colors # higher point color will replace the lower ones
        
        global_rgb_map = (global_rgb_map * 255).astype(np.uint8)
        global_rgb_map = global_rgb_map[::-1] # flip the image to match the habitat coordinate system
        global_rgb_map = cv2.cvtColor(global_rgb_map, cv2.COLOR_RGB2BGR)
        return global_rgb_map