#!/usr/bin/env python3
import math
import time
import os

import torch
import open3d as o3d
from multiprocessing import Process, Queue

from habitat.core.simulator import Observations

from PIL import Image
import yaml
import quaternion
from yacs.config import CfgNode as CN
import logging
from io import BytesIO
import numpy as np
import cv2
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
from utils.compute_similarities import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
    color_by_clip_sim,
    cal_clip_sim
)
from constants import category_to_id
from utils.vis import init_vis_image, draw_line, vis_result_fast
from utils.explored_map_utils import (
    build_full_scene_pcd,
    detect_frontier,
)


import ast
import base64
from PIL import Image, ImageDraw, ImageFont
from utils.equ_ranking import Equilibrium_Ranking
# from agents.system_prompt import Instruction_system_prompt
from agents.objnav_agent import ObjectNav_Agent
    


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


class VLObjectNav_Agent(ObjectNav_Agent):
    def __init__(self, args, follower=None, logger=None) -> None:
        self.args = args
        if follower != None:
            self.follower = follower

        if logger != None:
            self.logger = logger
        super().__init__(args=args, follower=follower)
        # ------------------------------------------------------------------
            
        # ------------------------------------------------------------------
        ##### Initialize langauge navigation
        # ------------------------------------------------------------------     
        self.candidate_num = 0
        self.candidate_objects = []
        self.image_list = []
        

    def reset(self) -> None:
        super().reset()
        self.candidate_num = 0
        self.candidate_objects = []
        self.image_list = []
        

        # ------------------------------------------------------------------

    def act(self, observations: Observations, agent_state, send_queue= Queue(), receive_queue= Queue()):
        time_step_info = 'Mapping time (s): \n'

        preprocess_s_time = time.time()
        # ------------------------------------------------------------------
        ##### At first step, get the object name and init the visualization
        # ------------------------------------------------------------------
        if self.l_step == 0:
            self.init_sim_position = agent_state.sensor_states["depth"].position
            self.init_agent_position = agent_state.position
            self.init_sim_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)

            self.text_queries = observations["instruction"]['text']
            # self.object_category = category_to_id[observations['objectgoal'][0]]

            if self.args.vln_mode == 'clip':
                self.target_data = self.text_queries
                self.classes = [self.text_queries]
            elif 'vlm' in self.args.vln_mode :
                self.all_objects = []
                self.equ_ranking = Equilibrium_Ranking(self.text_queries, self.logger)
                response_message = self.equ_ranking.response_message

                ground_json = ast.literal_eval(response_message)
                self.target_data = ground_json["target"]
                self.all_objects.append(self.target_data)
                if "landmark" in ground_json:
                    for landmark in ground_json["landmark"]:
                        self.all_objects.append(landmark)
                        
                if self.args.detector == "dino":
                    self.classes = self.all_objects
                        
                print("self.all_objects: ", self.all_objects)

        # ------------------------------------------------------------------
        ##### Preprocess the observation
        # ------------------------------------------------------------------
            
        image_rgb = observations['rgb']
        depth = observations['depth']
        image = transform_rgb_bgr(image_rgb) 
        image_pil = Image.fromarray(image_rgb)
        self.annotated_image = image
        if 'vlm' in self.args.vln_mode :
            self.image_list.append(image_rgb)
            
        get_results, detections = self.obj_det_seg.detect(image, image_rgb, self.classes) 
        
        # if 'vlm' in self.args.vln_mode :
        #     self.annotated_image  = vis_result_fast(image, detections, self.classes)
        #     self.image_list.append(transform_rgb_bgr(self.annotated_image))
        
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
        )

        
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
        
        
        # ------------------------------------------------------------------
        ##### 2D Obstacle Map
        # ------------------------------------------------------------------
        f_map_time = time.time()
        
        local_grid_pose = [camera_position[0]*100/self.args.map_resolution + int(self.origins_grid[0]), 
                      camera_position[2]*100/self.args.map_resolution + int(self.origins_grid[1])]
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
        self.point_sum += self.remove_full_points_cell(full_scene_pcd, camera_position)
        
        obs_i_values, obs_j_values = self.update_map(full_scene_pcd, camera_position, self.args.map_height_cm / 100.0 /2.0)


        target_score, target_edge_map, target_point_list = detect_frontier(self.explored_map, self.obstacle_map, self.current_grid_pose, threshold_point=8)
        
         
        v_map_time = time.time()
        # print('voxel map: %.3f秒'%(v_map_time - f_map_time)) 
        
        vis_image = None
        if self.args.print_images or self.args.visualize:
            self.annotated_image  = vis_result_fast(image, detections, self.classes)
            vis_image = self._visualize(self.obstacle_map, self.explored_map, target_edge_map, self.goal_map, self.target_data)
            
            self.save_rgbd_image(image, depth)
            

        
        # ------------------------------------------------------------------
        
        # ------------------------------------------------------------------
        ##### Calculate similarities for objects and landmarks
        # ------------------------------------------------------------------
        if not send_queue.empty():
            text_input, self.is_running = send_queue.get()
            if text_input is not None:
                self.target_data = text_input
            # print("self.text_queries: ", self.text_queries)

        clip_time = time.time()
        
        total_candidate_objects = []
        candidate_target = []
      
        clip_candidate_objects = []
        candidate_id = []
        clip_candidate_objects_id = []
        similarity_threshold = 0.26
        det_threshold = 0.85
        
        if "chair" in self.target_data:
            similarity_threshold = 0.25


        if len(self.objects) > 0:
            self.objects, target_similarities = color_by_clip_sim("looks like a " + self.target_data, 
                                                        self.objects, 
                                                        self.clip_model, 
                                                        self.clip_tokenizer)
            
            target_similarities = target_similarities.cpu().numpy()
            candidate_target = [self.objects[i] for i in range(len(self.objects)) 
                                if (target_similarities[i] > similarity_threshold and \
                                    self.objects[i]['num_detections'] > 2) \
                                    or \
                                    max(self.objects[i]['conf']) > det_threshold]
            
            candidate_id = [i for i in range(len(self.objects)) 
                                if (target_similarities[i] > similarity_threshold and \
                                    self.objects[i]['num_detections'] > 2) \
                                    or \
                                    max(self.objects[i]['conf']) > det_threshold]
            
            clip_candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
                                if (target_similarities[i] > similarity_threshold - 0.02 and \
                                    self.objects[i]['num_detections'] > 2 and self.objects[i]['num_detections'] < 20) ]

            clip_candidate_objects_id = [i for i in range(len(self.objects)) 
                                if (target_similarities[i] > similarity_threshold - 0.02 and \
                                    self.objects[i]['num_detections'] > 2 and self.objects[i]['num_detections'] < 20) ]

            
            # if len(candidate_target) > 0:
            #     print("find targets: " + self.target_data, len(candidate_target), " ", max(target_similarities))
            total_candidate_objects.extend(candidate_target)
            # print("max_sim target: ", max(target_similarities.cpu().numpy()))
            
            if self.args.vln_mode == "clip":
                self.candidate_objects = candidate_target
        
            
            if len(self.candidate_objects) == 0 and len(candidate_target) > 0 and (self.l_step+1) % cfg.filter_interval == 0 and len(total_candidate_objects) != self.candidate_num and 'vlm' in self.args.vln_mode :
                
                candidate_image_list = []
                for id in candidate_id:
                    candidate_image_list.append(np.asarray(self.image_list[self.objects[id]['image_idx'][0]]))
                receive_queue.put([candidate_image_list, 
                                depth, 
                                [] , 
                                self.objects.to_serializable(), 
                                np.asarray(self.point_sum.points), 
                                np.asarray(self.point_sum.colors), 
                                self.Open3D_traj,
                                self.episode_n,
                                [],
                                [],
                                [0,0,0],
                                time_step_info,
                                candidate_id]
                                ) 
                
                base64_image_list = []
                for id in candidate_id:
                    dump_dir = "{}/{}/episodes".format(self.args.dump_location, self.args.exp_name)
                    image_filename = '{}/{}/Vis-{}.jpeg'.format(dump_dir, self.episode_n, id)
                    while True:
                        if os.path.exists(image_filename) and os.path.getsize(image_filename) > 0:
                            try:
                                with open(image_filename, "rb") as image_file:
                                    base64_image_list.append(base64.b64encode(image_file.read()).decode("utf-8"))
                                    print("get image " + str(id) + " in " + str(candidate_id))
                                    break
                            except Exception as e:
                                print(f"Failed to save image: {str(e)}")
                        else:
                            time.sleep(0.1)
                            print("Waiting for the generation.")
                                            
                if self.args.vln_mode == "vlm":
                    candidate_index = self.equ_ranking.generator_search(base64_image_list, candidate_id)
                elif self.args.vln_mode == "vlm_rank":
                    candidate_index = self.equ_ranking.generator_search(base64_image_list, candidate_id, 5)
                elif self.args.vln_mode == "vlm_game":
                    candidate_index = self.equ_ranking.equilibrium_search(base64_image_list, candidate_id)

                if candidate_index == -1 : # -1: not sure
                    print("reject all candidates")
                else:
                    self.candidate_objects = [self.objects[candidate_index]]
                    self.logger.info("Found candidate_objects!"+str(candidate_index))
                    print("Found candidate_objects!", candidate_index)  
                    
                self.candidate_num = len(total_candidate_objects)
            
            for i in range(len(self.objects)):
                if target_similarities[i] > 0.24:
                    self.similarity_obj_map[self.object_map_building(self.objects[i]['pcd'])] = target_similarities[i]
                    
             

        image_clip_sim = cal_clip_sim(self.text_queries, current_image_feats, self.clip_model, self.clip_tokenizer)
        self.similarity_img_map[obs_i_values, obs_j_values] = image_clip_sim.cpu().numpy()
        # for i in range(len(fg_detection_list)):
        #     self.similarity_img_map[self.object_map_building(fg_detection_list[i]['pcd'])] = image_clip_sim.cpu().numpy()
        
        
        f_clip_time = time.time()
        # print('calculate clip sim: %.3f秒'%(f_clip_time - clip_time)) 
        time_step_info += 'calculate clip sim time:%.3fs\n'%(f_clip_time - clip_time)
        # ------------------------------------------------------------------
        

        # ------------------------------------------------------------------
        ##### frontier selection
        # ------------------------------------------------------------------
        if len(self.candidate_objects) > 0 : 
            self.goal_map = np.zeros((self.local_w, self.local_h))
            self.goal_map[self.object_map_building(self.candidate_objects[0]['pcd'])] = 1
            self.nearest_point = self.find_nearest_point_cloud(self.candidate_objects[0]['pcd'], camera_position)
            self.found_goal = True
            print("found goal!")
        # elif len(self.objects) > 0 and max(similarities) < similarity_threshold:
        #     self.found_goal = False
        
        similarity_map = np.max(np.stack([self.similarity_obj_map, self.similarity_img_map]), axis=0)
        self.save_similarity_map(similarity_map)
        
        if not self.found_goal:
            stg = None
            if np.sum(self.goal_map) == 1:
                f_pos = np.argwhere(self.goal_map == 1)
                stg = f_pos[0]

            self.goal_map = np.zeros((self.local_w, self.local_h))
            if len(clip_candidate_objects) > 0 :
                
                max_conf = [target_similarities[id] for id in clip_candidate_objects_id]
                max_index = np.argmax(max_conf)
                self.goal_map[self.object_map_building(clip_candidate_objects[max_index]['pcd'])] = 1
                self.nearest_point = self.find_nearest_point_cloud(clip_candidate_objects[max_index]['pcd'], camera_position)
                
                if self.greedy_stop_count > 20:
                    self.objects[clip_candidate_objects_id[max_index]]['num_detections'] = 21
                    
            elif len(target_point_list) > 0:
                self.no_frontiers_count = 0
                simi_max_score = []
                for i in range(len(target_point_list)):
                    fmb = self.get_frontier_boundaries((target_point_list[i][0], 
                                                    target_point_list[i][1]),
                                                    (self.local_w/8, self.local_h/8),
                                                    (self.local_w, self.local_h))
                    # similarity_map = self.similarity_img_map
                    # similarity_map = np.max(np.stack([self.similarity_obj_map, self.similarity_img_map]), axis=0)
                    cropped_sim_map = similarity_map[fmb[0]:fmb[1], fmb[2]:fmb[3]]
                    simi_max_score.append(np.max(cropped_sim_map))

                # print("simi_max_score: ", simi_max_score)
                global_item = 0
                if len(simi_max_score) > 0:
                    # print(simi_max_score)
                    if max(simi_max_score) > 0.24:
                        global_item = simi_max_score.index(max(simi_max_score))
                        
                if np.array_equal(stg, target_point_list[global_item]) and target_score[global_item] < 30:
                    self.curr_frontier_count += 1
                else:
                    self.curr_frontier_count = 0
                    
                if self.curr_frontier_count > 20 or self.replan_count > 20 or self.greedy_stop_count > 20:
                    self.obstacle_map[target_edge_map == global_item+1] = 1
                    self.curr_frontier_count = 0
                    self.replan_count = 0
                    self.greedy_stop_count = 0
                    
                self.goal_map[target_point_list[global_item][0], target_point_list[global_item][1]] = 1
            elif len(self.objects) > 0:
                self.no_frontiers_count += 1
                self.goal_map = np.zeros((self.local_w, self.local_h))
                max_index = np.argmax(target_similarities)
                self.goal_map[self.object_map_building(self.objects[max_index]['pcd'])] = 1
                self.nearest_point = self.find_nearest_point_cloud(self.objects[max_index]['pcd'], camera_position)

                if self.no_frontiers_count > 5 and max(target_similarities) > 0.23 :
                    self.found_goal = True
                                    
            else:
                goal_pose_x = int(np.random.rand() * self.map_size)
                goal_pose_y = int(np.random.rand() * self.map_size)
                self.goal_map[goal_pose_x, goal_pose_y] = 1
        
        
        if np.sum(self.goal_map) == 1:
            f_pos = np.argwhere(self.goal_map == 1)
            stg = f_pos[0]
            x = (stg[0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            y = camera_position[1]
            z = (stg[1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
        elif np.sum(self.goal_map) == 0:
            # stg = self._get_closed_goal(self.obstacle_map, self.last_grid_pose, find_big_connect(self.goal_map))
            self.found_goal == False
            goal_pose_x = int(np.random.rand() * self.map_size)
            goal_pose_y = int(np.random.rand() * self.map_size)
            self.goal_map[goal_pose_x, goal_pose_y] = 1
            f_pos = np.argwhere(self.goal_map == 1)
            stg = f_pos[0]
            
            x = (stg[0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            y = camera_position[1]
            z = (stg[1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
        else: # > 1           
            x = self.nearest_point[0]
            y = self.nearest_point[1]
            z = self.nearest_point[2]

        Open3d_goal_pose = [x, y, z]
        
        Rx = np.array([[0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0]])
        R_habitat2open3d = self.init_sim_rotation @ Rx.T
        self.habitat_goal_pose = np.dot(R_habitat2open3d, Open3d_goal_pose) + self.init_agent_position
        habitat_final_pose = self.habitat_goal_pose.astype(np.float32)

        plan_path = []
        plan_path = self.search_navigable_path(
            habitat_final_pose
        )

        if len(plan_path) > 1:
            plan_path = np.dot(R_habitat2open3d.T, (np.array(plan_path) - self.init_agent_position).T).T
            action = self.greedy_follower_act(plan_path)

        else:
            # plan a path by fmm
            self.stg, self.stop, plan_path = self._get_stg(self.obstacle_map, self.last_grid_pose, np.copy(self.goal_map))
            plan_path = np.array(plan_path) 
            plan_path_x = (plan_path[:, 0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            plan_path_y = plan_path[:, 0] * 0
            plan_path_z = (plan_path[:, 1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0

            plan_path = np.stack((plan_path_x, plan_path_y, plan_path_z), axis=-1)

            action = self.ffm_act()

 
        # action = self.keyboard_act()
        
        
        
        if len(self.objects) > 0:
            time_step_info += 'max similarity: %3fs\n'%(np.max(target_similarities))

        # ------------------------------------------------------------------
        ##### Send to Open3D visualization thread
        # ------------------------------------------------------------------
        if self.args.visualize:
            if cfg.filter_interval > 0 and (self.l_step+1) % cfg.filter_interval == 0:
                self.point_sum = self.point_sum.voxel_down_sample(0.05)
            receive_queue.put([image_rgb, 
                                depth, 
                                self.annotated_image , 
                                self.objects.to_serializable(), 
                                np.asarray(self.point_sum.points), 
                                np.asarray(self.point_sum.colors), 
                                self.Open3D_traj,
                                self.episode_n,
                                plan_path,
                                transform_rgb_bgr(vis_image),
                                Open3d_goal_pose,
                                time_step_info,
                                []]
                                )   

        self.last_action = action
        

        # transfer_time = time.time()
        # time_step_info += 'transfer data time:%.3fs\n'%(transfer_time - dd_map_time)
        # cv2.imshow("episode_n {}".format(self.episode_n), self.annotated_image)
        # cv2.waitKey(1)
        # print(time_step_info)

        return action
    



    def draw_number(self, image, candidate_id):
        # Convert the numpy array to a PIL Image object
        pil_image = Image.fromarray(image)
        

        # add the number on the image
        # Initialize drawing context
        draw = ImageDraw.Draw(pil_image)
        # Define the text to be added and its position
        font_size = 40

        try:
            # Attempt to use a truetype font if available
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            # If the truetype font is not available, use the default PIL font
            font = ImageFont.load_default(font_size)

        # Calculate text size to center it
        # bbox = draw.textbbox((0, 0), text, font=font)
        text_width = 45
        text_height = 45
        padding = 3
        position = (3, 3)  # Adjust position as needed

        # Define the rectangle coordinates
        rect_x0 = position[0] - padding
        rect_y0 = position[1] - padding
        rect_x1 = position[0] + text_width + padding
        rect_y1 = position[1] + text_height + padding

        # Draw the white rectangle
        draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill="white")

        # Add text to image
        draw.text(position, str(candidate_id), fill="red", font=font)

        # Save the image using PIL
        dump_dir = "{}/{}/episodes/{}".format(self.args.dump_location,
                                    self.args.exp_name, self.episode_n)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        image_filename = '{}/Vis-{}.jpeg'.format(dump_dir, candidate_id)
        pil_image.save(image_filename)
        

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return buffered