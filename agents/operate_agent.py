#!/usr/bin/env python3
import argparse
import os
import random
from typing import Dict, Optional
import math
import time

import numba
import numpy as np
import torch
import torch.nn as nn
import gym
from torchvision import transforms
import torch.nn.functional as F

from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import numpy as np

import cv2

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"



class Operate_Agent(Agent):
    def __init__(self, args) -> None:
        self.args = args

   
    def reset(self) -> None:
        
        pass

        

    def mapping(self, observations: Observations):

        # ------------------------------------------------------------------
        ##### Preprocess the observation
        # ------------------------------------------------------------------
        rgb = observations['rgb'].astype(np.uint8)
        depth = observations['depth']
        
        return 1


    
    def act(self):
        # ------------------------------------------------------------------
        ##### Update long-term goal if target object is found
        ##### Otherwise, use the LLM to select the goal
        # ------------------------------------------------------------------
        keystroke = cv2.waitKey(0)
        action = None
        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.STOP
            print("action: FINISH")
        else:
            print("INVALID KEY")

        return action


    def _preprocess_depth(self, depth, min_d, max_d):
        # print("depth origin: ", depth.shape)
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + (max_d-min_d) * depth * 100.0
        # depth = depth*1000.

        return depth

