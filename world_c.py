# -----------------------
# Simulation env and robot
# -----------------------

import math
import random
import pygame
import sys
import numpy as np
from collections import deque

import main

class World:
    """Simple 2D world with point landmarks."""
    def __init__(self, size=(800, 600), n_landmarks=12, seed=0):
        self.width, self.height = size
        random.seed(seed)
        # Place landmarks in the world, but keep margin
        margin = 60
        self.landmarks = []
        for _ in range(n_landmarks):
            x = random.uniform(margin, self.width - margin)
            y = random.uniform(margin, self.height - margin)
            self.landmarks.append(np.array([x, y]))
        # Robot true state (x, y, theta)
        self.robot_pose = np.array([self.width/4, self.height/2, 0.0])
        self.dt = 0.1

    def step_true(self, v, omega):
        """Move true robot with control (v, omega) and dt."""
        x, y, th = self.robot_pose
        # simple unicycle integration
        th_new = th + omega * self.dt
        x_new = x + v * math.cos(th) * self.dt
        y_new = y + v * math.sin(th) * self.dt
        self.robot_pose = np.array([x_new, y_new, main.wrap_to_pi(th_new)])

    def get_true_pose(self):
        return self.robot_pose.copy()
