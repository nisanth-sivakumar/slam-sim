# -----------------------
# Visualization (pygame)
# -----------------------

import math
import random
import pygame
import sys
import numpy as np
from collections import deque

import main

class Viewer:
    def __init__(self, world, ekf):
        pygame.init()
        self.world = world
        self.ekf = ekf
        self.size = (world.width, world.height)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("EKF-SLAM demo")
        self.font = pygame.font.SysFont("Segoe UI", 14)
        self.clock = pygame.time.Clock()
        self.trace_true = deque(maxlen=1000)
        self.trace_est = deque(maxlen=1000)
        self.auto_drive = True

    def draw_robot(self, pose, color=(0,0,0), radius=8, label=None):
        x, y, th = pose
        pygame.draw.circle(self.screen, color, (int(x), int(y)), radius, 2)
        # heading line
        hx = x + radius * math.cos(th)
        hy = y + radius * math.sin(th)
        pygame.draw.line(self.screen, color, (int(x), int(y)), (int(hx), int(hy)), 2)
        if label:
            txt = self.font.render(label, True, color)
            self.screen.blit(txt, (x+10, y+10))

    def draw_landmarks(self, true_lms, est_lms):
        # draw true landmarks
        for lm in true_lms:
            pygame.draw.circle(self.screen, (0,150,0), (int(lm[0]), int(lm[1])), 5)
        # draw estimated landmarks
        for j, lm in enumerate(est_lms):
            pygame.draw.circle(self.screen, (150,0,150), (int(lm[0]), int(lm[1])), 4, 1)
            # covariance ellipse
            cov = self.ekf.get_landmark_cov(j)
            w,h,ang = main.ellipse_from_cov(lm, cov, nsig=2)
            # pygame doesn't have ellipse rotation, so draw transformed ellipse using polygon approx
            pts = ellipse_points((lm[0], lm[1]), w/2, h/2, ang, segments=16)
            pygame.draw.polygon(self.screen, (150,0,150), pts, 1)

    def draw_traces(self):
        if len(self.trace_true) >= 2:
            pygame.draw.lines(self.screen, (0,120,0), False, list(self.trace_true), 2)
        if len(self.trace_est) >= 2:
            pygame.draw.lines(self.screen, (120,0,120), False, list(self.trace_est), 2)

    def draw_covariance_robot(self):
        # robot covariance is first 3x3 in P; draw ellipse for x,y covariance
        cov = self.ekf.P[:2, :2]
        mu = self.ekf.x[:2]
        w,h,ang = main.ellipse_from_cov(mu, cov, nsig=2)
        pts = ellipse_points((mu[0], mu[1]), w/2, h/2, ang, segments=18)
        pygame.draw.polygon(self.screen, (0,0,200), pts, 1)

    def draw_text(self, texts):
        for i, t in enumerate(texts):
            surf = self.font.render(t, True, (0,0,0))
            self.screen.blit(surf, (10, 10 + 18*i))

    def update(self):
        pygame.display.flip()
        self.clock.tick(30)

    def toggle_auto(self):
        self.auto_drive = not self.auto_drive

def ellipse_points(center, a, b, angle, segments=20):
    """Return polygon points approximating ellipse centered at center, half-axes a,b, rotated by angle."""
    cx, cy = center
    pts = []
    for i in range(segments):
        th = 2*math.pi*i/segments
        x = a * math.cos(th)
        y = b * math.sin(th)
        # rotate
        xr = x*math.cos(angle) - y*math.sin(angle)
        yr = x*math.sin(angle) + y*math.cos(angle)
        pts.append((cx + xr, cy + yr))
    return pts
