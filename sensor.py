# -----------------------
# Sensor model (simulated)
# -----------------------

import math
import random
import pygame
import sys
import numpy as np

import main

def simulate_odometry(true_pose_prev, true_pose_new, odom_noise_std=(0.5, 0.05)):
    """
    From two true poses, compute odometry (v, omega) with additive noise.
    Here v in pixels/sec and omega in rad/sec.
    """
    x0, y0, th0 = true_pose_prev
    x1, y1, th1 = true_pose_new
    dt = 0.1
    # compute v and omega from true change
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    v = dist / dt
    dth = main.wrap_to_pi(th1 - th0)
    w = dth / dt
    # add noise
    v_n = v + np.random.normal(0, odom_noise_std[0])
    w_n = w + np.random.normal(0, odom_noise_std[1])
    return v_n, w_n

def simulate_measurements(true_pose, landmarks, max_range=250.0, sensor_noise=(1.0, np.deg2rad(3.0))):
    """
    Simulate range-bearing measurements to landmarks within max_range.
    Returns list of (range, bearing).
    """
    rx, ry, rth = true_pose
    measurements = []
    for lm in landmarks:
        dx = lm[0] - rx
        dy = lm[1] - ry
        r = math.hypot(dx, dy)
        if r <= max_range:
            bearing = main.wrap_to_pi(math.atan2(dy, dx) - rth)
            # add noise
            r_meas = r + np.random.normal(0, sensor_noise[0])
            b_meas = main.wrap_to_pi(bearing + np.random.normal(0, sensor_noise[1]))
            measurements.append((r_meas, b_meas))
    return measurements
