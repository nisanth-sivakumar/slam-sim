"""
EKF-SLAM 2D simulation with pygame visualization.

Author: ChatGPT (example project for resume)
Features:
 - Differential-drive motion with noisy odometry
 - EKF-SLAM: joint robot pose + landmark positions + covariance
 - Landmark initialization and nearest-neighbor association
 - Visualization: true vs estimated pose, landmarks, covariance ellipses

Single-file, easy to run. Good for demonstrating SLAM + Kalman filtering on a resume.

"""

import math
import random
import pygame
import sys
import numpy as np
from collections import deque

import ekf_slam
import world_c
import viewer_c
import sensor

# -----------------------
# Utility functions
# -----------------------

def wrap_to_pi(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotation_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]])

def angle_diff(a, b):
    """Return (a - b) normalized to [-pi, pi]."""
    return wrap_to_pi(a - b)

def ellipse_from_cov(mu2, cov2, nsig=2):
    """Return width, height, angle for ellipse plotting given 2D cov."""
    # eigen-decomposition
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    # axis lengths = 2 * nsig * sqrt(eigenvalues)
    width = 2 * nsig * math.sqrt(max(vals[0], 0))
    height = 2 * nsig * math.sqrt(max(vals[1], 0))
    angle = math.atan2(vecs[1,0], vecs[0,0])
    return width, height, angle

# -----------------------
# Main simulation loop
# -----------------------

def main():
    # World and ekf
    world = world_c.World(size=(900,700), n_landmarks=16, seed=2)
    ekf = ekf_slam.EKF_SLAM(max_landmarks=200)
    # set initial ekf robot estimate near true
    ekf.reset(init_pose=world.get_true_pose() + np.array([5.0, -5.0, np.deg2rad(5.0)]))
    viewer = viewer_c.Viewer(world, ekf)

    running = True
    dt = world.dt
    auto = True
    t = 0.0

    # Controls
    max_v = 60.0  # pixels/sec
    max_w = np.deg2rad(60)
    user_v = 0.0
    user_w = 0.0

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_SPACE:
                    viewer.toggle_auto()
                elif event.key == pygame.K_r:
                    # reset
                    world = world.World(size=(900,700), n_landmarks=16, seed=random.randint(0,10000))
                    ekf = ekf_slam.EKF_SLAM(max_landmarks=200)
                    ekf.reset(init_pose=world.get_true_pose() + np.array([5.0, -5.0, 0.0]))
                    viewer = viewer.Viewer(world, ekf)
                # manual drive toggles
                elif event.key == pygame.K_LEFT:
                    user_w = -max_w * 0.5
                elif event.key == pygame.K_RIGHT:
                    user_w = max_w * 0.5
                elif event.key == pygame.K_UP:
                    user_v = max_v * 0.8
                elif event.key == pygame.K_DOWN:
                    user_v = -max_v * 0.5
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    user_w = 0.0
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    user_v = 0.0

        # Decide control (auto or manual)
        if viewer.auto_drive:
            # Random wanderer: steer to random waypoint occasionally
            if t % 2.0 < 1e-6 or random.random() < 0.03:
                # random velocities
                v_cmd = random.uniform(0.0, max_v*0.9)
                w_cmd = random.uniform(-max_w, max_w)
            # keep previous commands across frames
            if 'v_cmd' not in locals():
                v_cmd = max_v*0.5
                w_cmd = 0.0
            control_true = (v_cmd, w_cmd)
        else:
            control_true = (user_v, user_w)

        # Save previous true pose for odometry simulation
        true_pose_prev = world.get_true_pose()
        world.step_true(*control_true)
        true_pose = world.get_true_pose()

        # Simulate odometry (noisy) from previous to current true pose
        odom = sensor.simulate_odometry(true_pose_prev, true_pose, odom_noise_std=(0.5, 0.02))
        # EKF predict step with odometry as control
        ekf.predict(odom, dt)

        # Simulate measurements from true robot to true landmarks
        measurements = sensor.simulate_measurements(true_pose, world.landmarks, max_range=250.0, sensor_noise=(1.5, np.deg2rad(3.0)))
        # EKF observe
        ekf.observe(measurements)

        # Visualization
        viewer.screen.fill((240,240,240))
        # draw landmarks
        est_lms = ekf.get_landmark_estimates()
        viewer.draw_landmarks(world.landmarks, est_lms)

        # draw true robot
        viewer.draw_robot(true_pose, color=(0,120,0), radius=6, label="true")
        # draw estimated robot
        est_pose = ekf.get_robot_estimate()
        viewer.draw_robot(est_pose, color=(120,0,120), radius=6, label="est")
        # draw covariance ellipse for robot
        viewer.draw_covariance_robot()

        # traces
        viewer.trace_true.append((true_pose[0], true_pose[1]))
        viewer.trace_est.append((est_pose[0], est_pose[1]))
        viewer.draw_traces()

        # Draw measurements as green lines from true robot to true landmark they correspond to (for visible ones)
        for z, lm in zip(measurements, [lm for lm in world.landmarks if math.hypot(lm[0]-true_pose[0], lm[1]-true_pose[1]) <= 250.0]):
            # reconstruct sensor endpoint from true pose + measured (noisy) range/bearing in robot frame
            r_meas, b_meas = z
            ang = wrap_to_pi(b_meas + true_pose[2])
            x_end = true_pose[0] + r_meas * math.cos(ang)
            y_end = true_pose[1] + r_meas * math.sin(ang)
            pygame.draw.line(viewer.screen, (0,200,0), (true_pose[0], true_pose[1]), (x_end, y_end), 1)

        # Text info
        texts = [
            f"EKF-SLAM demo - landmarks: {ekf.n_landmarks}",
            f"Press SPACE to toggle auto drive (now {'ON' if viewer.auto_drive else 'OFF'}), R to reset",
            f"True pose: x={true_pose[0]:.1f}, y={true_pose[1]:.1f}, th={math.degrees(true_pose[2]):.1f} deg",
            f"Est pose: x={est_pose[0]:.1f}, y={est_pose[1]:.1f}, th={math.degrees(est_pose[2]):.1f} deg",
            f"Measurements (visible): {len(measurements)}"
        ]
        viewer.draw_text(texts)

        viewer.update()
        t += dt

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
