# -----------------------
# EKF-SLAM Implementation
# -----------------------

import math
import random
import pygame
import sys
import numpy as np
from collections import deque
import main

class EKF_SLAM:
    """
    EKF-SLAM with state vector:
      x = [rx, ry, rtheta, lm1_x, lm1_y, lm2_x, lm2_y, ...]
    Covariance P is a (3+2M) x (3+2M) matrix.
    Landmark initialization when first observed.
    """
    def __init__(self, max_landmarks=100):
        self.robot_dim = 3
        self.max_landmarks = max_landmarks

        # Initially only robot pose in state
        self.x = np.zeros((self.robot_dim,))  # robot estimate
        self.P = np.eye(self.robot_dim) * 1e-3  # small initial uncertainty in robot pose
        # Keep track of which landmarks are initialized
        self.n_landmarks = 0

        # Process (motion) noise parameters (for control->odom)
        # Represent noise in control space: [v_noise, omega_noise]
        self.motion_noise = np.diag([0.5**2, (np.deg2rad(10))**2])  # fairly large

        # Measurement noise: range and bearing
        self.R = np.diag([2.0**2, np.deg2rad(5.0)**2])  # range variance (pixels), bearing variance (rad)

    def expand_state_for_new_landmark(self, lm_pos):
        """Add a new landmark (2 dims) at lm_pos to the state vector and covariance."""
        # new state size
        old_n = self.x.shape[0]
        new_n = old_n + 2
        x_new = np.zeros((new_n,))
        x_new[:old_n] = self.x
        x_new[old_n:old_n+2] = lm_pos
        # new P
        P_new = np.zeros((new_n, new_n))
        P_new[:old_n, :old_n] = self.P
        # Initialize new landmark covariance large
        large = 1e3
        P_new[old_n:old_n+2, old_n:old_n+2] = np.eye(2) * large
        # Cross-covariances zero
        self.x = x_new
        self.P = P_new
        self.n_landmarks += 1

    def predict(self, control, dt):
        """
        Motion update using unicycle model.
        control = (v, omega) in world units (pixels/sec, rad/sec)
        dt: time step
        """
        v, w = control
        x_r = self.x[:3].copy()
        x, y, th = x_r
        # motion model
        th_new = th + w * dt
        x_new = x + v * math.cos(th) * dt
        y_new = y + v * math.sin(th) * dt

        # Update robot mean
        self.x[0] = x_new
        self.x[1] = y_new
        self.x[2] = main.wrap_to_pi(th_new)

        # Build Jacobians
        n = self.x.shape[0]
        # Fx: maps robot portion
        Fx = np.zeros((3, n))
        Fx[:, :3] = np.eye(3)

        # Motion Jacobian wrt robot pose (G_r)
        G_r = np.eye(3)
        G_r[0,2] = -v * math.sin(th) * dt
        G_r[1,2] = v * math.cos(th) * dt

        # Motion Jacobian wrt control (V)
        V = np.zeros((3,2))
        V[0,0] = math.cos(th) * dt
        V[1,0] = math.sin(th) * dt
        V[2,1] = dt

        # Build full G (n x n) with robot part replaced
        G = np.eye(n)
        G[:3, :3] = G_r

        # Motion noise in state space: Q = V M V^T (in robot pose)
        M = self.motion_noise
        Qr = V @ M @ V.T

        # Expand Qr into full state size
        Q = np.zeros((n, n))
        Q[:3, :3] = Qr

        # Covariance prediction: P = G P G^T + Q
        self.P = G @ self.P @ G.T + Q

    def observe(self, measurements):
        """
        measurements: list of (range, bearing, landmark_id_hint_or_None)
        For this simple sim we don't get landmark ids. We'll do NN association.
        measurement model for landmark i:
          z = [r, phi] where
            r = sqrt((lx - rx)^2 + (ly - ry)^2) + noise
            phi = atan2(ly-r y, lx - rx) - rtheta + noise
        """
        # For each measurement, either associate or initialize
        for z in measurements:
            r_meas, bearing_meas = z  # in robot frame
            # Attempt to associate to an existing landmark
            associated_index = self.data_association(z)
            if associated_index is None:
                # Initialize new landmark in map
                self.initialize_landmark(z)
            else:
                # Update existing landmark via EKF update
                self.update_landmark(z, associated_index)

    def data_association(self, measurement, gate_threshold=5.991):  # 95% ellipse for 2 DOF is ~5.99
        """
        Nearest-neighbor association: for each known landmark, compute expected measurement,
        innovation, Mahalanobis distance; choose lowest if within gate.
        Return landmark index (0..n_landmarks-1) or None.
        """
        if self.n_landmarks == 0:
            return None

        r_meas, bearing_meas = measurement
        best_j = None
        best_maha = None
        for j in range(self.n_landmarks):
            lm_idx = 3 + 2*j
            lx = self.x[lm_idx]
            ly = self.x[lm_idx + 1]
            rx, ry, rth = self.x[:3]
            dx = lx - rx
            dy = ly - ry
            q = dx*dx + dy*dy
            expected_r = math.sqrt(q)
            expected_bearing = main.wrap_to_pi(math.atan2(dy, dx) - rth)
            # Compute innovation
            nu = np.array([r_meas - expected_r, main.wrap_to_pi(bearing_meas - expected_bearing)])
            # Measurement Jacobian H (2 x state_dim)
            n = self.x.shape[0]
            H = np.zeros((2, n))
            sqrt_q = expected_r
            # derivatives
            H[0,0] = -dx / sqrt_q
            H[0,1] = -dy / sqrt_q
            H[0,2] = 0.0
            H[1,0] = dy / q
            H[1,1] = -dx / q
            H[1,2] = -1.0
            H[0, lm_idx] = dx / sqrt_q
            H[0, lm_idx+1] = dy / sqrt_q
            H[1, lm_idx] = -dy / q
            H[1, lm_idx+1] = dx / q
            # Innovation covariance S = H P H^T + R
            S = H @ self.P @ H.T + self.R
            # Mahalanobis distance
            try:
                maha = float(nu.T @ np.linalg.inv(S) @ nu)
            except np.linalg.LinAlgError:
                maha = float('inf')
            if best_maha is None or maha < best_maha:
                best_maha = maha
                best_j = j
        # Gate test (chi-square with df=2). gate_threshold default ≈ 95%
        if best_maha is not None and best_maha < gate_threshold:
            return best_j
        else:
            return None

    def initialize_landmark(self, measurement):
        """
        Transform range-bearing measurement into world coordinates and add as a landmark.
        z = (r, bearing)
        landmark world position:
          lx = rx + r * cos(bearing + theta)
          ly = ry + r * sin(bearing + theta)
        """
        r_meas, bearing_meas = measurement
        rx, ry, rth = self.x[:3]
        bearing_world = main.wrap_to_pi(bearing_meas + rth)
        lx = rx + r_meas * math.cos(bearing_world)
        ly = ry + r_meas * math.sin(bearing_world)
        self.expand_state_for_new_landmark(np.array([lx, ly]))

    def update_landmark(self, measurement, landmark_index):
        """
        EKF update for a single landmark.
        landmark_index: index among initialized landmarks (0..n_landmarks-1)
        """
        j = landmark_index
        lm_idx = 3 + 2*j
        rx, ry, rth = self.x[:3]
        lx = self.x[lm_idx]
        ly = self.x[lm_idx+1]
        dx = lx - rx
        dy = ly - ry
        q = dx*dx + dy*dy
        sqrt_q = math.sqrt(q)
        expected = np.array([sqrt_q, main.wrap_to_pi(math.atan2(dy, dx) - rth)])
        z = np.array(measurement)
        nu = np.array([z[0] - expected[0], main.wrap_to_pi(z[1] - expected[1])])

        # Compute Jacobian H (2 x n)
        n = self.x.shape[0]
        H = np.zeros((2,n))
        H[0,0] = -dx / sqrt_q
        H[0,1] = -dy / sqrt_q
        H[0,2] = 0.0
        H[1,0] = dy / q
        H[1,1] = -dx / q
        H[1,2] = -1.0
        H[0,lm_idx]   = dx / sqrt_q
        H[0,lm_idx+1] = dy / sqrt_q
        H[1,lm_idx]   = -dy / q
        H[1,lm_idx+1] = dx / q

        S = H @ self.P @ H.T + self.R
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # numerically unstable; skip update
            return
        # State update
        self.x = self.x + (K @ nu)
        self.x[2] = main.wrap_to_pi(self.x[2])
        # Covariance update
        I = np.eye(n)
        self.P = (I - K @ H) @ self.P

    def get_robot_estimate(self):
        return self.x[:3].copy()

    def get_landmark_estimates(self):
        lms = []
        for j in range(self.n_landmarks):
            lm_idx = 3 + 2*j
            lms.append(self.x[lm_idx:lm_idx+2].copy())
        return lms

    def get_landmark_cov(self, j):
        lm_idx = 3 + 2*j
        cov = self.P[lm_idx:lm_idx+2, lm_idx:lm_idx+2].copy()
        return cov

    def reset(self, init_pose=np.array([50.0, 50.0, 0.0])):
        self.x = np.zeros((3,))
        self.x[:3] = init_pose.copy()
        self.P = np.eye(3) * 1e-3
        self.n_landmarks = 0