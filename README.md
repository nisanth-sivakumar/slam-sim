# EKF-SLAM 2D Simulation

A real-time 2D Simultaneous Localization and Mapping (SLAM) simulation built from scratch in Python, implementing a full Extended Kalman Filter SLAM pipeline with a LiDAR-like sensor model, nearest-neighbor data association, and live pygame visualization.

---

## Overview

This project implements the full EKF-SLAM loop — motion prediction, sensor simulation, data association, and state update — operating on a joint state vector over robot pose and landmark positions simultaneously.

The goal was to build the algorithm from the math up: Jacobian derivations by hand, covariance propagation through the unicycle motion model, and chi-square gated nearest-neighbor association — rather than using an off-the-shelf SLAM library.

**Key results across 20+ randomized environments:**
- Localization error held under **5% of total path length**
- Landmark position estimates converge within **~10 observations** per landmark
- Stable covariance convergence with no filter divergence under default noise parameters

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    main.py (loop)                   │
│                                                     │
│  ┌──────────┐   odometry    ┌─────────────────────┐ │
│  │ world_c  │ ────────────► │     ekf_slam.py     │ │
│  │  (truth) │               │                     │ │
│  │          │ ─range/bear.► │  predict()          │ │
│  │ sensor.py│               │  observe()          │ │
│  │ (noise)  │               │    data_association │ │
│  └──────────┘               │    update_landmark  │ │
│                             └─────────────────────┘ │
│                                      │              │
│                             ┌────────▼────────┐     │
│                             │   viewer_c.py   │     │
│                             │  (pygame viz)   │     │
│                             └─────────────────┘     │
└─────────────────────────────────────────────────────┘
```

Each simulation step:
1. **world_c** steps the ground-truth robot pose using a unicycle model
2. **sensor** corrupts the true pose delta into a noisy odometry reading, and generates range-bearing measurements to visible landmarks with configurable Gaussian noise
3. **ekf_slam** runs a predict step from odometry, then an observe step for each measurement — associating to known landmarks or initializing new ones
4. **viewer_c** renders ground truth vs. estimate, covariance ellipses, sensor rays, and path traces

---

## EKF-SLAM Implementation

### State Vector

The filter maintains a joint state over robot pose and all observed landmark positions:

```
x = [r_x, r_y, r_θ,  lm₁_x, lm₁_y,  lm₂_x, lm₂_y,  ...]
    └──── robot ────┘ └── landmark 1 ─┘ └── landmark 2 ─┘

Covariance P: (3 + 2M) × (3 + 2M)
```

New landmarks are appended dynamically on first observation. Cross-covariance blocks between robot and landmark states are maintained throughout, capturing the correlations that make EKF-SLAM consistent.

---

### Prediction Step — Unicycle Motion Model

Robot pose is propagated using a differential-drive (unicycle) model:

```
x'  =  x + v·cos(θ)·dt
y'  =  y + v·sin(θ)·dt
θ'  =  θ + ω·dt
```

The full state Jacobian **G** is constructed to apply the motion update only to the robot sub-state while leaving landmark estimates unchanged. Process noise **Q** is derived from control-space noise via the input Jacobian **V**:

```
Q_robot = V · M · Vᵀ
M = diag([σ_v², σ_ω²])  =  diag([0.25,  (10°)²])
```

Covariance is propagated as `P = G·P·Gᵀ + Q`.

---

### Measurement Model — Range-Bearing Sensor

The sensor produces range and bearing measurements to landmarks within `max_range = 250 px`:

```
z = [r,  φ]
r  =  √((lx - rx)² + (ly - ry)²)  +  η_r
φ  =  atan2(ly - ry, lx - rx) - θ  +  η_φ

Measurement noise:  R = diag([σ_r², σ_φ²]) = diag([4.0,  (5°)²])
```

The measurement Jacobian **H** is derived analytically with respect to the full state vector, with nonzero entries for both the robot pose block and the associated landmark block.

---

### Data Association — Mahalanobis Gating

For each incoming measurement, the filter searches all initialized landmarks and computes the **Mahalanobis distance** against the predicted measurement and its innovation covariance:

```
ν  =  z - ẑ(x, lm_j)
S  =  H·P·Hᵀ + R
d²  =  νᵀ · S⁻¹ · ν
```

Association is accepted only if `d² < 5.991` — the 95% confidence threshold of a chi-square distribution with 2 degrees of freedom. Measurements that fail the gate for all known landmarks initialize a new landmark.

Euclidean distance alone would produce false associations under uncertainty; Mahalanobis gating accounts for the current covariance of each landmark estimate, rejecting associations that are statistically inconsistent even if spatially close.

---

### EKF Update

For an associated measurement, the standard EKF update is applied:

```
K  =  P·Hᵀ·S⁻¹          (Kalman gain)
x  =  x + K·ν            (state update)
P  =  (I - K·H)·P        (covariance update)
```

Bearing innovations are wrapped to `[-π, π]` before the update to prevent angle discontinuity errors.

---

## File Structure

```
slam-sim/
├── main.py          # Simulation loop, control logic, rendering orchestration
├── ekf_slam.py      # EKF-SLAM algorithm: predict, observe, data association, update
├── sensor.py        # Odometry and range-bearing measurement simulation with noise models
├── world_c.py       # Ground-truth world: robot dynamics, landmark map, timestep
└── viewer_c.py      # Pygame renderer: poses, traces, covariance ellipses, HUD
```

---

## Installation

```bash
git clone https://github.com/nisanth-sivakumar/slam-sim.git
cd slam-sim
pip install numpy pygame
python main.py
```

**Requirements:** Python 3.8+, NumPy, Pygame

---

## Controls

| Key | Action |
|---|---|
| `SPACE` | Toggle autonomous / manual drive |
| `↑ / ↓` | Forward / reverse (manual mode) |
| `← / →` | Turn left / right (manual mode) |
| `R` | Reset with a new randomized landmark map |
| `ESC` | Quit |

---

## Configuration

Key parameters in `main.py` and `ekf_slam.py`:

| Parameter | Default | Effect |
|---|---|---|
| `n_landmarks` | 16 | Number of landmarks in the world |
| `max_range` | 250 px | Sensor range cutoff |
| `sensor_noise` | `(1.5, 3°)` | Range / bearing std dev |
| `odom_noise_std` | `(0.5, 0.02)` | Translational / rotational odometry noise |
| `motion_noise` | `diag([0.25, (10°)²])` | EKF process noise (control space) |
| `R` | `diag([4.0, (5°)²])` | EKF measurement noise |
| `gate_threshold` | 5.991 | Chi-square gate (95%, df=2) |

---

## What This Demonstrates

- **EKF-SLAM from scratch** — no SLAM library; full Jacobian derivations, joint covariance propagation, and landmark lifecycle management implemented directly
- **Principled data association** — chi-square Mahalanobis gating rather than naive nearest-neighbor
- **Sensor modeling** — configurable Gaussian noise on range and bearing to simulate real LiDAR conditions
- **Filter consistency** — covariance ellipses remain calibrated and converge correctly across varied environments
