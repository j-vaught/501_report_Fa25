# config.py - Shared constants for photometric stereo experiments
"""
Configuration constants shared across all experiment modules.
Maps to project_restructured.tex experimental parameters.
"""

import numpy as np

# Grid resolution
NX = 256
NY = 256

# Domain ranges
X_RANGE = (-1, 1)
Y_RANGE = (-1, 1)

# Light configuration (Section 5.2)
DEFAULT_NUM_LIGHTS = 32
LIGHT_ELEVATION_DEG = 45.0

# Gaussian surface parameters (Section 5.1)
GAUSSIAN_SIGMA = 0.4

# Sphere/Cube parameters
SPHERE_RADIUS = 0.9
CUBE_HALF = 0.35
CUBE_EDGE = 0.1
CUBE_HEIGHT = 0.6

# Ellipsoid parameters
ELLIPSOID_A = 0.8
ELLIPSOID_B = 0.6
ELLIPSOID_C = 0.5

# Cone parameters
CONE_HEIGHT = 0.8
CONE_RADIUS = 0.9

# Saddle parameters
SADDLE_SCALE = 0.3

# Sinusoid parameters
SINUSOID_AMPLITUDE = 0.3

# Peaks domain (wider range)
PEAKS_X_RANGE = (-3, 3)
PEAKS_Y_RANGE = (-3, 3)

# Ablation study parameters (Section 5.7)
LIGHT_SWEEP_RANGE = list(range(3, 21))  # m = 3 to 20
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.08]
RESOLUTION_RANGE = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384]

# Tikhonov regularization sweep (Section 5.8)
TIKHONOV_LAMBDAS = np.logspace(-5, 0, 9)  # 10^-5 to 1

# Output directory
OUTPUT_DIR = "output"
