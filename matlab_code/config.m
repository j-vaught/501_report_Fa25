% config.m - Configuration parameters for photometric stereo experiments
% All experiments should use these shared constants

% Grid resolution
NX = 128;
NY = 128;

% Domain extent
X_MIN = -2.0;
X_MAX = 2.0;
Y_MIN = -2.0;
Y_MAX = 2.0;

% Lighting configuration
DEFAULT_NUM_LIGHTS = 32;
DEFAULT_ELEVATION_DEG = 45.0;

% CG solver parameters
CG_TOL = 1e-10;
CG_MAX_ITER = 5000;

% Noise levels for ablation
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.08];
