# MATLAB Photometric Stereo Code

MATLAB implementation of the photometric stereo pipeline and Poisson solvers.

## Directory Structure

```
matlab_code/
├── config.m            # Shared parameters
├── runner.m            # Main experiment script
├── surfaces/           # Test surface generators
│   ├── create_gaussian_surface.m
│   ├── create_sphere_surface.m
│   ├── create_cone_surface.m
│   ├── create_saddle_surface.m
│   ├── create_sinusoid_surface.m
│   └── create_peaks_surface.m
├── solvers/            # Poisson equation solvers
│   ├── solve_poisson_fft.m    # Periodic BC
│   ├── solve_poisson_dct.m    # Neumann BC
│   └── solve_poisson_cg.m     # CG iterative (Dirichlet)
└── photometric/        # PS pipeline functions
    ├── make_rotating_lights.m
    ├── height_to_normals.m
    ├── normals_to_gradients.m
    └── compute_divergence.m
```

## Usage

1. Open MATLAB
2. Navigate to `matlab_code/` directory
3. Run the main script:

```matlab
>> runner
```

## Dependencies

- MATLAB R2019b or later
- Signal Processing Toolbox (for dct2/idct2)

## Solvers

1. **FFT** - Uses 2D FFT with periodic boundary conditions
2. **DCT** - Uses DCT-II with Neumann (zero-flux) boundary conditions
3. **CG** - Conjugate Gradient iteration with Dirichlet (zero) boundary conditions
