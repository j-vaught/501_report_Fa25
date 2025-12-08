# experiments/exp_solver_compare.py
"""
Solver comparison experiments: test each shape with all three solvers.
Maps to Section 5.8 Solver Comparison Experiments in project_restructured.tex.
"""

import time
import os
import numpy as np
from typing import Dict, Callable, Tuple, Any

# Import from sibling modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surfaces import (
    create_gaussian_surface,
    create_sphere_surface,
    create_cube_surface,
    create_ellipsoid_surface,
    create_cone_surface,
    create_saddle_surface,
    create_peaks_surface,
    create_sinusoid_surface,
)
from solvers import (
    solve_poisson_fft,
    solve_poisson_fd_dirichlet,
    solve_poisson_dct_neumann,
)
from photometric import (
    make_rotating_lights,
    render_photometric_images,
    photometric_stereo,
    gradients_from_normals,
    normals_from_height,
    compute_divergence,
)
from visualization import (
    save_heatmap,
    save_error_map,
    save_3d_surface,
    save_profile_plot,
    save_error_histogram,
    save_normal_rgb,
)
from config import OUTPUT_DIR


def compute_metrics(Z_true: np.ndarray, Z_est: np.ndarray) -> Dict[str, float]:
    """Compute RMSE between ground truth and estimated height maps."""
    Z_true_c = Z_true - np.mean(Z_true)
    Z_est_c = Z_est - np.mean(Z_est)
    rmse = np.sqrt(np.mean((Z_true_c - Z_est_c)**2))
    return {"rmse": float(rmse)}


def generate_figures(
    shape_name: str,
    solver_name: str,
    Z_true: np.ndarray,
    Z_est: np.ndarray,
    N_true: np.ndarray = None,
    N_est: np.ndarray = None,
) -> None:
    """
    Generate and save figures for a shape/solver combination.
    
    Generates:
    - 3D surface (ground truth and estimated)
    - Depth heatmaps
    - Error map
    - Center-line profile
    - Error histogram
    - Normal maps (if provided)
    """
    base_dir = os.path.join(OUTPUT_DIR, "figures", shape_name, solver_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # 3D surfaces
    save_3d_surface(Z_true, os.path.join(base_dir, "3d_true.png"), 
                    title=f"{shape_name} - Ground Truth")
    save_3d_surface(Z_est, os.path.join(base_dir, "3d_est.png"),
                    title=f"{shape_name} - {solver_name}")
    
    # Depth heatmaps
    save_heatmap(Z_true, os.path.join(base_dir, "depth_true.png"),
                 title=f"{shape_name} - True Depth")
    save_heatmap(Z_est, os.path.join(base_dir, "depth_est.png"),
                 title=f"{shape_name} - Estimated ({solver_name})")
    
    # Error map
    save_error_map(Z_true, Z_est, os.path.join(base_dir, "error_map.png"),
                   title=f"{shape_name} - Depth Error ({solver_name})")
    
    # Profile plot
    save_profile_plot(Z_true, Z_est, os.path.join(base_dir, "profile.png"),
                      title=f"{shape_name} - Center Profile ({solver_name})")
    
    # Error histogram
    save_error_histogram(Z_true, Z_est, os.path.join(base_dir, "histogram.png"),
                         title=f"{shape_name} - Error Distribution ({solver_name})")
    
    # Normal maps
    if N_true is not None:
        save_normal_rgb(N_true, os.path.join(base_dir, "normals_true.png"),
                        title=f"{shape_name} - True Normals")
    if N_est is not None:
        save_normal_rgb(N_est, os.path.join(base_dir, "normals_est.png"),
                        title=f"{shape_name} - Estimated Normals")


def run_shape_all_solvers(
    shape_name: str,
    create_fn: Callable,
    m_lights: int = 16,
    elevation_deg: float = 45.0,
    noise_std: float = 0.0,
    generate_figs: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Test ONE SHAPE with ALL THREE SOLVERS.
    
    Protocol (Section 5.8.1):
    1. Create surface → get Z_true, dx, dy
    2. Compute ground truth normals
    3. Render Lambertian images
    4. Run photometric stereo → get estimated normals
    5. Convert to gradients → compute divergence f
    6. Fork to 3 solvers: FFT, FD-Dirichlet, FD-Neumann
    7. Mean-center all, compute RMSE vs Z_true
    8. Generate figures for each solver
    """
    # Step 1: Create surface
    X, Y, Z_true, dx, dy = create_fn()
    
    # Step 2: Compute ground truth normals
    N_true = normals_from_height(Z_true, dx, dy)
    
    # Step 3: Set up lights and render
    lights = make_rotating_lights(m_lights, elevation_deg)
    images = render_photometric_images(N_true, lights, albedo=1.0, noise_std=noise_std)
    
    # Step 4: Photometric stereo
    N_est = photometric_stereo(images, lights)
    
    # Step 5: Gradients and divergence
    p, q = gradients_from_normals(N_est)
    f = compute_divergence(p, q, dx, dy)
    
    # Step 6: Run all three solvers
    results = {}
    
    solvers = {
        "fft": solve_poisson_fft,
        "fd_dirichlet": solve_poisson_fd_dirichlet,
        "dct_neumann": solve_poisson_dct_neumann,
    }
    
    for solver_name, solver_fn in solvers.items():
        t0 = time.time()
        try:
            Z_est = solver_fn(f, dx, dy)
            elapsed_ms = (time.time() - t0) * 1000
            metrics = compute_metrics(Z_true, Z_est)
            metrics["time_ms"] = elapsed_ms
            metrics["success"] = True
            
            # Generate figures
            if generate_figs:
                generate_figures(shape_name, solver_name, Z_true, Z_est, N_true, N_est)
                
        except Exception as e:
            metrics = {"rmse": float('nan'), "time_ms": 0, "success": False, "error": str(e)}
            Z_est = None
        
        results[solver_name] = metrics
    
    return results


def run_all_shapes_all_solvers(
    m_lights: int = 16,
    elevation_deg: float = 45.0,
    noise_std: float = 0.0,
    generate_figs: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Test ALL 8 SHAPES with ALL 3 SOLVERS.
    """
    shapes = {
        "gaussian": create_gaussian_surface,
        "sphere": create_sphere_surface,
        "cube": create_cube_surface,
        "ellipsoid": create_ellipsoid_surface,
        "cone": create_cone_surface,
        "saddle": create_saddle_surface,
        "peaks": create_peaks_surface,
        "sinusoid": create_sinusoid_surface,
    }
    
    results = {}
    
    for name, create_fn in shapes.items():
        print(f"  Running {name}...")
        results[name] = run_shape_all_solvers(
            shape_name=name,
            create_fn=create_fn,
            m_lights=m_lights,
            elevation_deg=elevation_deg,
            noise_std=noise_std,
            generate_figs=generate_figs,
        )
    
    return results


def print_results_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Pretty-print results as a table."""
    print("\n" + "="*80)
    print("SOLVER COMPARISON RESULTS (256x256, 32 lights)")
    print("="*80)
    print(f"{'Shape':<12} | {'FFT (Periodic)':<14} | {'FD-Dirichlet':<12} | {'DCT (Neumann)':<13}")
    print("-"*80)
    
    for shape_name, solvers in results.items():
        fft_rmse = solvers.get("fft", {}).get("rmse", float('nan'))
        dir_rmse = solvers.get("fd_dirichlet", {}).get("rmse", float('nan'))
        dct_rmse = solvers.get("dct_neumann", {}).get("rmse", float('nan'))
        
        print(f"{shape_name:<12} | {fft_rmse:<14.6f} | {dir_rmse:<12.6f} | {dct_rmse:<13.6f}")
    
    print("="*80)


if __name__ == "__main__":
    print("Running solver comparison on all shapes...")
    results = run_all_shapes_all_solvers()
    print_results_table(results)
