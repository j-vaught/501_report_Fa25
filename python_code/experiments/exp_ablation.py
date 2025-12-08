# experiments/exp_ablation.py
"""
Ablation studies: Light count sweep, Noise robustness, Tikhonov regularization.
Maps to Section 5.7 Ablation Studies in project_restructured.tex.
"""

import time
import os
import numpy as np
from typing import Dict, List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surfaces import create_gaussian_surface
from solvers import solve_poisson_fft, solve_poisson_tikhonov
from photometric import (
    make_rotating_lights,
    render_photometric_images,
    photometric_stereo,
    gradients_from_normals,
    normals_from_height,
    compute_divergence,
)
from config import OUTPUT_DIR, LIGHT_SWEEP_RANGE, NOISE_LEVELS, TIKHONOV_LAMBDAS


def compute_rmse(Z_true: np.ndarray, Z_est: np.ndarray) -> float:
    """Compute RMSE between mean-centered height maps."""
    Z_true_c = Z_true - np.mean(Z_true)
    Z_est_c = Z_est - np.mean(Z_est)
    return float(np.sqrt(np.mean((Z_true_c - Z_est_c)**2)))


# ============================================================================
# Ablation 1: Light Count Sweep
# ============================================================================

def run_light_count_sweep(
    m_values: List[int] = None,
    noise_std: float = 0.0,
) -> Dict[int, float]:
    """
    Sweep number of lights from 3 to 20 and measure RMSE.
    Uses Gaussian surface with FFT solver.
    """
    if m_values is None:
        m_values = LIGHT_SWEEP_RANGE
    
    X, Y, Z_true, dx, dy = create_gaussian_surface()
    N_true = normals_from_height(Z_true, dx, dy)
    
    results = {}
    
    for m in m_values:
        lights = make_rotating_lights(m, elevation_deg=45.0)
        images = render_photometric_images(N_true, lights, noise_std=noise_std)
        N_est = photometric_stereo(images, lights)
        p, q = gradients_from_normals(N_est)
        f = compute_divergence(p, q, dx, dy)
        Z_est = solve_poisson_fft(f, dx, dy)
        
        rmse = compute_rmse(Z_true, Z_est)
        results[m] = rmse
        print(f"    m={m:2d} lights: RMSE = {rmse:.6f}")
    
    return results


# ============================================================================
# Ablation 2: Noise Robustness
# ============================================================================

def run_noise_sweep(
    noise_levels: List[float] = None,
    m_lights: int = 16,
) -> Dict[float, float]:
    """
    Sweep noise level from 0 to 0.08 and measure RMSE.
    Uses Gaussian surface with FFT solver.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS
    
    X, Y, Z_true, dx, dy = create_gaussian_surface()
    N_true = normals_from_height(Z_true, dx, dy)
    lights = make_rotating_lights(m_lights, elevation_deg=45.0)
    
    results = {}
    
    for sigma in noise_levels:
        images = render_photometric_images(N_true, lights, noise_std=sigma)
        N_est = photometric_stereo(images, lights)
        p, q = gradients_from_normals(N_est)
        f = compute_divergence(p, q, dx, dy)
        Z_est = solve_poisson_fft(f, dx, dy)
        
        rmse = compute_rmse(Z_true, Z_est)
        results[sigma] = rmse
        print(f"    σ={sigma:.3f}: RMSE = {rmse:.6f}")
    
    return results


# ============================================================================
# Ablation 3: Tikhonov Regularization Sweep
# ============================================================================

def run_tikhonov_sweep(
    lambdas: np.ndarray = None,
    noise_std: float = 0.05,
    m_lights: int = 16,
) -> Dict[str, Dict]:
    """
    Compare unregularized FFT vs Tikhonov-regularized FFT under noise.
    Sweeps lambda from 10^-5 to 1.
    
    Returns dict with:
        - 'fft_rmse': RMSE of unregularized FFT
        - 'tikhonov_results': {lambda: rmse} for each lambda
        - 'optimal_lambda': lambda with lowest RMSE
        - 'optimal_rmse': RMSE at optimal lambda
    """
    if lambdas is None:
        lambdas = TIKHONOV_LAMBDAS
    
    X, Y, Z_true, dx, dy = create_gaussian_surface()
    N_true = normals_from_height(Z_true, dx, dy)
    lights = make_rotating_lights(m_lights, elevation_deg=45.0)
    
    # Add noise
    images = render_photometric_images(N_true, lights, noise_std=noise_std)
    N_est = photometric_stereo(images, lights)
    p, q = gradients_from_normals(N_est)
    f = compute_divergence(p, q, dx, dy)
    
    # Unregularized FFT baseline
    Z_fft = solve_poisson_fft(f, dx, dy)
    fft_rmse = compute_rmse(Z_true, Z_fft)
    print(f"    FFT (no regularization): RMSE = {fft_rmse:.6f}")
    
    # Tikhonov sweep
    tikhonov_results = {}
    for lam in lambdas:
        Z_tik = solve_poisson_tikhonov(f, dx, dy, lam=lam)
        rmse = compute_rmse(Z_true, Z_tik)
        tikhonov_results[float(lam)] = rmse
        print(f"    Tikhonov λ={lam:.1e}: RMSE = {rmse:.6f}")
    
    # Find optimal lambda
    optimal_lambda = min(tikhonov_results, key=tikhonov_results.get)
    optimal_rmse = tikhonov_results[optimal_lambda]
    
    print(f"\n    Optimal λ = {optimal_lambda:.1e} with RMSE = {optimal_rmse:.6f}")
    print(f"    Improvement over FFT: {(fft_rmse - optimal_rmse) / fft_rmse * 100:.1f}%")
    
    return {
        'fft_rmse': fft_rmse,
        'tikhonov_results': tikhonov_results,
        'optimal_lambda': optimal_lambda,
        'optimal_rmse': optimal_rmse,
        'noise_std': noise_std,
    }


# ============================================================================
# Run All Ablation Studies
# ============================================================================

def run_all_ablation_studies() -> Dict:
    """Run all ablation experiments and return results."""
    results = {}
    
    print("\n" + "="*60)
    print("ABLATION STUDY 1: Light Count Sweep")
    print("="*60)
    results['light_sweep'] = run_light_count_sweep()
    
    print("\n" + "="*60)
    print("ABLATION STUDY 2: Noise Robustness")
    print("="*60)
    results['noise_sweep'] = run_noise_sweep()
    
    print("\n" + "="*60)
    print("ABLATION STUDY 3: Tikhonov Regularization Sweep (σ=0.05)")
    print("="*60)
    results['tikhonov_sweep'] = run_tikhonov_sweep(noise_std=0.05)
    
    return results


if __name__ == "__main__":
    results = run_all_ablation_studies()
    
    print("\n" + "="*60)
    print("ABLATION STUDIES COMPLETE")
    print("="*60)
