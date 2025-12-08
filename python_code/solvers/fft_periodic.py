# solvers/fft_periodic.py
"""FFT-based Poisson solver with periodic boundary conditions."""

import numpy as np


def solve_poisson_fft(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Solve Poisson equation ∇²z = f using FFT (periodic BC).
    
    This is Solver 1 from Section 3.2 of project_restructured.tex.
    Assumes periodic boundary conditions.
    
    Parameters
    ----------
    f : ndarray
        Divergence field (source term)
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    Z : ndarray
        Height field solution (mean-centered)
    """
    Ny, Nx = f.shape
    
    # Frequency grids
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    
    # Laplacian eigenvalues: -k² = -(kx² + ky²)
    denom = -(KX**2 + KY**2)
    
    # Avoid division by zero at DC component
    denom[0, 0] = 1.0
    
    # Transform, divide, inverse transform
    F_hat = np.fft.fft2(f)
    Z_hat = F_hat / denom
    Z_hat[0, 0] = 0  # Set DC to zero (removes constant ambiguity)
    
    Z = np.real(np.fft.ifft2(Z_hat))
    
    # Mean-center the result
    Z = Z - np.mean(Z)
    
    return Z
