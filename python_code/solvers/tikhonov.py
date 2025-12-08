# solvers/tikhonov.py
"""Tikhonov-regularized FFT Poisson solver."""

import numpy as np


def solve_poisson_tikhonov(
    f: np.ndarray,
    dx: float,
    dy: float,
    lam: float = 0.01
) -> np.ndarray:
    """
    Solve regularized Poisson: ∇²z + λ∇⁴z = f via FFT.
    
    This is the Tikhonov regularization from Section 3.5 of project_restructured.tex.
    Higher λ → smoother solution. λ=0 recovers standard Poisson.
    
    Parameters
    ----------
    f : ndarray
        Divergence field (source term)
    dx, dy : float
        Grid spacing
    lam : float
        Regularization parameter (default 0.01)
        
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
    
    k2 = KX**2 + KY**2
    
    # Regularized eigenvalues: -k² - λk⁴
    denom = -k2 - lam * k2**2
    
    # Avoid division by zero at DC
    denom[0, 0] = 1.0
    
    # Transform, divide, inverse transform
    F_hat = np.fft.fft2(f)
    Z_hat = F_hat / denom
    Z_hat[0, 0] = 0  # Set DC to zero
    
    Z = np.real(np.fft.ifft2(Z_hat))
    
    # Mean-center
    Z = Z - np.mean(Z)
    
    return Z
