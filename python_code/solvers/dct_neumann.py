# solvers/dct_neumann.py
"""DCT-based Poisson solver with true Neumann boundary conditions."""

import numpy as np
from scipy.fftpack import dctn, idctn


def solve_poisson_dct_neumann(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Solve Poisson equation ∇²z = f using DCT (true Neumann BC).
    
    The Discrete Cosine Transform (Type II) naturally enforces
    homogeneous Neumann boundary conditions (∂z/∂n = 0) because
    cosine basis functions have zero derivative at boundaries.
    
    This is more appropriate for Neumann BC than FFT (which assumes periodic).
    
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
    
    # Enforce compatibility: mean(f) must be 0 for pure Neumann
    f_compat = f - np.mean(f)
    
    # DCT-II eigenvalues for Laplacian with Neumann BC
    # λ_ij = -2/dx² * (1 - cos(πi/Nx)) - 2/dy² * (1 - cos(πj/Ny))
    i = np.arange(Nx)
    j = np.arange(Ny)
    
    # Eigenvalues
    lambda_x = -2 / dx**2 * (1 - np.cos(np.pi * i / Nx))
    lambda_y = -2 / dy**2 * (1 - np.cos(np.pi * j / Ny))
    
    LAMBDA_X, LAMBDA_Y = np.meshgrid(lambda_x, lambda_y, indexing='xy')
    denom = LAMBDA_X + LAMBDA_Y
    
    # Avoid division by zero at (0,0) - the DC component
    denom[0, 0] = 1.0
    
    # Forward DCT-II
    F_hat = dctn(f_compat, type=2, norm='ortho')
    
    # Solve in spectral domain
    Z_hat = F_hat / denom
    Z_hat[0, 0] = 0  # Fix DC component (constant ambiguity)
    
    # Inverse DCT-II
    Z = idctn(Z_hat, type=2, norm='ortho')
    
    # Mean-center
    Z = Z - np.mean(Z)
    
    return Z
