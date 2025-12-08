# photometric/gradient.py
"""Gradient and normal computation from height maps."""

import numpy as np


def compute_gradients(Z: np.ndarray, dx: float, dy: float) -> tuple:
    """
    Compute finite-difference gradients p = ∂Z/∂x, q = ∂Z/∂y.
    
    Uses central differences in interior, forward/backward at boundaries.
    
    Parameters
    ----------
    Z : ndarray
        Height map
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    p, q : ndarray
        Gradient fields
    """
    p = np.zeros_like(Z)
    q = np.zeros_like(Z)
    
    # Central differences for interior
    p[:, 1:-1] = (Z[:, 2:] - Z[:, :-2]) / (2 * dx)
    q[1:-1, :] = (Z[2:, :] - Z[:-2, :]) / (2 * dy)
    
    # Forward/backward at boundaries
    p[:, 0] = (Z[:, 1] - Z[:, 0]) / dx
    p[:, -1] = (Z[:, -1] - Z[:, -2]) / dx
    q[0, :] = (Z[1, :] - Z[0, :]) / dy
    q[-1, :] = (Z[-1, :] - Z[-2, :]) / dy
    
    return p, q


def normals_from_height(Z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute unit surface normals from height map.
    
    n = [-p, -q, 1] / ||[-p, -q, 1]||
    
    Parameters
    ----------
    Z : ndarray
        Height map
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    N : ndarray, shape (Ny, Nx, 3)
        Unit surface normals
    """
    p, q = compute_gradients(Z, dx, dy)
    
    Ny, Nx = Z.shape
    N = np.zeros((Ny, Nx, 3))
    N[:, :, 0] = -p
    N[:, :, 1] = -q
    N[:, :, 2] = 1.0
    
    # Normalize
    norms = np.linalg.norm(N, axis=2, keepdims=True)
    N = N / norms
    
    return N


def compute_divergence(p: np.ndarray, q: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute divergence field f = ∂p/∂x + ∂q/∂y.
    
    This is the source term for the Poisson equation.
    
    Parameters
    ----------
    p, q : ndarray
        Gradient fields
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    f : ndarray
        Divergence field
    """
    # Central differences
    dp_dx = np.zeros_like(p)
    dq_dy = np.zeros_like(q)
    
    dp_dx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dx)
    dq_dy[1:-1, :] = (q[2:, :] - q[:-2, :]) / (2 * dy)
    
    # Boundaries
    dp_dx[:, 0] = (p[:, 1] - p[:, 0]) / dx
    dp_dx[:, -1] = (p[:, -1] - p[:, -2]) / dx
    dq_dy[0, :] = (q[1, :] - q[0, :]) / dy
    dq_dy[-1, :] = (q[-1, :] - q[-2, :]) / dy
    
    f = dp_dx + dq_dy
    
    return f
