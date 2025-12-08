# photometric/stereo.py
"""Photometric stereo normal estimation."""

import numpy as np


def photometric_stereo(
    images: np.ndarray,
    lights: np.ndarray
) -> np.ndarray:
    """
    Per-pixel least-squares photometric stereo.
    
    Solves S @ g = I for each pixel, where g = albedo * n.
    Returns estimated unit normals.
    
    Parameters
    ----------
    images : ndarray, shape (m, Ny, Nx)
        Intensity images from m light sources
    lights : ndarray, shape (m, 3)
        Light direction matrix S
        
    Returns
    -------
    N_est : ndarray, shape (Ny, Nx, 3)
        Estimated unit surface normals
    """
    m, Ny, Nx = images.shape
    
    # Stack images as (m, N_pixels)
    I = images.reshape(m, -1)  # (m, Ny*Nx)
    
    # Solve least squares: S @ g = I → g = S^+ @ I
    S_pinv = np.linalg.pinv(lights)  # (3, m)
    G = S_pinv @ I  # (3, Ny*Nx)
    
    # Normalize to get unit normals
    norms = np.linalg.norm(G, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    N_flat = G / norms  # (3, Ny*Nx)
    
    # Reshape to (Ny, Nx, 3)
    N_est = N_flat.T.reshape(Ny, Nx, 3)
    
    return N_est


def gradients_from_normals(N_est: np.ndarray) -> tuple:
    """
    Convert unit normals to gradient fields (p, q).
    
    p = -nx/nz (∂z/∂x)
    q = -ny/nz (∂z/∂y)
    
    Parameters
    ----------
    N_est : ndarray, shape (Ny, Nx, 3)
        Unit surface normals
        
    Returns
    -------
    p, q : ndarray
        Gradient fields
    """
    nx = N_est[:, :, 0]
    ny = N_est[:, :, 1]
    nz = N_est[:, :, 2]
    
    # Avoid division by zero where surface is too steep
    nz_safe = np.where(np.abs(nz) > 1e-6, nz, 1e-6)
    
    p = -nx / nz_safe
    q = -ny / nz_safe
    
    return p, q
