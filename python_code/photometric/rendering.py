# photometric/rendering.py
"""Lambertian image rendering for photometric stereo."""

import numpy as np


def render_photometric_images(
    N: np.ndarray,
    lights: np.ndarray,
    albedo: float = 1.0,
    noise_std: float = 0.0
) -> np.ndarray:
    """
    Render Lambertian images given surface normals and light directions.
    
    I = albedo * max(0, N · L) + noise
    
    Parameters
    ----------
    N : ndarray, shape (Ny, Nx, 3)
        Unit surface normals
    lights : ndarray, shape (m, 3)
        Unit light direction vectors
    albedo : float
        Surface albedo (reflectance)
    noise_std : float
        Standard deviation of Gaussian noise to add
        
    Returns
    -------
    images : ndarray, shape (m, Ny, Nx)
        Rendered intensity images
    """
    m = lights.shape[0]
    Ny, Nx = N.shape[:2]
    
    images = np.zeros((m, Ny, Nx))
    
    for k in range(m):
        L = lights[k]
        # Dot product N · L at each pixel
        I = albedo * np.sum(N * L, axis=2)
        # Clamp negative (self-shadowing)
        I = np.maximum(0, I)
        images[k] = I
    
    # Add noise if requested
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, images.shape)
        images = images + noise
        images = np.clip(images, 0, None)  # Keep non-negative
    
    return images
