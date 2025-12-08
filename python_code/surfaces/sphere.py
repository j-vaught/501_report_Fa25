# surfaces/sphere.py
"""Hemisphere surface generator."""

import numpy as np
from typing import Tuple


def create_sphere_surface(
    Nx: int = 128,
    Ny: int = 128,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    radius: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Create a hemisphere surface on a regular grid.
    
    Parameters
    ----------
    Nx, Ny : int
        Grid resolution
    x_range, y_range : tuple
        Domain bounds
    radius : float
        Hemisphere radius
        
    Returns
    -------
    X, Y : ndarray
        Meshgrid coordinates
    Z : ndarray
        Height map (0 outside sphere)
    dx, dy : float
        Grid spacing
    """
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    r2 = X**2 + Y**2
    inside = r2 <= radius**2
    
    Z = np.zeros_like(X)
    Z[inside] = np.sqrt(radius**2 - r2[inside])
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    return X, Y, Z, dx, dy
