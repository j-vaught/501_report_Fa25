# surfaces/saddle.py
"""Hyperbolic paraboloid (saddle) surface generator."""

import numpy as np
from typing import Tuple


def create_saddle_surface(
    Nx: int = 128,
    Ny: int = 128,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    scale: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Create a saddle (hyperbolic paraboloid) surface on a regular grid.
    
    z(x,y) = scale * x * y
    
    Parameters
    ----------
    Nx, Ny : int
        Grid resolution
    x_range, y_range : tuple
        Domain bounds
    scale : float
        Amplitude scaling factor
        
    Returns
    -------
    X, Y : ndarray
        Meshgrid coordinates
    Z : ndarray
        Height map
    dx, dy : float
        Grid spacing
    """
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    Z = scale * X * Y
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    return X, Y, Z, dx, dy
