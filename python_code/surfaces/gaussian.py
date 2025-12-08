# surfaces/gaussian.py
"""Gaussian bump surface generator."""

import numpy as np
from typing import Tuple


def create_gaussian_surface(
    Nx: int = 128,
    Ny: int = 128,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    sigma: float = 0.4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Create a Gaussian bump surface on a regular grid.
    
    Parameters
    ----------
    Nx, Ny : int
        Grid resolution
    x_range, y_range : tuple
        Domain bounds
    sigma : float
        Gaussian width parameter
        
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
    
    Z = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    return X, Y, Z, dx, dy
