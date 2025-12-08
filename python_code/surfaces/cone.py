# surfaces/cone.py
"""Soft cone surface generator."""

import numpy as np
from typing import Tuple


def create_cone_surface(
    Nx: int = 128,
    Ny: int = 128,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    height: float = 0.8,
    radius: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Create a soft cone surface on a regular grid.
    
    z(x,y) = height * max(0, 1 - r/radius)
    
    Parameters
    ----------
    Nx, Ny : int
        Grid resolution
    x_range, y_range : tuple
        Domain bounds
    height : float
        Cone apex height
    radius : float
        Base radius
        
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
    
    r = np.sqrt(X**2 + Y**2)
    Z = height * np.maximum(0, 1 - r / radius)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    return X, Y, Z, dx, dy
