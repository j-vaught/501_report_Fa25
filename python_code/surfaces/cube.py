# surfaces/cube.py
"""Softened cube surface generator."""

import numpy as np
from typing import Tuple


def create_cube_surface(
    Nx: int = 128,
    Ny: int = 128,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    cube_half: float = 0.35,
    cube_edge: float = 0.1,
    cube_height: float = 0.6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Create a softened cube surface on a regular grid.
    
    The cube uses a smooth transition at edges to avoid gradient singularities.
    
    Parameters
    ----------
    Nx, Ny : int
        Grid resolution
    x_range, y_range : tuple
        Domain bounds
    cube_half : float
        Half-width of flat top region
    cube_edge : float
        Width of edge transition zone
    cube_height : float
        Maximum height of cube
        
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
    
    # Distance from center in L-infinity norm
    d = np.maximum(np.abs(X), np.abs(Y))
    
    # Soft transition: 1 inside, 0 outside, smooth between
    t = (d - cube_half) / cube_edge
    t = np.clip(t, 0, 1)
    
    Z = cube_height * (1 - t)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    return X, Y, Z, dx, dy
