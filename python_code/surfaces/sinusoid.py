# surfaces/sinusoid.py
"""Sinusoidal surface generator."""

import numpy as np
from typing import Tuple


def create_sinusoid_surface(
    Nx: int = 128,
    Ny: int = 128,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    amplitude: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Create a sinusoidal bump surface on a regular grid.
    
    z(x,y) = amplitude * sin(πx) * sin(πy)
    
    Parameters
    ----------
    Nx, Ny : int
        Grid resolution
    x_range, y_range : tuple
        Domain bounds
    amplitude : float
        Surface amplitude
        
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
    
    Z = amplitude * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    return X, Y, Z, dx, dy
