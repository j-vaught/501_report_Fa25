# surfaces/peaks.py
"""MATLAB peaks function surface generator."""

import numpy as np
from typing import Tuple


def create_peaks_surface(
    Nx: int = 128,
    Ny: int = 128,
    x_range: Tuple[float, float] = (-3, 3),
    y_range: Tuple[float, float] = (-3, 3)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Create MATLAB peaks function surface on a regular grid.
    
    z(x,y) = 3(1-x)²e^(-x²-(y+1)²) - 10(x/5 - x³ - y⁵)e^(-x²-y²) - (1/3)e^(-(x+1)²-y²)
    
    Parameters
    ----------
    Nx, Ny : int
        Grid resolution
    x_range, y_range : tuple
        Domain bounds (default wider for peaks)
        
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
    
    Z = (3 * (1 - X)**2 * np.exp(-X**2 - (Y + 1)**2)
         - 10 * (X/5 - X**3 - Y**5) * np.exp(-X**2 - Y**2)
         - (1/3) * np.exp(-(X + 1)**2 - Y**2))
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    return X, Y, Z, dx, dy
