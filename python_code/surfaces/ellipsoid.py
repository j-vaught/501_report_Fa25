# surfaces/ellipsoid.py
"""Ellipsoid (triaxial) surface generator."""

import numpy as np
from typing import Tuple


def create_ellipsoid_surface(
    Nx: int = 128,
    Ny: int = 128,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    a: float = 0.8,
    b: float = 0.6,
    c: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Create an ellipsoid surface on a regular grid.
    
    z(x,y) = c * sqrt(1 - x²/a² - y²/b²)
    
    Parameters
    ----------
    Nx, Ny : int
        Grid resolution
    x_range, y_range : tuple
        Domain bounds
    a, b, c : float
        Semi-axes in x, y, z directions
        
    Returns
    -------
    X, Y : ndarray
        Meshgrid coordinates
    Z : ndarray
        Height map (0 outside ellipsoid footprint)
    dx, dy : float
        Grid spacing
    """
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Ellipse footprint
    r2_norm = (X / a)**2 + (Y / b)**2
    inside = r2_norm <= 1
    
    Z = np.zeros_like(X)
    Z[inside] = c * np.sqrt(1 - r2_norm[inside])
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    return X, Y, Z, dx, dy
