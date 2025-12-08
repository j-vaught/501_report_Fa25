# visualization/surfaces_3d.py
"""3D surface visualization."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def save_3d_surface(
    Z: np.ndarray,
    filepath: str,
    title: str = "",
    elev: float = 30,
    azim: float = 45,
    X: np.ndarray = None,
    Y: np.ndarray = None
) -> None:
    """
    Save 3D surface mesh visualization.
    
    Parameters
    ----------
    Z : ndarray
        Height map
    filepath : str
        Output file path
    title : str
        Plot title
    elev, azim : float
        Viewing angles (elevation, azimuth)
    X, Y : ndarray, optional
        Coordinate grids (default: indices)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if X is None or Y is None:
        Ny, Nx = Z.shape
        X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for faster rendering if large
    step = max(1, Z.shape[0] // 64)
    ax.plot_surface(
        X[::step, ::step], 
        Y[::step, ::step], 
        Z[::step, ::step],
        cmap='viridis',
        edgecolor='none',
        alpha=0.9
    )
    
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
