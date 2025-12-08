# visualization/heatmaps.py
"""Heatmap and error map visualization."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def save_heatmap(
    Z: np.ndarray,
    filepath: str,
    title: str = "",
    cmap: str = "viridis",
    center_zero: bool = False
) -> None:
    """
    Save a 2D array as a heatmap image.
    
    Parameters
    ----------
    Z : ndarray
        2D data to visualize
    filepath : str
        Output file path
    title : str
        Plot title
    cmap : str
        Matplotlib colormap
    center_zero : bool
        If True, center colormap at zero (useful for error maps)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if center_zero:
        vmax = np.max(np.abs(Z))
        im = ax.imshow(Z, cmap=cmap, vmin=-vmax, vmax=vmax, origin='lower')
    else:
        im = ax.imshow(Z, cmap=cmap, origin='lower')
    
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)


def save_error_map(
    Z_true: np.ndarray,
    Z_est: np.ndarray,
    filepath: str,
    title: str = "Depth Error"
) -> None:
    """
    Save error map between true and estimated depth.
    
    Parameters
    ----------
    Z_true, Z_est : ndarray
        True and estimated height maps
    filepath : str
        Output file path
    title : str
        Plot title
    """
    # Mean-center both
    Z_true_c = Z_true - np.mean(Z_true)
    Z_est_c = Z_est - np.mean(Z_est)
    
    error = Z_est_c - Z_true_c
    save_heatmap(error, filepath, title, cmap='RdBu_r', center_zero=True)
