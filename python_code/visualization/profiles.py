# visualization/profiles.py
"""Profile/cross-section plot visualization."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def save_profile_plot(
    Z_true: np.ndarray,
    Z_est: np.ndarray,
    filepath: str,
    title: str = "Center-line Profile",
    axis: str = "x"
) -> None:
    """
    Save a central cross-section comparison plot.
    
    Parameters
    ----------
    Z_true : ndarray
        Ground truth height map
    Z_est : ndarray
        Estimated height map
    filepath : str
        Output file path
    title : str
        Plot title
    axis : str
        'x' for horizontal slice, 'y' for vertical slice
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Mean-center
    Z_true_c = Z_true - np.mean(Z_true)
    Z_est_c = Z_est - np.mean(Z_est)
    
    Ny, Nx = Z_true.shape
    
    if axis == 'x':
        mid = Ny // 2
        x = np.arange(Nx)
        true_profile = Z_true_c[mid, :]
        est_profile = Z_est_c[mid, :]
        xlabel = 'x index'
    else:
        mid = Nx // 2
        x = np.arange(Ny)
        true_profile = Z_true_c[:, mid]
        est_profile = Z_est_c[:, mid]
        xlabel = 'y index'
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, true_profile, 'b-', label='Ground Truth', linewidth=2)
    ax.plot(x, est_profile, 'r--', label='Estimated', linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Height (centered)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
