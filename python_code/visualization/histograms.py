# visualization/histograms.py
"""Error histogram visualization."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def save_error_histogram(
    Z_true: np.ndarray,
    Z_est: np.ndarray,
    filepath: str,
    title: str = "Depth Error Histogram"
) -> None:
    """
    Save histogram of depth errors.
    
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
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Mean-center
    Z_true_c = Z_true - np.mean(Z_true)
    Z_est_c = Z_est - np.mean(Z_est)
    
    error = (Z_est_c - Z_true_c).flatten()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(error, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Depth Error')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    
    # Add RMSE annotation
    rmse = np.sqrt(np.mean(error**2))
    ax.text(0.95, 0.95, f'RMSE: {rmse:.4f}', 
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
