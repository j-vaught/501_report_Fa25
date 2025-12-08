# visualization/normals.py
"""Surface normal visualization."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def save_normal_rgb(
    N: np.ndarray,
    filepath: str,
    title: str = "Surface Normals"
) -> None:
    """
    Save surface normals as RGB image.
    
    Normals are encoded as colors: (nx, ny, nz) → (R, G, B)
    where [-1, 1] is mapped to [0, 1].
    
    Parameters
    ----------
    N : ndarray, shape (Ny, Nx, 3)
        Unit surface normals
    filepath : str
        Output file path
    title : str
        Plot title
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Map [-1, 1] to [0, 1]
    rgb = (N + 1) / 2
    rgb = np.clip(rgb, 0, 1)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb, origin='lower')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
