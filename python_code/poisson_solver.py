"""
Poisson solver utilities for 2D height field reconstruction.
"""
import numpy as np


def compute_spacing(x: np.ndarray, y: np.ndarray):
    """Compute grid spacing assuming uniform linspace grids along x and y."""
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    return dx, dy


def solve_poisson_fft(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Solve Laplace(z) = f on a 2D grid with spacing dx, dy using FFT.

    The solution enforces zero mean to fix the additive constant.
    Periodic boundary conditions are implied by the FFT formulation.
    """
    H, W = f.shape
    # Frequency grids
    kx = 2 * np.pi * np.fft.fftfreq(W, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(H, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    lambda_k = -(KX ** 2 + KY ** 2)

    F = np.fft.fft2(f)
    # Avoid division by zero at the DC component; fix mean to zero
    lambda_k[0, 0] = 1.0
    F[0, 0] = 0.0

    Z_hat = F / lambda_k
    Z = np.fft.ifft2(Z_hat).real
    return Z
