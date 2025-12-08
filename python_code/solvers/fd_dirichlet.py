# solvers/fd_dirichlet.py
"""Finite difference Poisson solver with Dirichlet boundary conditions."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


def solve_poisson_fd_dirichlet(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Solve Poisson equation ∇²z = f using finite differences (Dirichlet BC).
    
    This is Solver 2 from Section 3.3 of project_restructured.tex.
    Boundary values are pinned to zero: z|∂Ω = 0.
    
    Parameters
    ----------
    f : ndarray
        Divergence field (source term)
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    Z : ndarray
        Height field solution
    """
    Ny, Nx = f.shape
    N = Nx * Ny
    
    # Assume uniform spacing for simplicity
    h2 = dx * dy  # Approximation; typically dx ≈ dy
    
    # Build sparse Laplacian matrix with Dirichlet BC (z=0 at boundary)
    # Interior points: (z_i+1,j + z_i-1,j + z_i,j+1 + z_i,j-1 - 4*z_i,j) / h²
    
    main_diag = -4.0 * np.ones(N)
    off_diag_1 = np.ones(N - 1)
    off_diag_Nx = np.ones(N - Nx)
    
    # Remove connections across row boundaries (for off_diag_1)
    for i in range(1, Ny):
        off_diag_1[i * Nx - 1] = 0
    
    diagonals = [main_diag, off_diag_1, off_diag_1, off_diag_Nx, off_diag_Nx]
    offsets = [0, 1, -1, Nx, -Nx]
    
    A = sparse.diags(diagonals, offsets, shape=(N, N), format='csr')
    A = A / h2
    
    # Right-hand side
    b = f.flatten()
    
    # Apply Dirichlet BC: boundary points have z = 0
    # For interior solve, we keep the system and boundary contributions go to RHS
    # But with z=0 on boundary, no modification needed for homogeneous Dirichlet
    
    # Solve using Conjugate Gradient
    z_flat, info = cg(A, b, maxiter=2000, rtol=1e-8)
    
    if info != 0:
        print(f"Warning: CG did not converge (info={info})")
    
    Z = z_flat.reshape((Ny, Nx))
    
    return Z
