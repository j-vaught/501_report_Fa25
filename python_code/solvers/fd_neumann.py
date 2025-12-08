# solvers/fd_neumann.py
"""Finite difference Poisson solver with Neumann boundary conditions."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def solve_poisson_fd_neumann(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Solve Poisson equation ∇²z = f using finite differences (Neumann BC).
    
    This is Solver 3 from Section 3.4 of project_restructured.tex.
    Zero-flux boundaries: ∂z/∂n|∂Ω = 0.
    
    Parameters
    ----------
    f : ndarray
        Divergence field (source term)
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    Z : ndarray
        Height field solution (mean-centered)
    """
    Ny, Nx = f.shape
    N = Nx * Ny
    h2 = dx * dy
    
    # Enforce compatibility: ∫∫f dA = 0
    f_compat = f - np.mean(f)
    
    def laplacian_matvec(z_vec):
        """Apply discrete Laplacian with Neumann BC."""
        z = z_vec.reshape((Ny, Nx))
        Lz = np.zeros_like(z)
        
        # Interior points: standard 5-point stencil
        Lz[1:-1, 1:-1] = (
            z[1:-1, 2:] + z[1:-1, :-2] +
            z[2:, 1:-1] + z[:-2, 1:-1] -
            4 * z[1:-1, 1:-1]
        ) / h2
        
        # Boundary points with Neumann BC (ghost point reflection)
        # Left edge (i=0): z[-1,j] = z[1,j] → stencil uses 2*z[1,j]
        Lz[1:-1, 0] = (2*z[1:-1, 1] + z[2:, 0] + z[:-2, 0] - 4*z[1:-1, 0]) / h2
        # Right edge (i=Nx-1)
        Lz[1:-1, -1] = (2*z[1:-1, -2] + z[2:, -1] + z[:-2, -1] - 4*z[1:-1, -1]) / h2
        # Bottom edge (j=0)
        Lz[0, 1:-1] = (z[0, 2:] + z[0, :-2] + 2*z[1, 1:-1] - 4*z[0, 1:-1]) / h2
        # Top edge (j=Ny-1)
        Lz[-1, 1:-1] = (z[-1, 2:] + z[-1, :-2] + 2*z[-2, 1:-1] - 4*z[-1, 1:-1]) / h2
        
        # Corners
        Lz[0, 0] = (2*z[0, 1] + 2*z[1, 0] - 4*z[0, 0]) / h2
        Lz[0, -1] = (2*z[0, -2] + 2*z[1, -1] - 4*z[0, -1]) / h2
        Lz[-1, 0] = (2*z[-1, 1] + 2*z[-2, 0] - 4*z[-1, 0]) / h2
        Lz[-1, -1] = (2*z[-1, -2] + 2*z[-2, -1] - 4*z[-1, -1]) / h2
        
        return Lz.flatten()
    
    # Create LinearOperator for matrix-free CG
    A = LinearOperator((N, N), matvec=laplacian_matvec)
    
    # Solve
    z_flat, info = cg(A, f_compat.flatten(), maxiter=2000, rtol=1e-8)
    
    if info != 0:
        print(f"Warning: CG did not converge for Neumann solver (info={info})")
    
    Z = z_flat.reshape((Ny, Nx))
    
    # Mean-center (Neumann solution is unique up to constant)
    Z = Z - np.mean(Z)
    
    return Z
