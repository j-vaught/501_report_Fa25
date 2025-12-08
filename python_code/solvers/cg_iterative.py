# solvers/cg_iterative.py
"""
Sparse iterative Conjugate Gradient solver for Poisson equation.
Uses scipy.sparse.linalg.cg with explicit iteration count tracking.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


def solve_poisson_cg_iterative(
    f: np.ndarray, 
    dx: float, 
    dy: float,
    tol: float = 1e-10,
    maxiter: int = 5000,
    verbose: bool = False
) -> tuple:
    """
    Solve the Poisson equation using Conjugate Gradient iteration.
    
    This is a pure iterative solver (no direct methods), useful for:
    - Large-scale problems where direct solvers run out of memory
    - Testing convergence behavior
    - Comparison with spectral methods
    
    Parameters
    ----------
    f : np.ndarray
        2D divergence field (right-hand side)
    dx, dy : float
        Grid spacing
    tol : float
        Convergence tolerance (relative residual)
    maxiter : int
        Maximum number of CG iterations
    verbose : bool
        If True, print iteration count
        
    Returns
    -------
    z : np.ndarray
        Reconstructed height field (mean-centered)
    info : dict
        Contains 'iterations', 'converged', 'residual'
    """
    ny, nx = f.shape
    N = nx * ny
    
    # Build sparse Laplacian matrix with Dirichlet BC (z=0 on boundary)
    # Interior points use standard 5-point stencil
    # Boundary points are fixed at 0
    
    # Create coefficient for each term
    cx = 1.0 / (dx * dx)
    cy = 1.0 / (dy * dy)
    cc = -2.0 * (cx + cy)  # center coefficient
    
    # Build sparse matrix using COO format for efficiency
    row_idx = []
    col_idx = []
    values = []
    
    def idx(i, j):
        return i * nx + j
    
    # Build matrix
    for i in range(ny):
        for j in range(nx):
            k = idx(i, j)
            
            # Check if boundary
            if i == 0 or i == ny-1 or j == 0 or j == nx-1:
                # Dirichlet BC: z = 0 on boundary
                row_idx.append(k)
                col_idx.append(k)
                values.append(1.0)
            else:
                # Interior point: 5-point stencil
                # Center
                row_idx.append(k)
                col_idx.append(k)
                values.append(cc)
                
                # Left (j-1)
                row_idx.append(k)
                col_idx.append(idx(i, j-1))
                values.append(cx)
                
                # Right (j+1)
                row_idx.append(k)
                col_idx.append(idx(i, j+1))
                values.append(cx)
                
                # Up (i-1)
                row_idx.append(k)
                col_idx.append(idx(i-1, j))
                values.append(cy)
                
                # Down (i+1)
                row_idx.append(k)
                col_idx.append(idx(i+1, j))
                values.append(cy)
    
    # Create sparse matrix
    A = sparse.coo_matrix((values, (row_idx, col_idx)), shape=(N, N))
    A = A.tocsr()  # Convert to CSR for efficient arithmetic
    
    # Build RHS vector
    b = f.flatten()
    # Zero out boundary values in RHS (Dirichlet BC)
    b_full = np.zeros(N)
    for i in range(ny):
        for j in range(nx):
            k = idx(i, j)
            if i == 0 or i == ny-1 or j == 0 or j == nx-1:
                b_full[k] = 0.0
            else:
                b_full[k] = b[k]
    
    # Track iterations
    iteration_count = [0]
    residuals = []
    
    def callback(xk):
        iteration_count[0] += 1
        residual = np.linalg.norm(A @ xk - b_full)
        residuals.append(residual)
    
    # Solve using CG
    z_flat, info_code = cg(A, b_full, rtol=tol, maxiter=maxiter, callback=callback)
    
    # Reshape to 2D
    z = z_flat.reshape((ny, nx))
    
    # Mean-center the result
    z = z - np.mean(z)
    
    # Build info dict
    converged = (info_code == 0)
    final_residual = residuals[-1] if residuals else 0.0
    
    info = {
        'iterations': iteration_count[0],
        'converged': converged,
        'residual': final_residual,
        'info_code': info_code,
    }
    
    if verbose:
        status = "converged" if converged else "NOT converged"
        print(f"CG solver: {iteration_count[0]} iterations, {status}, residual={final_residual:.2e}")
    
    return z, info


# Wrapper for compatibility with other solvers (returns just z)
def solve_poisson_cg(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Sparse iterative CG solver with default parameters.
    Compatible with other solver interfaces.
    """
    z, _ = solve_poisson_cg_iterative(f, dx, dy, verbose=False)
    return z
