# solvers/__init__.py
"""
Poisson solvers for gradient integration.
Maps to Chapter 3 Numerical Methods in project_restructured.tex.
"""

from .fft_periodic import solve_poisson_fft
from .fd_dirichlet import solve_poisson_fd_dirichlet
from .dct_neumann import solve_poisson_dct_neumann
from .tikhonov import solve_poisson_tikhonov

__all__ = [
    'solve_poisson_fft',
    'solve_poisson_fd_dirichlet',
    'solve_poisson_dct_neumann',
    'solve_poisson_tikhonov',
]
