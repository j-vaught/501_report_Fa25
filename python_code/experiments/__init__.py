# experiments/__init__.py
"""
Experiment runners for photometric stereo validation.
Maps to Chapter 5 Experimental Validation in project_restructured.tex.
"""

from .exp_solver_compare import run_shape_all_solvers, run_all_shapes_all_solvers

__all__ = [
    'run_shape_all_solvers',
    'run_all_shapes_all_solvers',
]
