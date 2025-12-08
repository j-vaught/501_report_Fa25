# photometric/__init__.py
"""
Photometric stereo pipeline components.
Maps to Chapter 4 Implementation in project_restructured.tex.
"""

from .lighting import make_rotating_lights
from .rendering import render_photometric_images
from .stereo import photometric_stereo, gradients_from_normals
from .gradient import compute_gradients, normals_from_height, compute_divergence

__all__ = [
    'make_rotating_lights',
    'render_photometric_images',
    'photometric_stereo',
    'gradients_from_normals',
    'compute_gradients',
    'normals_from_height',
    'compute_divergence',
]
