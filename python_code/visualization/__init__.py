# visualization/__init__.py
"""
Visualization utilities for photometric stereo results.
"""

from .heatmaps import save_heatmap, save_error_map
from .surfaces_3d import save_3d_surface
from .profiles import save_profile_plot
from .histograms import save_error_histogram
from .normals import save_normal_rgb

__all__ = [
    'save_heatmap',
    'save_error_map',
    'save_3d_surface',
    'save_profile_plot',
    'save_error_histogram',
    'save_normal_rgb',
]
