# surfaces/__init__.py
"""
Surface generators for photometric stereo experiments.
Maps to Section 5.1 Test Surfaces in project_restructured.tex.
"""

from .gaussian import create_gaussian_surface
from .sphere import create_sphere_surface
from .cube import create_cube_surface
from .ellipsoid import create_ellipsoid_surface
from .cone import create_cone_surface
from .saddle import create_saddle_surface
from .peaks import create_peaks_surface
from .sinusoid import create_sinusoid_surface

__all__ = [
    'create_gaussian_surface',
    'create_sphere_surface',
    'create_cube_surface',
    'create_ellipsoid_surface',
    'create_cone_surface',
    'create_saddle_surface',
    'create_peaks_surface',
    'create_sinusoid_surface',
]
