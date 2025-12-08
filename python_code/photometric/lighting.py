# photometric/lighting.py
"""Light source configuration for photometric stereo."""

import numpy as np
from typing import List


def make_rotating_lights(
    m: int = 16,
    elevation_deg: float = 45.0
) -> np.ndarray:
    """
    Create lights evenly spaced in azimuth at fixed elevation.
    
    Parameters
    ----------
    m : int
        Number of lights
    elevation_deg : float
        Elevation angle above horizontal (degrees)
        
    Returns
    -------
    lights : ndarray, shape (m, 3)
        Unit light direction vectors [Lx, Ly, Lz]
    """
    elevation = np.deg2rad(elevation_deg)
    azimuths = np.linspace(0, 2 * np.pi, m, endpoint=False)
    
    lights = np.zeros((m, 3))
    lights[:, 0] = np.cos(azimuths) * np.sin(elevation)  # Lx
    lights[:, 1] = np.sin(azimuths) * np.sin(elevation)  # Ly
    lights[:, 2] = np.cos(elevation)                      # Lz
    
    return lights
