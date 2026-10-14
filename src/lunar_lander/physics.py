from __future__ import annotations
import math
import numpy as np

"""Small, explicit physics helpers for the 2D lander."""

def rotate_vector(x: float, y: float, angle: float) -> tuple[float, float]:
    """
        Rotate a vector by angle radians (screen coords: x right, y down).
    """
    c, s = math.cos(angle), math.sin(angle)
    return x * c - y * s, x * s + y * c

def integrate( position: np.ndarray, velocity: np.ndarray,
               acceleration: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
        Semi-implicit Euler integration returns new position and velocity.
    """
    new_velocity = velocity + acceleration * dt
    new_position = position + new_velocity * dt
    return new_position, new_velocity