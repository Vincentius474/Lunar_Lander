import numpy as np
from lunar_lander.physics import integrate, rotate_vector

def test_rotate_vector_quarter_turn():
    x, y = rotate_vector(1.0, 0.0, np.pi / 2)
    assert abs(x) < 1e-8
    assert abs(y - 1.0) < 1e-8

def test_integrate_applies_acceleration():
    pos, vel = integrate(
        np.array([0.0, 0.0]), 
        np.array([0.0, 0.0]), 
        np.array([2.0, 4.0]), 0.5
    )
    assert np.allclose(vel, [1.0, 2.0])
    assert np.allclose(pos, [0.5, 1.0])