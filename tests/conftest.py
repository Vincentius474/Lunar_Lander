"""
    Shared pytest fixtures for the LunarLander test suite.
"""

import numpy as np
import pytest

from lunar_lander.environment import LanderEnv, LanderConfig


@pytest.fixture
def fast_config() -> LanderConfig:
    """Deterministic config for fast, reproducible tests."""
    return LanderConfig(
        width=400.0,
        height=300.0,
        gravity=28.0,
        landing_pad_left=165.0,
        landing_pad_right=235.0,
        terrain_height=260.0,
    )


@pytest.fixture
def env(fast_config) -> LanderEnv:
    """A seeded environment at a fixed initial state."""
    e = LanderEnv(config=fast_config, seed=42)
    e.reset(seed=42)
    return e


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)