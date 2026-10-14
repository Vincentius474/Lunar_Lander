import numpy as np
from lunar_lander.environment import Action, EpisodeStatus, LanderEnv

def test_reset_returns_finite_observation():
    env = LanderEnv(seed=42)
    obs = env.reset(seed=42)
    assert obs.shape == (9,)
    assert np.isfinite(obs).all()
    assert env.status == EpisodeStatus.FLYING

def test_step_returns_gym_like_tuple():
    env = LanderEnv(seed=1)
    obs, reward, done, info = env.step(Action.NOOP)
    assert obs.shape == (9,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "status" in info

def test_safe_touchdown_is_success():
    env = LanderEnv(seed=3)
    env.position = np.array([400.0, env.config.terrain_height - 9.0])
    env.velocity = np.array([0.0, 0.0])
    env.angle = 0.0
    _, reward, done, info = env.step(Action.NOOP)
    assert done
    assert info["status"] == "landed"
    assert reward > 0

def test_fast_touchdown_is_crash():
    env = LanderEnv(seed=3)
    env.position = np.array([400.0, env.config.terrain_height - 9.0])
    env.velocity = np.array([0.0, 100.0])
    env.angle = 0.0
    _, reward, done, info = env.step(Action.NOOP)
    assert done
    assert info["status"] == "crashed"
    assert reward < 0