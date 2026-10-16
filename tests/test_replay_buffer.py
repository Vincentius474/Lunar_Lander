import numpy as np
from lunar_lander.replay_buffer import ReplayBuffer

def test_push_increases_size():
    buf = ReplayBuffer(capacity=10, obs_dim=3)
    assert len(buf) == 0
    for i in range(5):
        buf.push(np.zeros(3), 0, 1.0, np.ones(3), False)
    assert len(buf) == 5

def test_capacity_is_respected():
    buf = ReplayBuffer(capacity=4, obs_dim=2)
    for i in range(10):
        buf.push(np.full(2, i), 0, float(i), np.zeros(2), False)
    assert len(buf) == 4                    # capped at capacity

def test_sample_shapes_and_dtypes():
    buf = ReplayBuffer(capacity=100, obs_dim=3, seed=0)
    for _ in range(50):
        buf.push(np.random.randn(3).astype(np.float32), 1, 0.5,
                 np.random.randn(3).astype(np.float32), False)
    obs, actions, rewards, next_obs, dones = buf.sample(16)
    assert obs.shape == (16, 3)
    assert next_obs.shape == (16, 3)
    assert actions.shape == (16,)
    assert rewards.shape == (16,)
    assert dones.shape == (16,)
    assert obs.dtype == np.float32

def test_wraparound_preserves_recent_data():
    buf = ReplayBuffer(capacity=3, obs_dim=1, seed=0)
    for i in range(5):
        buf.push(
            np.array([float(i)], dtype=np.float32), 0, 0.0,
            np.zeros(1, dtype=np.float32), False,
        )

    # Buffer is full and holds the last `capacity` pushes: 2, 3, 4.
    assert len(buf) == 3
    stored = sorted(buf.obs[: buf.size, 0].tolist())
    assert stored == [2.0, 3.0, 4.0]

def test_sample_is_with_replacement():
    buf = ReplayBuffer(capacity=10, obs_dim=1, seed=0)
    for i in range(10):
        buf.push(
            np.array([float(i)], dtype=np.float32), 0, 0.0,
            np.zeros(1, dtype=np.float32), False,
        )

    # Force a specific RNG so the result is reproducible.
    buf.rng = np.random.default_rng(0)
    obs, *_ = buf.sample(10)

    # Draw 10 from 10 with replacement; some duplicate is essentially certain.
    # (Probability of all-unique is 10! / 10^10 ≈ 0.00036.)
    assert len(set(obs.flatten().tolist())) < 10