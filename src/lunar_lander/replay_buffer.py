from __future__ import annotations
import numpy as np

class ReplayBuffer:
    """Circular experience replay buffer for DQN."""

    def __init__(self, capacity: int = 50_000, obs_dim: int = 9, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")

        self.capacity = capacity
        self.obs_dim = obs_dim

        self.obs = np.zeros(
            (capacity, obs_dim), dtype=np.float32
        )
        self.next_obs = np.zeros(
            (capacity, obs_dim), dtype=np.float32
        )
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool,) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)

        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"Expected obs shape {(self.obs_dim,)}, "
                f"got {obs.shape}"
            )
        if next_obs.shape != (self.obs_dim,):
            raise ValueError(
                f"Expected next_obs shape {(self.obs_dim,)}, "
                f"got {next_obs.shape}"
            )
        if not np.all(np.isfinite(obs)):
            raise ValueError("obs contains non-finite values")
        if not np.all(np.isfinite(next_obs)):
            raise ValueError("next_obs contains non-finite values")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")

        i = self.ptr

        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size < batch_size:
            raise ValueError(
                f"Not enough samples: have {self.size}, "
                f"requested {batch_size}"
            )

        # Sample with replacement, as in the original implementation.
        idx = self.rng.integers(
            0, self.size, size=batch_size
        )

        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size
