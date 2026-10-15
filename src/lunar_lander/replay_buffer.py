from __future__ import annotations
import numpy as np

"""
Fixed-size experience replay buffer for off-policy RL.
"""

class ReplayBuffer:
    """
    Circular buffer storing (obs, action, reward, next_obs, done) tuples.

    Uses pre-allocated numpy arrays for speed. When full, the oldest entry is
    overwritten — this is the "circular" part.
    """

    def __init__(self, capacity: int = 100_000, obs_dim: int = 9, seed: int | None = None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0    
        self.size = 0     
        self.rng = np.random.default_rng(seed)

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """
        Store one transition. Oldest entry is overwritten when full.
        """
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Uniformly sample a batch of transitions.
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: have {self.size}, requested {batch_size}")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size