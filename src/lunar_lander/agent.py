from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import QNetwork

"""
DQN agent with target network and epsilon-greedy exploration.
"""

class DQNAgent:
    """
    Deep Q-Network agent.

    Key pieces:
      - online : the network we train on every step
      - target : a frozen copy used to compute the TD target
      - epsilon: probability of taking a random action (exploration)
    """

    def __init__(self, obs_dim: int = 9, n_actions: int = 6, hidden: int = 128, lr: float = 1e-3, gamma: float = 0.99, 
        epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: float = 0.995, 
        target_update: int = 500, device: str = "cpu", seed: int | None = None):
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.target_update = target_update
        self.device = torch.device(device)

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed or 0)

        # Two identical networks, target is never trained by the optimizer
        self.online = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        # Huber loss = squared error for small |error|, linear for large.
        # More robust than MSE when a rare big TD error shows up.
        self.loss_fn = nn.SmoothL1Loss()
        self.steps = 0

    # Action selection
    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action. `greedy=True` disables exploration (eval)."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(x)
            return int(q.argmax(dim=1).item())

    # Learning step
    def update(self, batch) -> float:
        """One gradient step. Returns the loss as a float for logging."""
        obs, actions, rewards, next_obs, dones = batch

        obs = torch.as_tensor(obs, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)

        # Q(s, a) for the actions we actually took
        q_values = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD target: r + γ * max_a' Q_target(s', a')  (zeroed if done)
        with torch.no_grad():
            next_q = self.target(next_obs).max(dim=1).values
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients: DQN sometimes produces huge gradients on a bad batch.
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Every target_update steps, copy online > target.
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        # Decay exploration rate.
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())


    # Persistence
    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self.steps = ckpt.get("steps", 0)