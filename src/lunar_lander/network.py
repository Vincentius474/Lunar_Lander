from __future__ import annotations
import torch
import torch.nn as nn

"""
Q-network for DQN: maps a 9-dim observation to one Q-value per action.
"""

class QNetwork(nn.Module):
    """
    Simple feed-forward Q-network.

    Input:  observation vector, shape (batch, obs_dim)
    Output: Q-value for each discrete action, shape (batch, n_actions)
    """

    def __init__(self, obs_dim: int = 9, n_actions: int = 6, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)