"""CNN policy and value networks for DQN and PPO agents.

Architecture follows the Nature-DQN design (Mnih et al., 2015), proven
effective for Doom-scale visual inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CNNBase(nn.Module):
    """Shared convolutional feature extractor.

    Input : (B, C_in, 84, 84)
    Output: (B, 512)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize uint8 images to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.fc(self.conv(x))


class DQNNetwork(nn.Module):
    """Q-network: maps stacked frames to per-action Q-values.

    Parameters
    ----------
    in_channels : int
        Number of stacked frames (e.g. 4).
    n_actions : int
        Size of the discrete action space.
    """

    def __init__(self, in_channels: int, n_actions: int) -> None:
        super().__init__()
        self.features = _CNNBase(in_channels)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ActorCriticNetwork(nn.Module):
    """Shared-backbone actor-critic for PPO.

    Parameters
    ----------
    in_channels : int
        Number of stacked frames (e.g. 4).
    n_actions : int
        Size of the discrete action space.

    Returns
    -------
    logits : Tensor of shape (B, n_actions)
    value  : Tensor of shape (B, 1)
    """

    def __init__(self, in_channels: int, n_actions: int) -> None:
        super().__init__()
        self.features = _CNNBase(in_channels)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        return self.actor(feat), self.critic(feat)
