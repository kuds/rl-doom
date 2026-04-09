"""Double DQN agent with epsilon-greedy exploration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_doom.models import DQNNetwork


class DQNAgent:
    """Double DQN agent.

    Parameters
    ----------
    obs_shape : tuple
        Observation shape, e.g. ``(4, 84, 84)``.
    n_actions : int
        Number of discrete actions.
    lr : float
        Learning rate for the Adam optimizer.
    gamma : float
        Discount factor.
    device : torch.device | str
        Torch device to use.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.n_actions = n_actions

        in_channels = obs_shape[0]
        self.policy_net = DQNNetwork(in_channels, n_actions).to(self.device)
        self.target_net = DQNNetwork(in_channels, n_actions).to(self.device)
        self.update_target()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, *, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            obs_t = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device)
            q_values = self.policy_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, batch: dict[str, Any]) -> float:
        """Run one gradient step on a batch from the replay buffer.

        Uses Double DQN: action selection from policy_net, evaluation
        from target_net, to reduce overestimation bias.
        """
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # Current Q-values for chosen actions
        q_values = self.policy_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_obs).argmax(dim=1)
            next_q = self.target_net(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self) -> None:
        """Hard-copy policy_net weights to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.update_target()
