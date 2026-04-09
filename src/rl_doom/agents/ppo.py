"""PPO (Proximal Policy Optimization) agent with GAE."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_doom.models import ActorCriticNetwork


class PPOAgent:
    """PPO agent with clipped surrogate objective and GAE.

    Parameters
    ----------
    obs_shape : tuple
        Observation shape, e.g. ``(4, 84, 84)``.
    n_actions : int
        Number of discrete actions.
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation.
    clip_eps : float
        PPO clipping parameter.
    entropy_coef : float
        Entropy bonus coefficient.
    value_coef : float
        Value loss coefficient.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    device : torch.device | str
        Torch device.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        in_channels = obs_shape[0]
        self.network = ActorCriticNetwork(in_channels, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray,
    ) -> tuple[int, float, float]:
        """Select an action using the current policy.

        Returns
        -------
        action : int
        log_prob : float
        value : float
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device)
            logits, value = self.network(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gae(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Returns (advantages, returns) as numpy arrays.
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(
        self,
        rollout: dict[str, Any],
        *,
        n_epochs: int = 4,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Run PPO update on a collected rollout.

        Returns dict with ``policy_loss``, ``value_loss``, ``entropy``,
        and ``clip_fraction``.
        """
        obs = rollout["obs"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        values = rollout["values"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        last_value = rollout["last_value"]

        advantages, returns = self._compute_gae(
            rewards, values, dones, last_value, self.gamma, self.gae_lambda,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_logp_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        n_samples = len(obs)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                mb_obs = obs_t[idx]
                mb_actions = actions_t[idx]
                mb_old_logp = old_logp_t[idx]
                mb_advantages = advantages_t[idx]
                mb_returns = returns_t[idx]

                logits, values_pred = self.network(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Clipped surrogate objective
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values_pred.squeeze(-1), mb_returns)

                # Combined loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track stats
                clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_frac += clip_frac
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "clip_fraction": total_clip_frac / n_updates,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(state_dict)
