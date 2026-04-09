"""Experience replay buffer for off-policy algorithms (DQN)."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer that stores transitions as dicts.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self._buffer: deque[tuple[Any, ...]] = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> dict[str, Any]:
        batch = random.sample(self._buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return {
            "obs": np.array(obs),
            "actions": np.array(actions),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_obs": np.array(next_obs),
            "dones": np.array(dones, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self._buffer)
