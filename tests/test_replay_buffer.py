"""Tests for the FIFO experience replay buffer."""

from __future__ import annotations

import numpy as np
import pytest

from rl_doom.replay_buffer import ReplayBuffer


def _make_transition(value: int = 0) -> tuple:
    obs = np.full((4, 84, 84), value, dtype=np.uint8)
    next_obs = np.full((4, 84, 84), value + 1, dtype=np.uint8)
    return obs, value % 3, float(value), next_obs, value % 2 == 0


def test_push_increments_length() -> None:
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    for i in range(3):
        buf.push(*_make_transition(i))
    assert len(buf) == 3


def test_fifo_eviction_when_full() -> None:
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(*_make_transition(i))
    assert len(buf) == 3


def test_sample_returns_correct_shapes_and_dtypes() -> None:
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        buf.push(*_make_transition(i))
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, 4, 84, 84)
    assert batch["actions"].shape == (8,)
    assert batch["rewards"].shape == (8,)
    assert batch["next_obs"].shape == (8, 4, 84, 84)
    assert batch["dones"].shape == (8,)
    assert batch["rewards"].dtype == np.float32
    assert batch["dones"].dtype == np.float32


def test_sample_larger_than_buffer_raises() -> None:
    buf = ReplayBuffer(capacity=10)
    buf.push(*_make_transition(0))
    with pytest.raises(ValueError):
        buf.sample(5)
