"""Smoke tests for the Dreamer env adapter. Requires the vizdoom binary."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

vizdoom = pytest.importorskip("vizdoom")


def test_reset_returns_dict_with_dreamer_keys() -> None:
    from rl_doom.dreamer_env import DreamerDoomEnv

    env = DreamerDoomEnv(
        scenario="defend_the_center", resize_shape=(64, 64), frame_skip=4, grayscale=False,
    )
    try:
        obs = env.reset()
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {"image", "is_first", "is_terminal"}
        assert obs["image"].shape == (64, 64, 3)
        assert obs["image"].dtype == np.uint8
        assert bool(obs["is_first"]) is True
        assert bool(obs["is_terminal"]) is False
    finally:
        env.close()


def test_grayscale_image_shape() -> None:
    from rl_doom.dreamer_env import DreamerDoomEnv

    env = DreamerDoomEnv(
        scenario="defend_the_center", resize_shape=(64, 64), frame_skip=4, grayscale=True,
    )
    try:
        obs = env.reset()
        assert obs["image"].shape == (64, 64, 1)
    finally:
        env.close()


def test_step_returns_4_tuple_and_flags() -> None:
    from rl_doom.dreamer_env import DreamerDoomEnv

    env = DreamerDoomEnv(
        scenario="defend_the_center", resize_shape=(64, 64), frame_skip=4, grayscale=False,
    )
    try:
        env.reset()
        assert isinstance(env.action_space, gym.spaces.Discrete)
        action = env.action_space.sample()
        result = env.step(int(action))
        assert len(result) == 4, "Dreamer port expects 4-tuple, not Gymnasium's 5-tuple"
        obs, reward, is_last, info = result
        assert isinstance(obs, dict)
        assert obs["image"].shape == (64, 64, 3)
        assert bool(obs["is_first"]) is False
        assert isinstance(float(reward), float)
        assert isinstance(is_last, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_rollout_many_steps_no_error() -> None:
    from rl_doom.dreamer_env import DreamerDoomEnv

    env = DreamerDoomEnv(
        scenario="defend_the_center", resize_shape=(64, 64), frame_skip=4, grayscale=False,
    )
    try:
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            _, _, is_last, _ = env.step(int(action))
            if is_last:
                env.reset()
    finally:
        env.close()
