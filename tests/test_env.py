"""Environment wrapper tests that require the vizdoom binary."""

from __future__ import annotations

import numpy as np
import pytest

vizdoom = pytest.importorskip("vizdoom")


def test_make_wrapped_env_reset_shape() -> None:
    from rl_doom.env import make_wrapped_env

    env = make_wrapped_env("basic", resize_shape=(84, 84), frame_skip=4, num_stack=4)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
    finally:
        env.close()


def test_make_wrapped_env_step_contract() -> None:
    from rl_doom.env import make_wrapped_env

    env = make_wrapped_env("basic", resize_shape=(84, 84), frame_skip=4, num_stack=4)
    try:
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (4, 84, 84)
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_unknown_scenario_raises() -> None:
    from rl_doom.env import DoomEnv

    with pytest.raises(ValueError):
        DoomEnv(scenario="not_a_real_scenario")
