"""Smoke tests for the Dreamer env adapter. Requires the vizdoom binary."""

from __future__ import annotations

from typing import Any

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


def test_step_accepts_one_hot_action() -> None:
    """The dreamerv3-torch onehot actor emits one-hot vectors, not ints."""
    from rl_doom.dreamer_env import DreamerDoomEnv

    env = DreamerDoomEnv(
        scenario="defend_the_center", resize_shape=(64, 64), frame_skip=4, grayscale=False,
    )
    try:
        env.reset()
        n = env.action_space.n  # type: ignore[attr-defined]
        one_hot = np.zeros(n, dtype=np.float32)
        one_hot[0] = 1.0
        obs, reward, is_last, info = env.step(one_hot)
        assert isinstance(obs, dict)
        assert isinstance(float(reward), float)
        assert isinstance(is_last, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_timeout_does_not_set_is_terminal() -> None:
    """ViZDoom's scenario timeout should be ``is_last=True, is_terminal=False``.

    ``DoomEnv.step`` folds the timeout into ``terminated=True`` (it never sets
    ``truncated``), distinguishing the cause only via ``info["termination_reason"]``.
    The dreamer adapter must consult that flag — otherwise the world model
    learns that clock-runout is an absorbing terminal state.
    """
    from rl_doom.dreamer_env import DreamerDoomEnv

    class _StubInner:
        def __init__(self) -> None:
            from gymnasium.spaces import Box, Discrete

            self.action_space = Discrete(3)
            self.observation_space = Box(0, 255, (84, 84), dtype=np.uint8)
            self._fake_image = np.zeros((84, 84), dtype=np.uint8)

        def step(self, _action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            return self._fake_image, 0.0, True, False, {"termination_reason": "timeout"}

        def reset(self) -> tuple[Any, dict[str, Any]]:
            return self._fake_image, {}

        def close(self) -> None:
            pass

    env = DreamerDoomEnv.__new__(DreamerDoomEnv)
    env._env = _StubInner()  # type: ignore[attr-defined]
    env._img_shape = (84, 84, 1)  # type: ignore[attr-defined]
    env._grayscale = True  # type: ignore[attr-defined]
    obs, _reward, is_last, _info = env.step(0)
    assert is_last is True, "timeout should still mark episode end"
    assert bool(obs["is_terminal"]) is False, "timeout must not be flagged as terminal"


def test_death_sets_is_terminal() -> None:
    """Real terminations (death / goal_reached) must keep ``is_terminal=True``."""
    from rl_doom.dreamer_env import DreamerDoomEnv

    class _StubInner:
        def __init__(self) -> None:
            from gymnasium.spaces import Box, Discrete

            self.action_space = Discrete(3)
            self.observation_space = Box(0, 255, (84, 84), dtype=np.uint8)
            self._fake_image = np.zeros((84, 84), dtype=np.uint8)

        def step(self, _action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            return self._fake_image, -1.0, True, False, {"termination_reason": "death"}

        def reset(self) -> tuple[Any, dict[str, Any]]:
            return self._fake_image, {}

        def close(self) -> None:
            pass

    env = DreamerDoomEnv.__new__(DreamerDoomEnv)
    env._env = _StubInner()  # type: ignore[attr-defined]
    env._img_shape = (84, 84, 1)  # type: ignore[attr-defined]
    env._grayscale = True  # type: ignore[attr-defined]
    obs, _reward, is_last, _info = env.step(0)
    assert is_last is True
    assert bool(obs["is_terminal"]) is True


def test_id_changes_on_reset() -> None:
    """``tools.simulate`` keys its episode cache by ``env.id``; each episode needs a fresh id."""
    from rl_doom.dreamer_env import DreamerDoomEnv

    env = DreamerDoomEnv(
        scenario="defend_the_center", resize_shape=(64, 64), frame_skip=4, grayscale=False,
    )
    try:
        first = env.id
        env.reset()
        second = env.id
        env.reset()
        third = env.id
        assert isinstance(first, str) and first
        assert second != first
        assert third != second
    finally:
        env.close()
