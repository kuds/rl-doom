"""Environment wrapper tests that require the vizdoom binary."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

vizdoom = pytest.importorskip("vizdoom")


def _discrete_n(space: gym.Space) -> int:
    """Narrow a generic ``Space`` to ``Discrete`` and return its size."""
    assert isinstance(space, gym.spaces.Discrete), (
        f"Expected Discrete action space, got {type(space).__name__}"
    )
    return int(space.n)


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


def test_num_bots_rejects_negative() -> None:
    from rl_doom.env import DoomEnv

    with pytest.raises(ValueError, match=r"num_bots must be in \[0, 8\]"):
        DoomEnv(scenario="deathmatch", num_bots=-1)


def test_num_bots_rejects_above_max() -> None:
    from rl_doom.env import DoomEnv

    with pytest.raises(ValueError, match=r"num_bots must be in \[0, 8\]"):
        DoomEnv(scenario="deathmatch", num_bots=9)


def test_num_bots_accepts_max() -> None:
    from rl_doom.env import MAX_NUM_BOTS, DoomEnv

    assert MAX_NUM_BOTS == 8
    env = DoomEnv(scenario="deathmatch", num_bots=MAX_NUM_BOTS)
    try:
        assert env._num_bots == MAX_NUM_BOTS
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Compound action space
# ---------------------------------------------------------------------------


def test_compound_action_space_expands_deadly_corridor() -> None:
    """Deadly Corridor should expose compound actions (e.g. MOVE_FORWARD+ATTACK)
    rather than the 7 one-hot single-button actions."""
    from rl_doom.env import SCENARIO_ACTION_SETS, DoomEnv

    env = DoomEnv(scenario="deadly_corridor")
    try:
        # 7 raw buttons -> curated set has 14 compound actions.
        assert _discrete_n(env.action_space) == len(SCENARIO_ACTION_SETS["deadly_corridor"])
        # At least one action must press two buttons simultaneously.
        multi = [a for a in env.available_actions if sum(a) > 1]
        assert multi, "compound action set did not produce any multi-button actions"
    finally:
        env.close()


def test_compound_action_space_opt_out_returns_one_hot() -> None:
    """With ``use_compound_actions=False`` we fall back to the legacy one-hot
    layout (one action per raw button)."""
    from rl_doom.env import DoomEnv

    env = DoomEnv(scenario="deadly_corridor", use_compound_actions=False)
    try:
        assert _discrete_n(env.action_space) == 7  # 7 raw buttons
        # Every action presses exactly one button.
        assert all(sum(a) == 1 for a in env.available_actions)
    finally:
        env.close()


def test_compound_action_space_basic_and_dtc() -> None:
    """Spot-check the other two curated scenarios."""
    from rl_doom.env import SCENARIO_ACTION_SETS, DoomEnv

    for scenario in ("basic", "defend_the_center"):
        env = DoomEnv(scenario=scenario)
        try:
            assert _discrete_n(env.action_space) == len(SCENARIO_ACTION_SETS[scenario])
            # Each scenario should include at least one "press + ATTACK" compound.
            assert any(sum(a) >= 2 for a in env.available_actions)
        finally:
            env.close()
