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


# ---------------------------------------------------------------------------
# Compound action space
# ---------------------------------------------------------------------------


def test_compound_action_space_expands_deadly_corridor() -> None:
    """Deadly Corridor should expose compound actions (e.g. MOVE_FORWARD+ATTACK)
    rather than the 7 one-hot single-button actions."""
    from rl_doom.env import DoomEnv, SCENARIO_ACTION_SETS

    env = DoomEnv(scenario="deadly_corridor")
    try:
        # 7 raw buttons -> curated set has 14 compound actions.
        assert env.action_space.n == len(SCENARIO_ACTION_SETS["deadly_corridor"])
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
        assert env.action_space.n == 7  # 7 raw buttons
        # Every action presses exactly one button.
        assert all(sum(a) == 1 for a in env.available_actions)
    finally:
        env.close()


def test_compound_action_space_basic_and_dtc() -> None:
    """Spot-check the other two curated scenarios."""
    from rl_doom.env import DoomEnv, SCENARIO_ACTION_SETS

    for scenario in ("basic", "defend_the_center"):
        env = DoomEnv(scenario=scenario)
        try:
            assert env.action_space.n == len(SCENARIO_ACTION_SETS[scenario])
            # Each scenario should include at least one "press + ATTACK" compound.
            assert any(sum(a) >= 2 for a in env.available_actions)
        finally:
            env.close()
