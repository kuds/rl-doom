"""Tests for :mod:`rl_doom.multiplayer_env`.

Two layers:

* Pure-logic tests stub out ``vizdoom`` so they run without the native binary
  and exercise reward aggregation, action routing, and PettingZoo contracts.
* An integration smoke test boots a real 1v1 match when the ``vizdoom`` package
  is installed; it is skipped otherwise so CI on vanilla machines stays green.

The fake ``vizdoom`` module + ``fake_vizdoom`` fixture live in ``conftest.py``
so both this file and ``test_self_play.py`` share the same scaffolding.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pure-logic tests
# ---------------------------------------------------------------------------


def test_1v1_env_spaces_and_reset(fake_vizdoom: types.ModuleType) -> None:
    from rl_doom.multiplayer_env import DEATHMATCH_ACTIONS, make_1v1_env

    env = make_1v1_env(resize_shape=(84, 84), num_stack=4)
    try:
        obs, infos = env.reset(seed=0)
        assert env.possible_agents == ["player_0", "player_1"]
        assert env.agents == ["player_0", "player_1"]
        assert set(obs) == {"player_0", "player_1"}
        for a in env.agents:
            assert obs[a].shape == (4, 84, 84)
            assert obs[a].dtype == np.uint8
            assert env.action_space(a).n == len(DEATHMATCH_ACTIONS)
            assert env.observation_space(a).shape == (4, 84, 84)
            assert infos[a] == {}
    finally:
        env.close()


def test_1v1_reward_is_own_frag_minus_own_death(fake_vizdoom: types.ModuleType) -> None:
    from rl_doom.multiplayer_env import make_1v1_env

    # player_0: frags 0 -> 2, deaths 0 -> 0  =>  reward +2
    # player_1: frags 0 -> 0, deaths 0 -> 1  =>  reward -1
    # Each inner list is "game-var values after tick N". Tick 0 is the
    # state read immediately after the first make_action(), which is what
    # step() compares against the reset-time prev of 0.
    fake_vizdoom.DoomGame.scripted = [
        [(2, 0)],
        [(0, 1)],
    ]
    env = make_1v1_env(resize_shape=(84, 84), num_stack=1)
    try:
        env.reset()
        _, rewards, terms, truncs, infos = env.step(
            {"player_0": 0, "player_1": 0},
        )
        assert rewards == {"player_0": 2.0, "player_1": -1.0}
        assert terms == {"player_0": False, "player_1": False}
        assert truncs == {"player_0": False, "player_1": False}
        assert infos["player_0"]["frags"] == 2
        assert infos["player_1"]["deaths"] == 1
    finally:
        env.close()


def test_2v2_reward_is_team_aggregated(fake_vizdoom: types.ModuleType) -> None:
    from rl_doom.multiplayer_env import make_2v2_env

    # Red team (p0 + p1): total frags +3, total deaths +1 => reward +2 each
    # Blue team (p2 + p3): total frags +1, total deaths +3 => reward -2 each
    fake_vizdoom.DoomGame.scripted = [
        [(2, 1)],  # player_0  red
        [(1, 0)],  # player_1  red
        [(1, 2)],  # player_2  blue
        [(0, 1)],  # player_3  blue
    ]
    env = make_2v2_env(resize_shape=(84, 84), num_stack=1)
    try:
        env.reset()
        _, rewards, _, _, infos = env.step(
            {"player_0": 0, "player_1": 1, "player_2": 2, "player_3": 3},
        )
        assert rewards["player_0"] == rewards["player_1"] == 2.0
        assert rewards["player_2"] == rewards["player_3"] == -2.0
        assert infos["player_0"]["team"] == "red"
        assert infos["player_3"]["team"] == "blue"
    finally:
        env.close()


def test_step_missing_action_raises(fake_vizdoom: types.ModuleType) -> None:
    from rl_doom.multiplayer_env import make_1v1_env

    env = make_1v1_env(resize_shape=(84, 84), num_stack=1)
    try:
        env.reset()
        with pytest.raises(KeyError, match="player_1"):
            env.step({"player_0": 0})
    finally:
        env.close()


def test_episode_finished_clears_agents(fake_vizdoom: types.ModuleType) -> None:
    from rl_doom.multiplayer_env import make_1v1_env

    env = make_1v1_env(resize_shape=(84, 84), num_stack=1)
    try:
        env.reset()
        # Force the host game to report a finished match.
        env._games["player_0"]._finished = True
        _, _, terms, _, _ = env.step({"player_0": 0, "player_1": 0})
        assert all(terms.values())
        assert env.agents == []
    finally:
        env.close()


def test_action_table_selects_correct_buttons(fake_vizdoom: types.ModuleType) -> None:
    """Action id 7 in DEATHMATCH_ACTIONS is MOVE_FORWARD + ATTACK: bits
    0 (MOVE_FORWARD) and 6 (ATTACK) must be the ones set in the submitted
    button vector."""
    from rl_doom.multiplayer_env import make_1v1_env

    env = make_1v1_env(resize_shape=(84, 84), num_stack=1)
    try:
        env.reset()
        env.step({"player_0": 7, "player_1": 0})
        vec = env._games["player_0"].actions_submitted[-1]
        assert vec[0] == 1  # MOVE_FORWARD
        assert vec[6] == 1  # ATTACK
        assert sum(vec) == 2
    finally:
        env.close()


def test_second_reset_calls_new_episode_on_all_games(
    fake_vizdoom: types.ModuleType,
) -> None:
    from rl_doom.multiplayer_env import make_1v1_env

    env = make_1v1_env(resize_shape=(84, 84), num_stack=1)
    try:
        env.reset()
        env.reset()
        for game in env._games.values():
            assert game.new_episode_calls == 1
    finally:
        env.close()


def test_rejects_mismatched_teams_length(fake_vizdoom: types.ModuleType) -> None:
    from rl_doom.multiplayer_env import DoomMultiplayerEnv

    with pytest.raises(ValueError, match="teams"):
        DoomMultiplayerEnv(num_players=4, teams=["red", "blue"])


def test_rejects_num_stack_without_resize(fake_vizdoom: types.ModuleType) -> None:
    from rl_doom.multiplayer_env import DoomMultiplayerEnv

    with pytest.raises(ValueError, match="resize_shape"):
        DoomMultiplayerEnv(num_players=2, resize_shape=None, num_stack=4)


# ---------------------------------------------------------------------------
# Live integration smoke test (only runs when vizdoom is installed)
# ---------------------------------------------------------------------------


def test_1v1_live_smoke() -> None:
    pytest.importorskip("vizdoom")
    pytest.importorskip("pettingzoo")
    """Boots a real 1v1 match and steps a few random actions.

    Kept deliberately short - goal is to catch init/teardown breakage, not
    exercise long trajectories. ``sv_spawnfarthest`` + a 1-minute timelimit
    keep the match cheap on CI / Colab.
    """
    from rl_doom.multiplayer_env import make_1v1_env

    env = make_1v1_env(time_limit=1.0, frame_skip=4, resize_shape=(84, 84), num_stack=4)
    try:
        obs, _ = env.reset(seed=0)
        assert set(obs) == {"player_0", "player_1"}
        for _ in range(3):
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rewards, terms, truncs, _ = env.step(actions)
            if all(terms.values()):
                break
        assert all(obs[a].shape == (4, 84, 84) for a in ["player_0", "player_1"])
    finally:
        env.close()
