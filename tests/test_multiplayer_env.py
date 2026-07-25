"""Tests for :mod:`rl_doom.multiplayer_env`.

Two layers:

* Pure-logic tests stub out ``vizdoom`` so they run without the native binary
  and exercise reward aggregation, action routing, and PettingZoo contracts.
* An integration smoke test boots a real 1v1 match. It is opt-in via
  ``RL_DOOM_MULTIPLAYER_TESTS=1`` because ViZDoom multiplayer needs loopback
  networking that most CI sandboxes lack, and fails there by segfaulting
  rather than raising.

The fake ``vizdoom`` module + ``fake_vizdoom`` fixture live in ``conftest.py``
so both this file and ``test_self_play.py`` share the same scaffolding.
"""

from __future__ import annotations

import os
import types
from typing import Any

import gymnasium as gym
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
        # Force the host game to report a finished match. ``_games`` is
        # annotated ``dict[str, vizdoom.DoomGame]``; at runtime the
        # ``fake_vizdoom`` fixture puts a ``_FakeDoomGame`` there, so go
        # through ``Any`` to reach the stub-only recording attributes.
        host: Any = env._games["player_0"]
        host._finished = True
        _, _, terms, _, _ = env.step({"player_0": 0, "player_1": 0})
        assert all(terms.values())
        assert env.agents == []
    finally:
        env.close()


def test_action_table_selects_correct_buttons(fake_vizdoom: types.ModuleType) -> None:
    """The submitted button vector must set exactly the combo's buttons.

    Indices are resolved through ``button_index`` rather than hardcoded: the
    action id -> button mapping is the contract, the positions within ViZDoom's
    button vector are an implementation detail of the scenario cfg.
    """
    from rl_doom.multiplayer_env import DEATHMATCH_ACTIONS, make_1v1_env

    action_id = DEATHMATCH_ACTIONS.index(["MOVE_FORWARD", "ATTACK"])

    env = make_1v1_env(resize_shape=(84, 84), num_stack=1)
    try:
        env.reset()
        env.step({"player_0": action_id, "player_1": 0})
        host: Any = env._games["player_0"]
        vec = host.actions_submitted[-1]
        assert vec[host.button_index("MOVE_FORWARD")] == 1
        assert vec[host.button_index("ATTACK")] == 1
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
            stub: Any = game
            assert stub.new_episode_calls == 1
    finally:
        env.close()


def test_rejects_mismatched_teams_length(fake_vizdoom: types.ModuleType) -> None:
    from rl_doom.multiplayer_env import DoomMultiplayerEnv

    with pytest.raises(ValueError, match="teams"):
        DoomMultiplayerEnv(num_players=4, teams=["red", "blue"])


def test_deathmatch_action_space_matches_single_player() -> None:
    """Self-play requires the same action space as single-player deathmatch.

    ``multiplayer_env``'s docstring promises that "models trained single-agent
    can be loaded directly for self-play". That holds only if both envs agree
    on the action-space *size* and on what each index means — an SB3 zip
    records its action space, so a mismatch in size fails to load outright, and
    a mismatch in ordering silently remaps the policy's outputs.

    The two tables were maintained separately and had drifted to 14 vs 16
    entries with indices 8/9 and 12/13 swapped. Runs without the fake fixture:
    it compares the shared source table against the real scenario cfg.
    """
    pytest.importorskip("vizdoom")

    from rl_doom.env import make_wrapped_env
    from rl_doom.multiplayer_env import DEATHMATCH_ACTIONS
    from rl_doom.scenario_limits import SCENARIO_ACTION_SETS

    assert DEATHMATCH_ACTIONS == SCENARIO_ACTION_SETS["deathmatch"]

    single = make_wrapped_env("deathmatch", resize_shape=(84, 84), num_stack=1)
    try:
        assert isinstance(single.action_space, gym.spaces.Discrete)
        assert int(single.action_space.n) == len(DEATHMATCH_ACTIONS)
    finally:
        single.close()


def test_rejects_num_stack_without_resize(fake_vizdoom: types.ModuleType) -> None:
    from rl_doom.multiplayer_env import DoomMultiplayerEnv

    with pytest.raises(ValueError, match="resize_shape"):
        DoomMultiplayerEnv(num_players=2, resize_shape=None, num_stack=4)


# ---------------------------------------------------------------------------
# Live integration smoke test (only runs when vizdoom is installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RL_DOOM_MULTIPLAYER_TESTS") != "1",
    reason=(
        "ViZDoom multiplayer needs loopback networking and one child process "
        "per seat. Where that is unavailable — most CI sandboxes and "
        "containers — the host/client handshake does not raise, it "
        "segfaults, taking the whole pytest process down with it (see "
        "DoomMultiplayerEnv._start_games, which waits on fut.result() with no "
        "timeout). Opt in with RL_DOOM_MULTIPLAYER_TESTS=1 on a machine where "
        "multiplayer works."
    ),
)
def test_1v1_live_smoke() -> None:
    """Boots a real 1v1 match and steps a few random actions.

    Kept deliberately short - goal is to catch init/teardown breakage, not
    exercise long trajectories. ``sv_spawnfarthest`` + a 1-minute timelimit
    keep the match cheap on CI / Colab.
    """
    pytest.importorskip("pettingzoo")
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
