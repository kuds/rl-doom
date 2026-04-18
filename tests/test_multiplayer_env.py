"""Tests for :mod:`rl_doom.multiplayer_env`.

Two layers:

* Pure-logic tests stub out ``vizdoom`` so they run without the native binary
  and exercise reward aggregation, action routing, and PettingZoo contracts.
* An integration smoke test boots a real 1v1 match when the ``vizdoom`` package
  is installed; it is skipped otherwise so CI on vanilla machines stays green.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fake ``vizdoom`` module for pure-logic tests
# ---------------------------------------------------------------------------


class _FakeButton:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeGameVariable:
    FRAGCOUNT = "FRAGCOUNT"
    DEATHCOUNT = "DEATHCOUNT"


class _FakeScreenFormat:
    RGB24 = "RGB24"


class _FakeMode:
    PLAYER = "PLAYER"
    ASYNC_PLAYER = "ASYNC_PLAYER"


class _FakeState:
    def __init__(self, screen: np.ndarray) -> None:
        self.screen_buffer = screen


class _FakeDoomGame:
    """Minimal stand-in that records scripted frag/death sequences.

    Tests set the class-level ``scripted`` list of ``(frags, deaths)`` tuples
    keyed by the order in which games are constructed; each ``make_action``
    call advances the per-game pointer so that successive ``get_game_variable``
    reads return the next scripted value. That lets a test exercise team
    reward math without any ViZDoom networking.
    """

    instances: list["_FakeDoomGame"] = []
    scripted: list[list[tuple[int, int]]] = []

    def __init__(self) -> None:
        # ``idx`` is resolved deterministically in ``add_game_args`` by parsing
        # the ``+name player_N`` token, because init calls run concurrently and
        # construction order is racy.
        self.idx = -1
        self.instances.append(self)
        self._tick = -1  # bumped to 0 on first make_action
        self._buttons = [
            _FakeButton(n)
            for n in (
                "MOVE_FORWARD",
                "MOVE_BACKWARD",
                "TURN_LEFT",
                "TURN_RIGHT",
                "MOVE_LEFT",
                "MOVE_RIGHT",
                "ATTACK",
            )
        ]
        self._available_vars: list[str] = []
        self._screen_h = 120
        self._screen_w = 160
        self._finished = False
        self.init_called = False
        self.closed = False
        self.new_episode_calls = 0
        self.actions_submitted: list[list[int]] = []

    # --- config + lifecycle -----------------------------------------------
    def load_config(self, path: str) -> None:  # noqa: D401
        pass

    def set_window_visible(self, v: bool) -> None:
        pass

    def set_screen_format(self, fmt: Any) -> None:
        pass

    def set_screen_resolution(self, res: Any) -> None:
        pass

    def set_mode(self, mode: Any) -> None:
        pass

    def set_seed(self, seed: int) -> None:
        pass

    def set_available_game_variables(self, vars_: list[Any]) -> None:
        self._available_vars = list(vars_)

    def get_available_game_variables(self) -> list[Any]:
        return list(self._available_vars)

    def add_game_args(self, args: str) -> None:
        # Pull the player index out of "+name player_N" so tests can script
        # per-agent scenarios reliably despite concurrent inits.
        for tok in args.split():
            if tok.startswith("player_"):
                self.idx = int(tok.split("_", 1)[1])
                break

    def init(self) -> None:
        self.init_called = True

    def close(self) -> None:
        self.closed = True

    def new_episode(self) -> None:
        self.new_episode_calls += 1
        self._tick = -1

    # --- per-tic protocol -------------------------------------------------
    def is_player_dead(self) -> bool:
        return False

    def respawn_player(self) -> None:
        pass

    def make_action(self, vec: list[int], tics: int) -> float:
        self.actions_submitted.append(list(vec))
        self._tick += 1
        return 0.0

    def is_episode_finished(self) -> bool:
        return self._finished

    def get_state(self) -> _FakeState:
        return _FakeState(
            np.full((self._screen_h, self._screen_w, 3), self.idx + 1, dtype=np.uint8),
        )

    def get_screen_height(self) -> int:
        return self._screen_h

    def get_screen_width(self) -> int:
        return self._screen_w

    # --- spaces -----------------------------------------------------------
    def get_available_buttons(self) -> list[_FakeButton]:
        return list(self._buttons)

    def get_available_buttons_size(self) -> int:
        return len(self._buttons)

    # --- scripted game variables -----------------------------------------
    def get_game_variable(self, var: str) -> float:
        script = self.scripted[self.idx] if self.idx < len(self.scripted) else []
        # Clamp to the last entry so tests can step past the script without
        # hitting IndexError; the "next" frag/death reading just stays flat.
        t = min(max(self._tick, 0), len(script) - 1) if script else 0
        if not script:
            return 0.0
        frags, deaths = script[t]
        if var == _FakeGameVariable.FRAGCOUNT:
            return float(frags)
        if var == _FakeGameVariable.DEATHCOUNT:
            return float(deaths)
        return 0.0


@pytest.fixture
def fake_vizdoom(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``vizdoom`` module for the duration of a test.

    Reload ``rl_doom.multiplayer_env`` against it so its ``import vizdoom``
    binding points at the fake instead of (a possibly missing) real install.
    """
    fake = types.ModuleType("vizdoom")
    fake.DoomGame = _FakeDoomGame
    fake.GameVariable = _FakeGameVariable
    fake.ScreenFormat = _FakeScreenFormat
    fake.Mode = _FakeMode
    fake.scenarios_path = "/tmp/scenarios"
    monkeypatch.setitem(sys.modules, "vizdoom", fake)

    # Reset per-test class state.
    _FakeDoomGame.instances = []
    _FakeDoomGame.scripted = []

    # Purge any cached import so the reload actually re-binds ``vizdoom``.
    sys.modules.pop("rl_doom.multiplayer_env", None)
    return fake


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
    _FakeDoomGame.scripted = [
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
    _FakeDoomGame.scripted = [
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
        env._games["player_0"]._finished = True  # type: ignore[attr-defined]
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
        vec = env._games["player_0"].actions_submitted[-1]  # type: ignore[attr-defined]
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
        for game in env._games.values():  # type: ignore[attr-defined]
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
