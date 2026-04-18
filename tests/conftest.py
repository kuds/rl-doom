"""Shared pytest fixtures.

The ``fake_vizdoom`` fixture + ``_FakeDoomGame`` live here so multiple test
files (multiplayer env, self-play wrapper, future tournament tests) can
exercise the same multiplayer scaffolding without the native ViZDoom binary.
Each test that depends on the fixture installs a fresh fake ``vizdoom``
module into :data:`sys.modules` and purges any cached import of
``rl_doom.multiplayer_env`` so its ``import vizdoom`` rebinds to the fake.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest


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
    def load_config(self, path: str) -> None:
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
    # setattr (instead of attribute assignment) keeps mypy quiet about
    # decorating a ``ModuleType`` instance with arbitrary attributes.
    for name, value in {
        "DoomGame": _FakeDoomGame,
        "GameVariable": _FakeGameVariable,
        "ScreenFormat": _FakeScreenFormat,
        "Mode": _FakeMode,
        "scenarios_path": "/tmp/scenarios",
    }.items():
        setattr(fake, name, value)
    monkeypatch.setitem(sys.modules, "vizdoom", fake)

    # Reset per-test class state.
    _FakeDoomGame.instances = []
    _FakeDoomGame.scripted = []

    # Purge any cached import so the reload actually re-binds ``vizdoom``.
    sys.modules.pop("rl_doom.multiplayer_env", None)
    return fake
