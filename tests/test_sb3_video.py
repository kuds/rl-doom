"""Tests for :func:`rl_doom.sb3_utils._record_video`.

These tests exercise the env-config plumbing that was added to keep the
recorded video's difficulty in lock-step with the eval env (otherwise
the clip shows a much harder game than the model was trained on, because
several scenario .cfg files default to ``doom_skill = 5``).

We fake out ``make_wrapped_env``, the SB3 model's ``predict``, and
``imageio`` so the test runs without ViZDoom / ffmpeg installed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from rl_doom import sb3_utils


class _FakeEnv:
    """Minimal env that reports the kwargs it was built with.

    ``_record_video`` walks ``.env`` to find a thing that exposes
    ``render()`` — this fake implements that directly and also records
    the env-config kwargs so the test can assert on them.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._steps = 0
        self._max_steps = 3  # terminates quickly to keep the test fast

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._steps = 0
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._steps += 1
        done = self._steps >= self._max_steps
        info = {"termination_reason": "goal_reached"} if done else {}
        return (
            np.zeros((4, 84, 84), dtype=np.uint8),
            1.0,
            done,
            False,
            info,
        )

    def render(self) -> np.ndarray:
        return np.zeros((120, 160, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


class _FakeModel:
    """SB3-like stand-in: only ``predict`` is exercised."""

    def predict(
        self, obs: np.ndarray, *, state: Any = None, episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]:
        return np.array(0), state


@pytest.fixture
def fake_make_env(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ``rl_doom.env.make_wrapped_env`` with a stub that captures kwargs.

    We install a minimal fake ``rl_doom.env`` module into ``sys.modules``
    so ``_record_video``'s inner ``from rl_doom.env import make_wrapped_env``
    picks up the stub without triggering the real module's ``import vizdoom``.
    That keeps this test runnable on CI where the ViZDoom native binary
    isn't installed.
    """
    import sys
    import types

    calls: list[dict[str, Any]] = []

    def _stub(scenario: str, **kwargs: Any) -> _FakeEnv:
        calls.append({"scenario": scenario, **kwargs})
        return _FakeEnv(scenario=scenario, **kwargs)

    fake_mod = types.ModuleType("rl_doom.env")
    fake_mod.make_wrapped_env = _stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rl_doom.env", fake_mod)
    return calls


@pytest.fixture
def fake_imageio(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-op imageio so the test doesn't need ffmpeg."""
    import sys
    import types

    fake = types.ModuleType("imageio")

    def _mimsave(path: Any, frames: Any, **_: Any) -> None:
        # Touch the file so the "written" path exists.
        open(str(path), "wb").close()

    fake.mimsave = _mimsave  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", fake)


def test_record_video_forwards_env_config(
    tmp_path, fake_make_env, fake_imageio,
) -> None:
    """The env passed to ``make_wrapped_env`` must carry the doom_skill /
    num_bots the caller specified."""
    out_path = tmp_path / "clip.mp4"

    sb3_utils._record_video(
        SimpleNamespace(predict=_FakeModel().predict),  # type: ignore[arg-type]
        "deadly_corridor",
        out_path,
        fps=10,
        n_episodes=1,
        doom_skill=3,
        num_bots=0,
        resize_shape=(84, 84),
        frame_skip=4,
        num_stack=4,
        use_compound_actions=True,
    )

    assert len(fake_make_env) == 1
    call = fake_make_env[0]
    assert call["scenario"] == "deadly_corridor"
    assert call["doom_skill"] == 3
    assert call["num_bots"] == 0
    assert call["frame_skip"] == 4
    assert call["num_stack"] == 4
    assert call["resize_shape"] == (84, 84)
    assert call["use_compound_actions"] is True


def test_record_video_writes_env_config_to_json(
    tmp_path, fake_make_env, fake_imageio,
) -> None:
    """The episodes.json sidecar must capture the env-config so readers
    can tell at a glance which difficulty the clips were recorded at."""
    out_path = tmp_path / "clip.mp4"

    sb3_utils._record_video(
        SimpleNamespace(predict=_FakeModel().predict),  # type: ignore[arg-type]
        "deathmatch",
        out_path,
        fps=20,
        n_episodes=2,
        doom_skill=5,
        num_bots=8,
    )

    stats_path = out_path.with_name(out_path.stem + "_episodes.json")
    assert stats_path.exists()
    payload = json.loads(stats_path.read_text())
    assert payload["fps"] == 20
    assert payload["env_config"] == {
        "scenario": "deathmatch",
        "doom_skill": 5,
        "num_bots": 8,
        "resize_shape": [84, 84],
        "frame_skip": 4,
        "num_stack": 4,
        "use_compound_actions": True,
    }
    assert len(payload["episodes"]) == 2
    for ep in payload["episodes"]:
        # The fake env terminates via "goal_reached" after 3 steps.
        assert ep["termination"] == "goal_reached"
        assert ep["length_steps"] == 3


def test_record_video_defaults_skill_to_none(
    tmp_path, fake_make_env, fake_imageio,
) -> None:
    """When no skill is provided, the JSON should reflect that (None) —
    which is the signal that the underlying cfg default is in effect."""
    out_path = tmp_path / "clip.mp4"
    sb3_utils._record_video(
        SimpleNamespace(predict=_FakeModel().predict),  # type: ignore[arg-type]
        "basic",
        out_path,
        n_episodes=1,
    )
    payload = json.loads(
        out_path.with_name(out_path.stem + "_episodes.json").read_text(),
    )
    assert payload["env_config"]["doom_skill"] is None
    assert payload["env_config"]["num_bots"] == 0
    assert fake_make_env[0]["doom_skill"] is None
