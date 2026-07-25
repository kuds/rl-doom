"""``train_sb3`` must not lose a run to a mid-training failure.

Everything that makes a run useful — the model, metrics, figures, video,
config.json, stage_summary.txt — is written after ``model.learn()`` returns.
While that call was unguarded, an exception discarded all of it: on a 2.5M-step
Colab run, a failure at step 2.4M left nothing but periodic checkpoint zips that
nothing in the repo reads. ViZDoom's per-env native processes leaked too, since
teardown was on the success path only.

These tests use a synthetic image env rather than ViZDoom so they stay fast, and
drive failure by raising from a callback — the same place a real OOM or a
preemption surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.callbacks import BaseCallback

from rl_doom.sb3_utils import train_sb3


class _Boom(RuntimeError):
    """Distinctive error so we can assert it propagated unchanged."""


class _RaiseAtStep(BaseCallback):
    """Raise partway through training, mimicking an OOM or preemption."""

    def __init__(self, at_step: int, exc: BaseException) -> None:
        super().__init__(verbose=0)
        self._at_step = at_step
        self._exc = exc

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._at_step:
            raise self._exc
        return True


class _TinyImgEnv(gym.Env):
    """Minimal env with DoomEnv's observation contract, no binary required."""

    observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
    action_space = gym.spaces.Discrete(3)

    def __init__(self) -> None:
        self._step = 0
        self.closed = False

    def reset(self, *, seed: int | None = None, options: Any = None):
        self._step = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._step += 1
        terminated = self._step >= 20
        info = {"termination_reason": "goal_reached"} if terminated else {}
        return self.observation_space.sample(), 1.0, terminated, False, info

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def patched_env(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Swap ``make_sb3_env`` for the synthetic env; record every vec env built."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _RecordingVecEnv(DummyVecEnv):
        """DummyVecEnv that remembers whether it was closed.

        Stands in for the fact that each real env owns a native ViZDoom
        process: if ``close()`` is skipped, that process is leaked.
        """

        closed = False

        def close(self) -> None:
            super().close()
            self.closed = True

    built: list[Any] = []

    def _fake_make_sb3_env(scenario: str, *, n_envs: int = 1, monitor_dir=None, **kw):
        def _thunk():
            env: gym.Env = _TinyImgEnv()
            if monitor_dir is not None:
                Path(monitor_dir).mkdir(parents=True, exist_ok=True)
                env = Monitor(env, filename=str(Path(monitor_dir) / "monitor_0"))
            return env

        vec = _RecordingVecEnv([_thunk for _ in range(n_envs)])
        built.append(vec)
        return vec

    import rl_doom.env

    monkeypatch.setattr(rl_doom.env, "make_sb3_env", _fake_make_sb3_env)
    return built


def _train(run_dir: Path, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = dict(
        algo="ppo",
        scenario="basic",
        run_dir=run_dir,
        hyperparams={
            "lr": 3e-4, "n_steps": 32, "batch_size": 16, "n_epochs": 1,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
            "entropy_coef": 0.0, "value_coef": 0.5, "max_grad_norm": 0.5,
        },
        seed=0,
        total_timesteps=128,
        n_envs=1,
        eval_freq=10_000,       # keep eval out of the way
        eval_episodes=1,
        checkpoint_freq=10_000,
        record_video=False,
        device="cpu",
    )
    kwargs.update(overrides)
    return train_sb3(**kwargs)


def test_successful_run_writes_artifacts_and_closes_envs(
    tmp_path: Path, patched_env: list[Any],
) -> None:
    """Baseline: the happy path still produces what it always did."""
    run_dir = tmp_path / "run"
    result = _train(run_dir)

    assert (run_dir / "checkpoints" / "final.zip").exists()
    assert (run_dir / "metrics" / "training.npz").exists()
    assert result["run_dir"] == run_dir
    assert len(patched_env) == 2  # train + eval
    for vec in patched_env:
        assert vec.closed


def test_failure_still_saves_the_model(tmp_path: Path, patched_env: list[Any]) -> None:
    run_dir = tmp_path / "run"

    with pytest.raises(_Boom, match="simulated OOM"):
        _train(
            run_dir,
            total_timesteps=1_000,
            extra_callbacks=[_RaiseAtStep(64, _Boom("simulated OOM"))],
        )

    assert (run_dir / "checkpoints" / "final.zip").exists(), (
        "weights were lost — the whole point of the salvage path"
    )


def test_failure_closes_both_envs(tmp_path: Path, patched_env: list[Any]) -> None:
    """ViZDoom spawns a process per env; a failed run must not leak them."""
    run_dir = tmp_path / "run"
    with pytest.raises(_Boom):
        _train(
            run_dir,
            total_timesteps=1_000,
            extra_callbacks=[_RaiseAtStep(64, _Boom("boom"))],
        )

    assert len(patched_env) == 2
    for vec in patched_env:
        assert vec.closed, "a failed run leaked an env"


def test_failure_marks_run_status_and_records_the_error(
    tmp_path: Path, patched_env: list[Any],
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"status": "running"}))

    with pytest.raises(_Boom):
        _train(
            run_dir,
            total_timesteps=1_000,
            extra_callbacks=[_RaiseAtStep(64, _Boom("simulated OOM"))],
        )

    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["status"] == "failed"
    assert "_Boom: simulated OOM" in cfg["error"]
    assert cfg["completed_timesteps"] < cfg["requested_timesteps"]


def test_keyboard_interrupt_is_salvaged_and_reraised(
    tmp_path: Path, patched_env: list[Any],
) -> None:
    """Ctrl-C on an overlong run must not throw the run away.

    KeyboardInterrupt derives from BaseException, so an ``except Exception``
    would miss the single most likely way a human ends a long run.
    """
    run_dir = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        _train(
            run_dir,
            total_timesteps=1_000,
            extra_callbacks=[_RaiseAtStep(64, KeyboardInterrupt())],
        )

    assert (run_dir / "checkpoints" / "final.zip").exists()
    for vec in patched_env:
        assert vec.closed
