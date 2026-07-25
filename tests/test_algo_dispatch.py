"""Algorithm-name -> SB3 class dispatch.

The dispatch used to be written out at three call sites, two of which ended in
a silent ``else: DQN``. That meant ``evaluate.load_run`` sent every
``recurrent_ppo`` run through ``DQN.load`` — notebook 05 analyses
``["dqn", "ppo", "recurrent_ppo"]``, so a third of its runs could not be loaded
at all — and ``train_sb3``'s video step would have mis-loaded at the very end
of a multi-hour run.

The save/load round-trip below is the regression guard: it writes a real
checkpoint plus the ``config.json`` that ``load_run`` reads, then loads it back
through the public path and checks the class that comes out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import DQN, PPO

from rl_doom.sb3_utils import resolve_algo_class


def _dummy_vec_env():
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _DummyImg(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
        action_space = gym.spaces.Discrete(3)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    return DummyVecEnv([_DummyImg])


def _write_run(tmp_path: Path, algo: str, model: Any) -> Path:
    """Lay out the minimal run directory that ``load_run`` expects."""
    run_dir = tmp_path / f"run_{algo}"
    (run_dir / "checkpoints").mkdir(parents=True)
    model.save(str(run_dir / "checkpoints" / "final.zip"))
    (run_dir / "config.json").write_text(json.dumps({"algo": algo}))
    return run_dir


def test_resolve_algo_class_known_names() -> None:
    assert resolve_algo_class("ppo") is PPO
    assert resolve_algo_class("dqn") is DQN
    assert resolve_algo_class("recurrent_ppo").__name__ == "RecurrentPPO"


def test_resolve_algo_class_is_case_insensitive() -> None:
    assert resolve_algo_class("PPO") is PPO
    assert resolve_algo_class("Recurrent_PPO").__name__ == "RecurrentPPO"


def test_resolve_algo_class_rejects_unknown() -> None:
    """An unrecognised name must raise, not fall back to a default class."""
    with pytest.raises(ValueError, match="Unknown algo 'dreamer'"):
        resolve_algo_class("dreamer")
    with pytest.raises(ValueError, match="Unknown algo"):
        resolve_algo_class("")


@pytest.mark.parametrize("algo", ["ppo", "dqn", "recurrent_ppo"])
def test_load_run_round_trips_every_algorithm(tmp_path: Path, algo: str) -> None:
    """A checkpoint written by each algorithm must load back as that class."""
    if algo == "recurrent_ppo":
        pytest.importorskip("sb3_contrib")

    from rl_doom.evaluate import load_run

    expected_cls = resolve_algo_class(algo)
    vec_env = _dummy_vec_env()
    if algo == "dqn":
        model: Any = DQN("CnnPolicy", vec_env, buffer_size=10, device="cpu", seed=0)
    elif algo == "ppo":
        model = PPO("CnnPolicy", vec_env, n_steps=8, batch_size=8, device="cpu", seed=0)
    else:
        # ``expected_cls`` is typed ``type[BaseAlgorithm]``, which does not
        # carry RecurrentPPO's constructor signature; the cast keeps the
        # keyword arguments checkable-by-eye without mypy rejecting them.
        recurrent_cls: Any = expected_cls
        model = recurrent_cls(
            "CnnLstmPolicy", vec_env, n_steps=8, batch_size=8, device="cpu", seed=0,
        )

    run_dir = _write_run(tmp_path, algo, model)
    loaded = load_run(run_dir, device="cpu")

    assert isinstance(loaded, expected_cls), (
        f"{algo} checkpoint loaded as {type(loaded).__name__}, expected "
        f"{expected_cls.__name__}"
    )
    # The shim contract downstream notebooks rely on.
    assert callable(getattr(loaded, "predict", None))
