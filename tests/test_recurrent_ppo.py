"""Recurrent PPO (sb3-contrib) model-build smoke test.

This lives in its own module because ``sb3-contrib`` is an optional
dependency. A module-level ``importorskip`` aborts collection of the *whole*
module, so keeping this test alongside the DQN/PPO unit tests meant a CI
environment without sb3-contrib silently skipped those too. One module per
optional dependency keeps the blast radius to the tests that actually need it.

The SB3-side training path is exercised end-to-end by the notebooks; this test
just guarantees that the YAML -> ``policy_kwargs`` translation and the
``_build_model`` dispatch produce a usable ``RecurrentPPO`` with the LSTM knobs
actually applied to the policy.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")


def test_build_model_recurrent_ppo_returns_recurrent_ppo_instance(tmp_path) -> None:
    # Lazy imports keep the SB3 dependency out of the module-import critical
    # path for tests that don't need it (e.g. the agent unit tests).
    import gymnasium as gym
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl_doom.sb3_utils import _build_model

    def _make_env() -> gym.Env:
        # Tiny synthetic image env that matches the obs/action shape DoomEnv
        # exposes (uint8 84x84 stacked frames + small discrete action set),
        # so we can build the model without a ViZDoom binary.
        class _DummyImg(gym.Env):
            observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
            action_space = gym.spaces.Discrete(3)

            def reset(self, *, seed=None, options=None):
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

        return _DummyImg()

    vec_env = DummyVecEnv([_make_env for _ in range(2)])
    hp = {
        "lr": 3e-4,
        "n_steps": 8,
        "batch_size": 8,
        "n_epochs": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "entropy_coef": 0.0,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    model = _build_model(
        "recurrent_ppo",
        vec_env,
        hyperparams=hp,
        tensorboard_log=tmp_path / "tb",
        device="cpu",
        seed=0,
        policy_kwargs={"lstm_hidden_size": 32, "n_lstm_layers": 1},
    )
    assert isinstance(model, RecurrentPPO)
    # Confirms the LSTM kwargs actually reached the policy. ``lstm_actor`` is
    # the per-policy LSTM module on CnnLstmPolicy; checking its hidden_size is
    # a stable way to verify the policy_kwargs plumbing without depending on
    # SB3-contrib's private state-shape attributes.
    lstm_actor = model.policy.lstm_actor
    assert lstm_actor is not None
    # ``getattr`` keeps mypy happy across SB3 versions where
    # ``lstm_actor`` is typed as the generic ``nn.Module | Tensor`` union.
    assert getattr(lstm_actor, "hidden_size", None) == 32
