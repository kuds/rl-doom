"""Unit tests for DQN and PPO agents that don't require a live env."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_doom.agents.dqn import DQNAgent
from rl_doom.agents.ppo import PPOAgent
from rl_doom.sb3_utils import policy_kwargs_from_config

OBS_SHAPE = (4, 84, 84)
N_ACTIONS = 3


def _random_obs() -> np.ndarray:
    return np.random.randint(0, 256, size=OBS_SHAPE, dtype=np.uint8)


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

def test_dqn_select_action_in_range() -> None:
    agent = DQNAgent(OBS_SHAPE, N_ACTIONS)
    action = agent.select_action(_random_obs(), epsilon=0.0)
    assert 0 <= action < N_ACTIONS


def test_dqn_select_action_epsilon_one_is_random() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    agent = DQNAgent(OBS_SHAPE, N_ACTIONS)
    actions = {agent.select_action(_random_obs(), epsilon=1.0) for _ in range(50)}
    # With 50 samples over 3 actions we should see more than one distinct value.
    assert len(actions) > 1


def test_dqn_train_step_returns_scalar_loss() -> None:
    agent = DQNAgent(OBS_SHAPE, N_ACTIONS)
    batch_size = 4
    batch = {
        "obs": np.random.rand(batch_size, *OBS_SHAPE).astype(np.float32),
        "actions": np.random.randint(0, N_ACTIONS, size=batch_size),
        "rewards": np.random.rand(batch_size).astype(np.float32),
        "next_obs": np.random.rand(batch_size, *OBS_SHAPE).astype(np.float32),
        "dones": np.zeros(batch_size, dtype=np.float32),
    }
    loss = agent.train_step(batch)
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_dqn_save_load_roundtrip(tmp_path) -> None:
    agent = DQNAgent(OBS_SHAPE, N_ACTIONS)
    ckpt = tmp_path / "dqn.pt"
    agent.save(str(ckpt))

    loaded = DQNAgent(OBS_SHAPE, N_ACTIONS)
    loaded.load(str(ckpt))

    for p1, p2 in zip(agent.policy_net.parameters(), loaded.policy_net.parameters()):
        assert torch.equal(p1, p2)
    # Target net must also be synced after load.
    for p1, p2 in zip(loaded.policy_net.parameters(), loaded.target_net.parameters()):
        assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def test_ppo_select_action_returns_triple() -> None:
    agent = PPOAgent(OBS_SHAPE, N_ACTIONS)
    action, log_prob, value = agent.select_action(_random_obs())
    assert 0 <= action < N_ACTIONS
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_ppo_compute_gae_shapes_and_sanity() -> None:
    n = 5
    rewards = np.ones(n, dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.float32)
    adv, ret = PPOAgent._compute_gae(
        rewards, values, dones, last_value=0.0, gamma=0.99, gae_lambda=0.95,
    )
    assert adv.shape == (n,)
    assert ret.shape == (n,)
    # With all-positive rewards and zero baseline, returns should be
    # positive & monotone-decreasing.
    assert np.all(ret > 0)
    assert ret[0] >= ret[-1]


def test_ppo_compute_gae_handles_terminal_states() -> None:
    # A 'done' should zero out bootstrapping past that index.
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dones = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    adv, _ = PPOAgent._compute_gae(
        rewards, values, dones, last_value=10.0, gamma=0.99, gae_lambda=1.0,
    )
    # Step 1 is terminal, so its advantage should equal its immediate reward (delta=1-0).
    assert np.isclose(adv[1], 1.0)


def test_ppo_update_runs_and_returns_stats() -> None:
    agent = PPOAgent(OBS_SHAPE, N_ACTIONS)
    n = 16
    rollout = {
        "obs": np.random.rand(n, *OBS_SHAPE).astype(np.float32),
        "actions": np.random.randint(0, N_ACTIONS, size=n),
        "log_probs": np.zeros(n, dtype=np.float32),
        "values": np.zeros(n, dtype=np.float32),
        "rewards": np.random.rand(n).astype(np.float32),
        "dones": np.zeros(n, dtype=np.float32),
        "last_value": 0.0,
    }
    stats = agent.update(rollout, n_epochs=2, batch_size=8)
    for key in ("policy_loss", "value_loss", "entropy", "clip_fraction"):
        assert key in stats
        assert np.isfinite(stats[key])


def test_ppo_save_load_roundtrip(tmp_path) -> None:
    agent = PPOAgent(OBS_SHAPE, N_ACTIONS)
    ckpt = tmp_path / "ppo.pt"
    agent.save(str(ckpt))

    loaded = PPOAgent(OBS_SHAPE, N_ACTIONS)
    loaded.load(str(ckpt))

    for p1, p2 in zip(agent.network.parameters(), loaded.network.parameters()):
        assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# Recurrent PPO (sb3-contrib) — model-build smoke tests
#
# The SB3-side training path is exercised end-to-end by the notebook; these
# tests just guarantee that the YAML -> policy_kwargs translation and the
# _build_model dispatch produce a usable RecurrentPPO instance with the right
# policy class. Skipped if sb3-contrib isn't installed in the test env.
# ---------------------------------------------------------------------------

sb3_contrib = pytest.importorskip("sb3_contrib")


def test_policy_kwargs_passes_lstm_knobs() -> None:
    cfg = {
        "features_dim": 256,
        "net_arch": [128],
        "lstm_hidden_size": 128,
        "n_lstm_layers": 2,
    }
    pk = policy_kwargs_from_config(cfg)
    assert pk["features_extractor_kwargs"]["features_dim"] == 256
    assert pk["net_arch"] == [128]
    assert pk["lstm_hidden_size"] == 128
    assert pk["n_lstm_layers"] == 2


def test_policy_kwargs_omits_lstm_knobs_when_absent() -> None:
    cfg = {"features_dim": 512, "net_arch": []}
    pk = policy_kwargs_from_config(cfg)
    assert "lstm_hidden_size" not in pk
    assert "n_lstm_layers" not in pk


def test_build_model_recurrent_ppo_returns_recurrent_ppo_instance() -> None:
    # Lazy imports keep the SB3 dependency out of the module-import critical
    # path for tests that don't need it (e.g. agent unit tests above).
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
        tensorboard_log="/tmp/rl_doom_test_tb",
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
