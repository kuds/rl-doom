"""Tests for :mod:`rl_doom.self_play`.

Each test reuses the ``fake_vizdoom`` fixture from ``test_multiplayer_env``
so the multiplayer base env works without the native ViZDoom binary. The
self-play layer is exercised with a fake loader that fabricates a simple
policy object from a zip path, which keeps these tests independent of SB3's
``PPO.load`` as well.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal fake policy + loader
# ---------------------------------------------------------------------------


class _FakePolicy:
    """Stand-in for an SB3 model exposing just ``.predict`` + ``.save``.

    ``fixed_action`` makes opponent behaviour deterministic, which lets the
    wrapper test verify that opponent actions are actually routed to the
    correct seats in the merged action dict.
    """

    def __init__(self, fixed_action: int = 3) -> None:
        self.fixed_action = fixed_action
        self.predict_calls: list[np.ndarray] = []

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, None]:
        self.predict_calls.append(observation)
        return self.fixed_action, None

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake-ckpt")


def _fake_loader_factory(fixed_action: int = 3):
    """Return a loader that ignores the path and returns a fresh _FakePolicy."""

    def _load(path: str) -> _FakePolicy:
        return _FakePolicy(fixed_action=fixed_action)

    return _load


# ---------------------------------------------------------------------------
# OpponentPool
# ---------------------------------------------------------------------------


def test_opponent_pool_empty_returns_random_policy(tmp_path: Path) -> None:
    from rl_doom.self_play import OpponentPool, RandomPolicy

    pool = OpponentPool(tmp_path / "opponents")
    action_space: gym.spaces.Discrete = gym.spaces.Discrete(5)
    sampled = pool.sample(action_space=action_space)
    assert isinstance(sampled, RandomPolicy)
    # Sampled actions must come from the bound space.
    for _ in range(20):
        action, state = sampled.predict(np.zeros((4, 84, 84), dtype=np.uint8))
        assert 0 <= action < 5
        assert state is None


def test_opponent_pool_empty_rejects_when_no_fallback(tmp_path: Path) -> None:
    from rl_doom.self_play import OpponentPool

    pool = OpponentPool(tmp_path / "opponents", random_fallback=False)
    with pytest.raises(RuntimeError, match="empty"):
        pool.sample(action_space=gym.spaces.Discrete(5))


def test_opponent_pool_samples_from_snapshots_and_caches(tmp_path: Path) -> None:
    from rl_doom.self_play import OpponentPool

    snap_dir = tmp_path / "opponents"
    snap_dir.mkdir()
    (snap_dir / "step_000000001.zip").write_bytes(b"a")
    (snap_dir / "step_000000002.zip").write_bytes(b"b")

    pool = OpponentPool(snap_dir, loader=_fake_loader_factory())
    assert len(pool) == 2
    # Seeded RNG makes ``choice`` deterministic across runs.
    rng = np.random.default_rng(0)
    policy = pool.sample(rng=rng)
    assert isinstance(policy, _FakePolicy)

    # Cache should return the same object on a second hit with the same path.
    rng2 = np.random.default_rng(0)
    policy2 = pool.sample(rng=rng2)
    assert policy2 is policy


def test_opponent_pool_add_snapshot_persists_model(tmp_path: Path) -> None:
    from rl_doom.self_play import OpponentPool

    pool = OpponentPool(tmp_path / "opponents")
    model = _FakePolicy()
    path = pool.add_snapshot(model, step=12345)
    assert path.exists()
    assert path.name == "step_000012345.zip"
    assert len(pool) == 1


# ---------------------------------------------------------------------------
# SelfPlayWrapper
# ---------------------------------------------------------------------------


def test_wrapper_exposes_learner_spaces_only(
    fake_vizdoom: types.ModuleType, tmp_path: Path,
) -> None:
    from rl_doom.multiplayer_env import DEATHMATCH_ACTIONS, make_1v1_env
    from rl_doom.self_play import OpponentPool, SelfPlayWrapper

    pool = OpponentPool(tmp_path / "opponents")
    base = make_1v1_env(resize_shape=(84, 84), num_stack=4)
    try:
        env = SelfPlayWrapper(base, learner_agent="player_0", opponent_pool=pool)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (4, 84, 84)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert int(env.action_space.n) == len(DEATHMATCH_ACTIONS)
        obs, info = env.reset(seed=0)
        assert obs.shape == (4, 84, 84)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_wrapper_rejects_unknown_learner_agent(
    fake_vizdoom: types.ModuleType, tmp_path: Path,
) -> None:
    from rl_doom.multiplayer_env import make_1v1_env
    from rl_doom.self_play import OpponentPool, SelfPlayWrapper

    pool = OpponentPool(tmp_path / "opponents")
    base = make_1v1_env(resize_shape=(84, 84), num_stack=1)
    try:
        with pytest.raises(ValueError, match="not in"):
            SelfPlayWrapper(base, learner_agent="player_42", opponent_pool=pool)
    finally:
        base.close()


def test_wrapper_routes_opponent_action_and_forwards_learner_reward(
    fake_vizdoom: types.ModuleType, tmp_path: Path,
) -> None:
    """Verify the wrapper (a) queries the opponent policy with the opponent's
    own observation, (b) merges that action under the right seat name when
    calling the base env, and (c) reports back only the learner's reward."""
    from rl_doom.multiplayer_env import make_1v1_env
    from rl_doom.self_play import OpponentPool, SelfPlayWrapper

    # player_0 scores +2 frags, player_1 dies once -> learner reward +2.
    fake_vizdoom.DoomGame.scripted = [[(2, 0)], [(0, 1)]]

    snap_dir = tmp_path / "opponents"
    snap_dir.mkdir()
    (snap_dir / "step_000000001.zip").write_bytes(b"x")
    pool = OpponentPool(snap_dir, loader=_fake_loader_factory(fixed_action=5))

    base = make_1v1_env(resize_shape=(84, 84), num_stack=1)
    env = SelfPlayWrapper(base, learner_agent="player_0", opponent_pool=pool)
    try:
        env.reset(seed=0)
        obs, reward, term, trunc, info = env.step(action=7)
        assert reward == 2.0
        assert term is False
        assert trunc is False
        assert obs.shape == (84, 84)
        # Opponent's game should have received a button vector whose action
        # index is 5 (from the fake policy's fixed_action).
        opponent_game: Any = base._games["player_1"]
        submitted = opponent_game.actions_submitted[-1]
        expected = base._action_tables["player_1"][5]
        assert submitted == expected
        assert info["frags"] == 2
    finally:
        env.close()


def test_wrapper_uses_random_fallback_when_pool_empty(
    fake_vizdoom: types.ModuleType, tmp_path: Path,
) -> None:
    """With an empty pool the opponent should be a ``RandomPolicy`` drawn from
    the base env's action space."""
    from rl_doom.multiplayer_env import make_1v1_env
    from rl_doom.self_play import OpponentPool, RandomPolicy, SelfPlayWrapper

    pool = OpponentPool(tmp_path / "opponents")
    base = make_1v1_env(resize_shape=(84, 84), num_stack=1)
    env = SelfPlayWrapper(base, learner_agent="player_0", opponent_pool=pool)
    try:
        env.reset(seed=0)
        assert isinstance(env._opp_policies["player_1"], RandomPolicy)
        # Action space of the fallback must match the base env's opponent seat.
        assert env._opp_policies["player_1"].action_space is base.action_space("player_1")
    finally:
        env.close()


def test_wrapper_2v2_learner_has_three_opponent_seats(
    fake_vizdoom: types.ModuleType, tmp_path: Path,
) -> None:
    from rl_doom.multiplayer_env import make_2v2_env
    from rl_doom.self_play import OpponentPool, SelfPlayWrapper

    pool = OpponentPool(tmp_path / "opponents")
    base = make_2v2_env(resize_shape=(84, 84), num_stack=1)
    env = SelfPlayWrapper(base, learner_agent="player_0", opponent_pool=pool)
    try:
        env.reset(seed=0)
        assert env.opponent_agents == ["player_1", "player_2", "player_3"]
        # Stepping must run to completion without deadlock and return a
        # single learner observation/reward pair.
        obs, reward, term, trunc, _ = env.step(0)
        assert obs.shape == (84, 84)
        assert isinstance(reward, float)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# SelfPlaySnapshotCallback
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal model surface the callback touches: ``save(path)``."""

    def __init__(self) -> None:
        self.saved: list[str] = []

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"ckpt")
        self.saved.append(path)


def test_snapshot_callback_fires_on_interval(tmp_path: Path) -> None:
    from rl_doom.self_play import OpponentPool, SelfPlaySnapshotCallback

    pool = OpponentPool(tmp_path / "opponents")
    model = _FakeModel()
    cb = SelfPlaySnapshotCallback(pool, snapshot_freq=100)
    cb.model = model  # type: ignore[assignment, unused-ignore]

    # Simulate SB3 stepping through timesteps. ``_on_step`` reads
    # ``num_timesteps`` off the callback (SB3 increments it externally).
    for step in (50, 99, 100, 150, 200, 250):
        cb.num_timesteps = step
        cb._on_step()

    snaps = pool.snapshots()
    # Expect snapshots at steps 100, 200 (150 is less than freq after 100;
    # 250 less than freq after 200). That's 2 writes.
    assert [p.name for p in snaps] == [
        "step_000000100.zip",
        "step_000000200.zip",
    ]


def test_snapshot_callback_evicts_oldest_beyond_max_pool_size(
    tmp_path: Path,
) -> None:
    from rl_doom.self_play import OpponentPool, SelfPlaySnapshotCallback

    pool = OpponentPool(tmp_path / "opponents")
    model = _FakeModel()
    cb = SelfPlaySnapshotCallback(pool, snapshot_freq=10, max_pool_size=2)
    cb.model = model  # type: ignore[assignment, unused-ignore]

    for step in (10, 20, 30, 40):
        cb.num_timesteps = step
        cb._on_step()

    snaps = pool.snapshots()
    assert len(snaps) == 2
    # Oldest two (10, 20) should have been evicted.
    assert [p.name for p in snaps] == [
        "step_000000030.zip",
        "step_000000040.zip",
    ]


def test_snapshot_callback_rejects_non_positive_freq(tmp_path: Path) -> None:
    from rl_doom.self_play import OpponentPool, SelfPlaySnapshotCallback

    pool = OpponentPool(tmp_path / "opponents")
    with pytest.raises(ValueError, match="snapshot_freq"):
        SelfPlaySnapshotCallback(pool, snapshot_freq=0)
