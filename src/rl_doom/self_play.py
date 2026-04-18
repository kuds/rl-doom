"""Self-play training glue for the multiplayer ViZDoom env.

The training loop trains one **learner seat** against frozen snapshots of
past learners drawn from an :class:`OpponentPool`. A
:class:`SelfPlaySnapshotCallback` periodically writes the live learner's
weights into the pool so the opponent distribution tracks the learner over
time (a minimal population-based / league-style setup).

Design:

* :class:`SelfPlayWrapper` collapses the ``DoomMultiplayerEnv`` ``ParallelEnv``
  into a single-agent Gymnasium env exposing only the learner's observation,
  reward, and termination signal. On every ``step()`` it queries each frozen
  opponent policy with that opponent's own per-player observation, merges the
  resulting actions into the base env's action dict, and advances one tic.
* :class:`OpponentPool` owns a directory of SB3 ``.zip`` snapshots plus a
  caller-supplied ``loader`` (defaults to ``stable_baselines3.PPO.load``).
  When the pool is empty it returns a :class:`RandomPolicy` so the very first
  episodes can still train from something.
* :func:`train_self_play` ties the above to SB3's PPO for a minimal driver
  that notebooks or a CLI can import.

In 2v2 the wrapper is *seat-singular*: one agent is the learner and every
other seat (teammate + two opponents) is driven by a sampled pool member.
That's intentional - it forces the learner to cooperate with an old version
of itself rather than magically predicting its teammate's intent, and it
keeps the training signal on-policy for the learner while still providing a
reactive environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    # Only referenced in type annotations - avoid pulling in ``vizdoom`` at
    # import time so callers that only need ``OpponentPool`` (e.g. tournament
    # scripts) don't require the native binary.
    from rl_doom.multiplayer_env import DoomMultiplayerEnv

# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------


class _Policy(Protocol):
    """Minimum surface area a frozen opponent policy must expose.

    SB3's ``BaseAlgorithm.predict`` satisfies this contract; any duck-typed
    callable that returns an ``(action, state)`` tuple works too.
    """

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = ...,
    ) -> tuple[np.ndarray | int, Any]: ...


class RandomPolicy:
    """Bootstrap policy used while the snapshot pool is still empty.

    Returns uniformly-random actions sampled from the underlying action
    space, mirroring the ``(action, state)`` tuple shape of SB3 models so
    the wrapper's call site doesn't need a branch.
    """

    def __init__(self, action_space: gym.spaces.Discrete) -> None:
        self.action_space = action_space

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, None]:
        return int(self.action_space.sample()), None


# ---------------------------------------------------------------------------
# Opponent pool
# ---------------------------------------------------------------------------


class OpponentPool:
    """Directory-backed pool of SB3 ``.zip`` snapshots.

    Parameters
    ----------
    snapshot_dir : str | Path
        Directory containing opponent snapshots. Created on first write.
    loader : callable
        Function ``(path: str) -> policy`` used to materialise a ``.zip``
        into something with a ``.predict(obs, deterministic=...)`` method.
        Defaults to lazy-import of ``stable_baselines3.PPO.load`` so callers
        that never load snapshots (e.g. pure tests) don't pay the import cost.
    random_fallback : bool
        If ``True`` (default), ``sample()`` returns a :class:`RandomPolicy`
        when no snapshots have been written yet, which is exactly what the
        first few self-play episodes need.
    cache_loaded : bool
        Cache loaded policies in memory keyed by file path so repeated
        sampling of the same snapshot doesn't re-read from disk.
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        *,
        loader: Callable[[str], Any] | None = None,
        random_fallback: bool = True,
        cache_loaded: bool = True,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self._loader = loader
        self._random_fallback = random_fallback
        self._cache_loaded = cache_loaded
        self._cache: dict[Path, Any] = {}

    # --- introspection ---------------------------------------------------

    def snapshots(self) -> list[Path]:
        """Sorted list of snapshot ``.zip`` files currently in the pool."""
        if not self.snapshot_dir.exists():
            return []
        return sorted(self.snapshot_dir.glob("*.zip"))

    def __len__(self) -> int:
        return len(self.snapshots())

    # --- sampling --------------------------------------------------------

    def sample(
        self,
        *,
        action_space: gym.spaces.Discrete | None = None,
        rng: np.random.Generator | None = None,
    ) -> _Policy:
        """Return a policy from the pool.

        When the pool is empty and ``random_fallback`` is set, returns a
        :class:`RandomPolicy` bound to ``action_space``. Otherwise samples a
        snapshot uniformly at random and returns the loaded policy.
        """
        snaps = self.snapshots()
        if not snaps:
            if not self._random_fallback:
                raise RuntimeError(
                    f"Opponent pool at {self.snapshot_dir} is empty and "
                    "random_fallback=False.",
                )
            if action_space is None:
                raise ValueError(
                    "action_space required for RandomPolicy fallback.",
                )
            return RandomPolicy(action_space)

        rng = rng if rng is not None else np.random.default_rng()
        path = Path(str(rng.choice([str(p) for p in snaps])))
        return self._load(path)

    def _load(self, path: Path) -> Any:
        if self._cache_loaded and path in self._cache:
            return self._cache[path]
        loader = self._loader or _default_loader()
        policy = loader(str(path))
        if self._cache_loaded:
            self._cache[path] = policy
        return policy

    # --- writes ---------------------------------------------------------

    def add_snapshot(self, model: Any, step: int) -> Path:
        """Persist ``model`` as ``step_{step}.zip`` under the pool dir."""
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"step_{step:09d}.zip"
        model.save(str(path))
        return path


def _default_loader() -> Callable[[str], Any]:
    """Import ``stable_baselines3.PPO.load`` lazily.

    Keeps ``self_play`` importable in environments where SB3 is installed but
    not yet initialised, and avoids a hard import at module load time.
    """
    from stable_baselines3 import PPO

    return PPO.load


# ---------------------------------------------------------------------------
# Gym env wrapper
# ---------------------------------------------------------------------------


class SelfPlayWrapper(gym.Env):
    """Expose a single seat of a :class:`DoomMultiplayerEnv` as a Gym env.

    Every non-learner seat is driven by a policy sampled from
    ``opponent_pool`` at episode start. Samples a *fresh* opponent per reset
    so the learner doesn't overfit to a single snapshot within one training
    chunk.

    Parameters
    ----------
    base_env : DoomMultiplayerEnv
        The underlying parallel env. Ownership transfers to the wrapper -
        ``close()`` propagates.
    learner_agent : str
        Which of ``base_env.possible_agents`` the learner controls (e.g.
        ``"player_0"`` for 1v1, any of ``"player_0"``..``"player_3"`` for 2v2).
    opponent_pool : OpponentPool
        Source of frozen opponent policies.
    seed : int | None
        Seed for the wrapper's opponent-selection RNG; deterministic opponent
        choice is useful for reproducibility in evaluation scripts.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        base_env: "DoomMultiplayerEnv",
        *,
        learner_agent: str = "player_0",
        opponent_pool: OpponentPool,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if learner_agent not in base_env.possible_agents:
            raise ValueError(
                f"learner_agent {learner_agent!r} not in "
                f"{base_env.possible_agents!r}",
            )
        self.base_env = base_env
        self.learner_agent = learner_agent
        self.opponent_agents = [
            a for a in base_env.possible_agents if a != learner_agent
        ]
        self.opponent_pool = opponent_pool
        self.observation_space = base_env.observation_space(learner_agent)
        self.action_space = base_env.action_space(learner_agent)

        self._rng = np.random.default_rng(seed)
        self._opp_policies: dict[str, _Policy] = {}
        # Keep a rolling copy of the last multi-agent observation so we can
        # feed each opponent its own POV frame on the next ``step()``.
        self._last_obs: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs_dict, info_dict = self.base_env.reset(seed=seed, options=options)
        # Sample a fresh opponent for each non-learner seat. Bound to each
        # opponent's own action space so a ``RandomPolicy`` fallback stays
        # well-typed even if seats ever have heterogeneous action sets.
        self._opp_policies = {
            a: self.opponent_pool.sample(
                action_space=self.base_env.action_space(a),
                rng=self._rng,
            )
            for a in self.opponent_agents
        }
        self._last_obs = dict(obs_dict)
        return obs_dict[self.learner_agent], info_dict.get(self.learner_agent, {})

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        merged: dict[str, int] = {self.learner_agent: int(action)}
        for agent in self.opponent_agents:
            policy = self._opp_policies[agent]
            opp_action, _ = policy.predict(
                self._last_obs[agent], deterministic=False,
            )
            # SB3 returns numpy arrays for Discrete actions; coerce to plain int
            # so the base env's ``_action_tables`` indexing is type-safe.
            merged[agent] = int(np.asarray(opp_action).item())

        obs_dict, rewards, terms, truncs, infos = self.base_env.step(merged)
        self._last_obs = dict(obs_dict)
        return (
            obs_dict[self.learner_agent],
            float(rewards[self.learner_agent]),
            bool(terms[self.learner_agent]),
            bool(truncs[self.learner_agent]),
            infos[self.learner_agent],
        )

    def render(self) -> np.ndarray | None:  # type: ignore[override]
        # Return just the learner's POV frame; callers that want all POVs can
        # drill into ``self.base_env.render()``.
        frames = self.base_env.render()
        return frames.get(self.learner_agent) if isinstance(frames, dict) else None

    def close(self) -> None:
        self.base_env.close()


# ---------------------------------------------------------------------------
# SB3 callback for periodic snapshots
# ---------------------------------------------------------------------------


class SelfPlaySnapshotCallback(BaseCallback):
    """Write the learner's current weights into an :class:`OpponentPool`.

    Fires every ``snapshot_freq`` env steps (not gradient steps). The first
    snapshot happens after the initial ``snapshot_freq`` steps so the pool
    has at least one non-bootstrap opponent before serious training starts.

    Parameters
    ----------
    pool : OpponentPool
        Destination pool.
    snapshot_freq : int
        Minimum env-step gap between consecutive snapshots.
    max_pool_size : int | None
        If given, evicts the oldest snapshot on write once the pool grows
        past this many entries. Useful for bounding disk / load time in long
        runs; ``None`` keeps the full league.
    """

    def __init__(
        self,
        pool: OpponentPool,
        snapshot_freq: int,
        *,
        max_pool_size: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if snapshot_freq <= 0:
            raise ValueError(f"snapshot_freq must be > 0, got {snapshot_freq}")
        self.pool = pool
        self.snapshot_freq = snapshot_freq
        self.max_pool_size = max_pool_size
        self._last_snapshot_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_snapshot_step < self.snapshot_freq:
            return True
        self._last_snapshot_step = self.num_timesteps
        path = self.pool.add_snapshot(self.model, self.num_timesteps)
        if self.verbose:
            print(f"[self-play] snapshot @ step {self.num_timesteps} -> {path}")
        if self.max_pool_size is not None:
            self._evict_oldest()
        return True

    def _evict_oldest(self) -> None:
        snaps = self.pool.snapshots()
        excess = len(snaps) - int(self.max_pool_size or 0)
        for old in snaps[:excess]:
            try:
                old.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def build_self_play_env(
    *,
    mode: str = "1v1",
    learner_agent: str = "player_0",
    opponent_pool: OpponentPool,
    seed: int | None = None,
    **env_kwargs: Any,
) -> SelfPlayWrapper:
    """Construct a :class:`SelfPlayWrapper` for the requested match format.

    ``mode`` is ``"1v1"`` or ``"2v2"``. Extra ``env_kwargs`` are forwarded to
    :func:`make_1v1_env` / :func:`make_2v2_env` (e.g. ``resize_shape``,
    ``num_stack``, ``frame_skip``).
    """
    # Deferred import: only this helper pulls in vizdoom via multiplayer_env;
    # the rest of ``self_play`` (pool + callback) can be used standalone.
    from rl_doom.multiplayer_env import make_1v1_env, make_2v2_env

    if mode == "1v1":
        base_env = make_1v1_env(**env_kwargs)
    elif mode == "2v2":
        base_env = make_2v2_env(**env_kwargs)
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected '1v1' or '2v2'.")
    return SelfPlayWrapper(
        base_env,
        learner_agent=learner_agent,
        opponent_pool=opponent_pool,
        seed=seed,
    )


def train_self_play(
    *,
    run_dir: str | Path,
    mode: str = "1v1",
    total_timesteps: int = 1_000_000,
    snapshot_freq: int = 50_000,
    max_pool_size: int | None = 20,
    learner_agent: str = "player_0",
    seed: int = 0,
    env_kwargs: dict[str, Any] | None = None,
    ppo_kwargs: dict[str, Any] | None = None,
    verbose: int = 0,
) -> Path:
    """End-to-end PPO self-play driver.

    Builds a snapshot pool at ``run_dir/opponents``, wraps the requested
    multiplayer env in :class:`SelfPlayWrapper`, and trains a fresh PPO
    agent that periodically adds its weights to the pool via
    :class:`SelfPlaySnapshotCallback`. The final model is saved to
    ``run_dir/checkpoints/final.zip``.

    Intentionally minimal: plot/video generation lives in
    :mod:`rl_doom.sb3_utils` and can be composed on top for a full
    artifact-producing run. This function only does the training loop.

    ``verbose`` is forwarded to both the PPO model and the snapshot callback.
    Default 0 keeps Colab notebook output tidy; set 1 for SB3's rolling
    training table and per-snapshot log lines.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    run_dir = Path(run_dir)
    opp_dir = run_dir / "opponents"
    ckpt_dir = run_dir / "checkpoints"
    for d in (opp_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    pool = OpponentPool(opp_dir)

    def _make_env() -> gym.Env:
        env = build_self_play_env(
            mode=mode,
            learner_agent=learner_agent,
            opponent_pool=pool,
            seed=seed,
            **(env_kwargs or {}),
        )
        return Monitor(env)

    # Single-env training: each multiplayer env already fans out N ViZDoom
    # processes, so ``n_envs=1`` is the right default on Colab. Callers can
    # wrap in ``DummyVecEnv`` with multiple envs if their machine affords it.
    vec_env = DummyVecEnv([_make_env])

    defaults: dict[str, Any] = dict(
        learning_rate=2.5e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    defaults.update(ppo_kwargs or {})
    model = PPO(
        "CnnPolicy",
        vec_env,
        seed=seed,
        verbose=verbose,
        tensorboard_log=str(run_dir / "tensorboard"),
        **defaults,
    )

    callback = SelfPlaySnapshotCallback(
        pool,
        snapshot_freq=snapshot_freq,
        max_pool_size=max_pool_size,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)

    final_path = ckpt_dir / "final.zip"
    model.save(str(final_path))
    vec_env.close()
    return final_path
