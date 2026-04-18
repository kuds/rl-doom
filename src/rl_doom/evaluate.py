"""Evaluation and gameplay recording utilities.

Supports both the legacy hand-rolled agents (:class:`rl_doom.agents.dqn.DQNAgent`,
:class:`rl_doom.agents.ppo.PPOAgent`) and Stable-Baselines3 models that expose
``.predict(obs, deterministic=True)``. ``_select_action`` dispatches on
capability, so callers can hand either one to :func:`evaluate_agent` and
:func:`record_episode`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from rl_doom.agents.dqn import DQNAgent
from rl_doom.agents.ppo import PPOAgent


def _select_action(agent: Any, obs: np.ndarray, *, epsilon: float = 0.0) -> int:
    """Dispatch to the agent's action API based on its class / duck-typing."""
    if isinstance(agent, DQNAgent):
        return agent.select_action(obs, epsilon=epsilon)
    if isinstance(agent, PPOAgent):
        action, _, _ = agent.select_action(obs)
        return action
    # Duck-typed Stable-Baselines3 models: they expose ``predict``.
    predict = getattr(agent, "predict", None)
    if callable(predict):
        action, _ = predict(obs, deterministic=True)
        # SB3 returns a numpy array for vectorised obs and a scalar for single.
        return int(np.asarray(action).flatten()[0])
    raise TypeError(f"Unsupported agent type: {type(agent).__name__}")


def evaluate_agent(
    agent: DQNAgent | PPOAgent,
    make_env: Callable[[], Any],
    n_episodes: int = 10,
) -> np.ndarray:
    """Run the agent for *n_episodes* and return per-episode total rewards."""
    env = make_env()
    rewards: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward, done = 0.0, False
        while not done:
            action = _select_action(agent, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        rewards.append(total_reward)
    env.close()
    return np.array(rewards)


def record_episode(
    agent: DQNAgent | PPOAgent,
    make_env: Callable[[], Any],
    *,
    epsilon: float = 0.0,
    max_steps: int = 5_000,
) -> list[np.ndarray]:
    """Record RGB frames from a single episode.

    Parameters
    ----------
    agent : DQNAgent or PPOAgent
        Trained agent.
    make_env : callable
        Factory that returns a wrapped environment.
    epsilon : float
        Exploration rate (only used for DQN agents).
    max_steps : int
        Safety cap to prevent infinite episodes.

    Returns
    -------
    list[np.ndarray]
        List of RGB frames (H, W, 3) suitable for display or GIF export.
    """
    env = make_env()

    # We also need the raw RGB frames for recording.  Unwrap to the
    # base DoomEnv and grab its render output alongside the wrapped env.
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    obs, _ = env.reset()
    frames: list[np.ndarray] = [base_env.render()]
    done = False
    step = 0

    while not done and step < max_steps:
        action = _select_action(agent, obs, epsilon=epsilon)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(base_env.render())
        done = terminated or truncated
        step += 1

    env.close()
    return frames


def save_video(
    frames: list[np.ndarray],
    path: str | Path,
    *,
    fps: int = 20,
) -> Path:
    """Save a list of RGB frames as a video file.

    The output format is chosen from the file extension:
    ``.mp4`` (default) writes an H.264 video via ``imageio-ffmpeg``;
    ``.gif`` writes an animated GIF. If MP4 encoding fails (e.g. because
    ``imageio-ffmpeg`` is not installed), the function falls back to a
    sibling ``.gif`` and returns that path instead.

    Returns the actual path written.
    """
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    # imageio.mimsave's type stub is overly strict about list invariance,
    # but it accepts a plain list of ndarrays at runtime.
    if ext == ".gif":
        imageio.mimsave(path, frames, fps=fps, loop=0)  # type: ignore[arg-type]
        return path

    try:
        imageio.mimsave(path, frames, fps=fps)  # type: ignore[arg-type]
        return path
    except Exception as exc:  # noqa: BLE001 — fall back on any encoder error
        fallback = path.with_suffix(".gif")
        imageio.mimsave(fallback, frames, fps=fps, loop=0)  # type: ignore[arg-type]
        print(
            f"Warning: video encoding to {path.suffix} failed ({exc}). "
            f"Saved GIF fallback to {fallback}"
        )
        return fallback


def record_and_save_video(
    agent: Any,
    make_env: Callable[[], Any],
    path: str | Path,
    *,
    epsilon: float = 0.0,
    max_steps: int = 5_000,
    fps: int = 20,
) -> Path:
    """Record one episode of *agent* and save it as a video file.

    Convenience wrapper around :func:`record_episode` + :func:`save_video`
    so notebooks can produce a "model in action" clip in a single call.
    Accepts either a legacy :class:`DQNAgent`/:class:`PPOAgent` or any
    SB3 model that exposes ``predict()``.
    """
    frames = record_episode(agent, make_env, epsilon=epsilon, max_steps=max_steps)
    return save_video(frames, path, fps=fps)


# ---------------------------------------------------------------------------
# Run loading (legacy + SB3)
# ---------------------------------------------------------------------------


def load_run(run_dir: str | Path, *, device: str = "cpu") -> Any:
    """Load the trained agent from a run directory, auto-detecting format.

    Prefers Stable-Baselines3 ``final.zip`` (written by :func:`rl_doom.sb3_utils.train_sb3`)
    when present. Falls back to the legacy ``final_full.pt`` / ``final.pt``
    custom-agent checkpoints.

    The returned object always exposes a ``predict(obs, deterministic=True)``
    method (either natively for SB3 models, or via a tiny shim for legacy
    agents), so downstream notebooks can treat both transparently.
    """
    run_dir = Path(run_dir)
    sb3_zip = run_dir / "checkpoints" / "final.zip"
    best_zip = run_dir / "checkpoints" / "best" / "best_model.zip"

    # Prefer a stage-end snapshot, then the best eval checkpoint.
    for candidate in (sb3_zip, best_zip):
        if candidate.exists():
            import json

            from stable_baselines3 import DQN, PPO

            algo = json.loads((run_dir / "config.json").read_text()).get("algo", "ppo")
            cls = PPO if algo.lower() == "ppo" else DQN
            return cls.load(str(candidate), device=device)

    # Legacy fallback: infer algo from config, load hand-rolled agent.
    import json

    cfg = json.loads((run_dir / "config.json").read_text())
    algo = cfg.get("algo", "ppo").lower()
    hp = cfg.get("hyperparams", {})
    ckpt = run_dir / "checkpoints" / "final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No SB3 zip or legacy .pt checkpoint found in {run_dir / 'checkpoints'}"
        )

    # Build a minimal env just to infer shapes.
    import gymnasium as gym

    from rl_doom.env import make_wrapped_env

    env = make_wrapped_env(cfg.get("env", "basic"))
    obs_space = env.observation_space
    action_space = env.action_space
    if not isinstance(obs_space, gym.spaces.Box) or obs_space.shape is None:
        raise TypeError(f"Expected Box observation space, got {type(obs_space).__name__}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"Expected Discrete action space, got {type(action_space).__name__}")
    obs_shape: tuple[int, ...] = tuple(obs_space.shape)
    n_actions = int(action_space.n)
    env.close()

    agent: DQNAgent | PPOAgent
    if algo == "dqn":
        agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions,
                         lr=hp.get("lr", 1e-4), gamma=hp.get("gamma", 0.99),
                         device=device)
    else:
        agent = PPOAgent(obs_shape=obs_shape, n_actions=n_actions,
                         lr=hp.get("lr", 3e-4), gamma=hp.get("gamma", 0.99),
                         device=device)
    agent.load(str(ckpt))

    # Shim so downstream code can call ``.predict(obs, deterministic=True)``.
    def _predict(obs, deterministic=True, _agent=agent, _algo=algo):
        if _algo == "dqn":
            return np.asarray(_agent.select_action(obs, epsilon=0.0)), None
        action, _, _ = _agent.select_action(obs)
        return np.asarray(action), None

    setattr(agent, "predict", _predict)
    return agent
