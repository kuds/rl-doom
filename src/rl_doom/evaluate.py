"""Evaluation and gameplay recording utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from rl_doom.agents.dqn import DQNAgent
from rl_doom.agents.ppo import PPOAgent


def _select_action(agent: Any, obs: np.ndarray, *, epsilon: float = 0.0) -> int:
    """Dispatch to the agent's action API based on its class."""
    if isinstance(agent, DQNAgent):
        return agent.select_action(obs, epsilon=epsilon)
    if isinstance(agent, PPOAgent):
        action, _, _ = agent.select_action(obs)
        return action
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

    if ext == ".gif":
        imageio.mimsave(path, frames, fps=fps, loop=0)
        return path

    try:
        imageio.mimsave(path, frames, fps=fps)
        return path
    except Exception as exc:  # noqa: BLE001 — fall back on any encoder error
        fallback = path.with_suffix(".gif")
        imageio.mimsave(fallback, frames, fps=fps, loop=0)
        print(
            f"Warning: video encoding to {path.suffix} failed ({exc}). "
            f"Saved GIF fallback to {fallback}"
        )
        return fallback


def record_and_save_video(
    agent: DQNAgent | PPOAgent,
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
    """
    frames = record_episode(agent, make_env, epsilon=epsilon, max_steps=max_steps)
    return save_video(frames, path, fps=fps)
