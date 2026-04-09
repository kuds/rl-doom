"""Evaluation and gameplay recording utilities."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def evaluate_agent(
    agent: Any,
    make_env: Callable[[], Any],
    n_episodes: int = 10,
) -> np.ndarray:
    """Run the agent for *n_episodes* and return per-episode total rewards.

    Automatically detects DQN (``agent.policy_net``) vs PPO
    (``agent.network``) to call the correct action-selection API.
    """
    is_dqn = hasattr(agent, "policy_net")
    env = make_env()
    rewards: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward, done = 0.0, False
        while not done:
            if is_dqn:
                action = agent.select_action(obs, epsilon=0.0)
            else:
                action, _, _ = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        rewards.append(total_reward)
    env.close()
    return np.array(rewards)


def record_episode(
    agent: Any,
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
    is_dqn = hasattr(agent, "policy_net")
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
        if is_dqn:
            action = agent.select_action(obs, epsilon=epsilon)
        else:
            action, _, _ = agent.select_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(base_env.render())
        done = terminated or truncated
        step += 1

    env.close()
    return frames
