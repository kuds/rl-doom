"""Environment adapter for the DreamerV3 PyTorch port (NM512/dreamerv3-torch).

The port expects an older gym-style interface that differs from Gymnasium in
several ways:

* ``reset()`` returns a single dict observation (not ``(obs, info)``).
* ``step(action)`` returns a 4-tuple ``(obs, reward, is_last, info)``
  (not Gymnasium's 5-tuple).
* Observations are a ``dict`` with an ``image`` key (uint8 RGB by default)
  plus ``is_first`` / ``is_terminal`` bool flags.
* Discrete actions are consumed as one-hot vectors via the port's own
  ``OneHotAction`` wrapper, so we still expose a ``Discrete`` action space.

See ``DREAMER_PLAN.md`` §3 and §5.2 for the rationale.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from rl_doom.env import DoomEnv, ResizeObservation, SkipFrame


class DreamerDoomEnv:
    """Adapt a wrapped ViZDoom env to the DreamerV3 port's interface.

    Notes
    -----
    This is intentionally **not** a ``gym.Env`` / ``gymnasium.Env`` subclass:
    its ``step`` / ``reset`` return signatures deviate from Gymnasium's
    contract. Mixing the two would invite silent bugs from Gymnasium's
    compat shims. Kept as a plain class whose interface matches what
    ``dreamerv3_torch.tools.simulate`` expects.
    """

    def __init__(
        self,
        scenario: str = "defend_the_center",
        resize_shape: tuple[int, int] = (64, 64),
        frame_skip: int = 4,
        grayscale: bool = False,
        use_compound_actions: bool = True,
        doom_skill: int | None = None,
        num_bots: int = 0,
    ) -> None:
        base: gym.Env = DoomEnv(
            scenario=scenario,
            use_compound_actions=use_compound_actions,
            doom_skill=doom_skill,
            num_bots=num_bots,
        )
        base = ResizeObservation(base, shape=resize_shape, grayscale=grayscale)
        base = SkipFrame(base, skip=frame_skip)
        # NOTE: no FrameStack — Dreamer's RSSM replaces frame stacking.
        self._env: gym.Env = base

        # Image shape Dreamer consumes: (H, W, C) with C=1 for grayscale.
        if grayscale:
            img_shape: tuple[int, int, int] = (*resize_shape, 1)
        else:
            img_shape = (*resize_shape, 3)
        self._img_shape = img_shape
        self._grayscale = grayscale

        # ``is_first`` / ``is_terminal`` are semantic booleans but Gymnasium's
        # Box accepts only numeric dtypes in its type stubs. We advertise them
        # as uint8 {0, 1} here; the dict values at runtime are ``np.bool_``
        # which Dreamer consumes as-is.
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape, dtype=np.uint8),
                "is_first": gym.spaces.Box(0, 1, (), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.uint8),
            }
        )
        self.action_space = self._env.action_space

    # ------------------------------------------------------------------

    def _format_image(self, obs: np.ndarray) -> np.ndarray:
        """Ensure the image has shape ``(H, W, C)`` matching ``_img_shape``."""
        if self._grayscale:
            # ResizeObservation returns (H, W) when grayscale=True.
            if obs.ndim == 2:
                obs = obs[..., np.newaxis]
        else:
            # Non-grayscale path already returns (H, W, 3).
            if obs.ndim != 3 or obs.shape[-1] != 3:
                raise ValueError(
                    f"Expected RGB (H, W, 3) from inner env, got shape {obs.shape}",
                )
        return obs.astype(np.uint8, copy=False)

    def _obs_dict(
        self, image: np.ndarray, *, is_first: bool, is_terminal: bool,
    ) -> dict[str, Any]:
        return {
            "image": self._format_image(image),
            "is_first": np.bool_(is_first),
            "is_terminal": np.bool_(is_terminal),
        }

    def reset(self) -> dict[str, Any]:
        obs, _info = self._env.reset()
        return self._obs_dict(obs, is_first=True, is_terminal=False)

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        is_last = bool(terminated or truncated)
        # Dreamer distinguishes ``is_last`` (episode ended) from
        # ``is_terminal`` (absorbing state, i.e. "real" termination not a
        # time-limit truncation). Only ``terminated`` counts as terminal.
        return (
            self._obs_dict(obs, is_first=False, is_terminal=bool(terminated)),
            float(reward),
            is_last,
            info,
        )

    def close(self) -> None:
        self._env.close()
