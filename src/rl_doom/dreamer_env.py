"""Environment adapter for the DreamerV3 PyTorch port (NM512/dreamerv3-torch).

The port expects an older gym-style interface that differs from Gymnasium in
several ways:

* ``reset()`` returns a single dict observation (not ``(obs, info)``).
* ``step(action)`` returns a 4-tuple ``(obs, reward, is_last, info)``
  (not Gymnasium's 5-tuple).
* Observations are a ``dict`` with an ``image`` key (uint8 RGB by default)
  plus ``is_first`` / ``is_terminal`` bool flags.
* The port's actor emits **one-hot** actions for discrete tasks, so
  ``step`` must accept either an int index or a one-hot vector.
* ``tools.simulate`` keys its episode cache by ``env.id``, so each env
  instance needs a unique ``id`` attribute.

See ``DREAMER_PLAN.md`` §3 / §5.2 for the rationale.
"""

from __future__ import annotations

import datetime
import uuid
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
        # ``tools.simulate`` keys its per-episode cache by ``env.id`` and
        # writes one .npz per id when an episode finishes. Mirror the
        # upstream port's UUID wrapper format (timestamp + hex) so episode
        # filenames sort chronologically.
        self.id = self._make_id()

    @staticmethod
    def _make_id() -> str:
        """Return a fresh ``YYYYMMDDTHHMMSS-<hex>`` id matching the port's UUID wrapper."""
        return f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex}"

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

    @staticmethod
    def _to_index(action: Any) -> int:
        """Coerce the port's one-hot action vector (or a raw int) to an index.

        DreamerV3's ``onehot`` actor emits a 1-D ``(num_actions,)`` numpy
        array (or torch tensor that's already been ``.cpu().numpy()``-ed by
        ``tools.simulate``). Discrete envs expect a Python int. We accept
        either form so callers can drive this env with raw ints in tests
        without going through the port's actor pipeline.
        """
        arr = np.asarray(action)
        if arr.ndim == 0:
            return int(arr.item())
        return int(np.argmax(arr))

    def reset(self) -> dict[str, Any]:
        # Refresh the id on every episode start so ``tools.simulate``'s
        # per-episode cache writes a fresh .npz instead of clobbering the
        # previous one. Matches the upstream ``UUID`` wrapper's behaviour.
        self.id = self._make_id()
        obs, _info = self._env.reset()
        return self._obs_dict(obs, is_first=True, is_terminal=False)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        idx = self._to_index(action)
        obs, reward, terminated, truncated, info = self._env.step(idx)
        is_last = bool(terminated or truncated)
        # Dreamer distinguishes ``is_last`` (episode ended) from
        # ``is_terminal`` (absorbing state, i.e. "real" termination not a
        # time-limit truncation). DoomEnv folds ViZDoom's scenario timeout
        # into ``terminated=True`` and never sets ``truncated``, so we have
        # to consult ``info["termination_reason"]`` to recover the
        # distinction — otherwise the world model learns that running out
        # the clock is an absorbing state.
        if terminated and info.get("termination_reason") == "timeout":
            is_terminal = False
        else:
            is_terminal = bool(terminated)
        return (
            self._obs_dict(obs, is_first=False, is_terminal=is_terminal),
            float(reward),
            is_last,
            info,
        )

    def close(self) -> None:
        self._env.close()
