"""ViZDoom environment wrappers conforming to the Gymnasium API.

Provides a base ``DoomEnv`` and composable observation wrappers
(``ResizeObservation``, ``SkipFrame``, ``FrameStack``) for standard
Atari-style preprocessing pipelines.
"""

from __future__ import annotations

from collections import deque
from typing import Any, SupportsFloat

import cv2
import gymnasium as gym
import numpy as np
import vizdoom

# Mapping from friendly scenario names to ViZDoom config filenames.
SCENARIO_MAP: dict[str, str] = {
    "basic": "basic.cfg",
    "deadly_corridor": "deadly_corridor.cfg",
    "defend_the_center": "defend_the_center.cfg",
    "deathmatch": "deathmatch.cfg",
    "health_gathering": "health_gathering.cfg",
    "my_way_home": "my_way_home.cfg",
    "predict_position": "predict_position.cfg",
}


class DoomEnv(gym.Env):
    """Gymnasium wrapper around a ViZDoom game instance.

    Parameters
    ----------
    scenario : str
        Scenario name (e.g. ``"basic"``, ``"deadly_corridor"``).
    frame_skip : int
        Number of internal ViZDoom tics per ``step()`` call.  Set to 1
        when using the external ``SkipFrame`` wrapper.
    render_mode : str | None
        Not used directly, kept for Gymnasium compatibility.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scenario: str = "basic",
        frame_skip: int = 1,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        cfg_name = SCENARIO_MAP.get(scenario)
        if cfg_name is None:
            raise ValueError(
                f"Unknown scenario {scenario!r}. "
                f"Available: {list(SCENARIO_MAP)}"
            )

        self.game = vizdoom.DoomGame()
        self.game.load_config(f"{vizdoom.scenarios_path}/{cfg_name}")
        self.game.set_window_visible(False)
        self.game.set_screen_format(vizdoom.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
        self.game.init()

        self._frame_skip = frame_skip
        n_buttons = self.game.get_available_buttons_size()

        # Build the list of one-hot action vectors (one per button).
        self._actions = np.eye(n_buttons, dtype=np.int32).tolist()
        self.available_actions = self._actions

        self.action_space = gym.spaces.Discrete(n_buttons)
        # Observation = RGB image from the screen buffer (H, W, 3).
        h, w = self.game.get_screen_height(), self.game.get_screen_width()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8,
        )

    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        state = self.game.get_state()
        if state is not None:
            buf = state.screen_buffer
            # With ScreenFormat.RGB24 the buffer is already (H, W, 3).
            # With channel-first formats (e.g. CRCGCB) it would be (3, H, W);
            # handle that case defensively so the observation is always HWC.
            if buf.ndim == 3 and buf.shape[0] == 3 and buf.shape[-1] != 3:
                buf = buf.transpose(1, 2, 0)
            return buf
        # After episode ends the state can be None; return a black frame.
        return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        return self._get_obs(), {}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        reward = self.game.make_action(self._actions[action], self._frame_skip)
        terminated = self.game.is_episode_finished()
        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def render(self) -> np.ndarray:  # type: ignore[override]
        # Gymnasium's base signature returns RenderFrame | list | None;
        # we always produce an RGB ndarray, which is a valid RenderFrame
        # but narrower than the abstract type.
        return self._get_obs()

    def close(self) -> None:
        self.game.close()


# ======================================================================
# Observation wrappers
# ======================================================================


class ResizeObservation(gym.ObservationWrapper):
    """Resize frames to ``(H, W)`` grayscale.

    The output observation is a 2-D ``uint8`` array of shape ``(H, W)``.
    """

    def __init__(self, env: gym.Env, shape: tuple[int, int] = (84, 84)) -> None:
        super().__init__(env)
        self._shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3 and obs.shape[2] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self._shape[1], self._shape[0]), interpolation=cv2.INTER_AREA)
        return obs


class SkipFrame(gym.Wrapper):
    """Return every ``skip``-th frame and accumulate rewards in between."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class FrameStack(gym.Wrapper):
    """Stack the last ``num_stack`` observations along a new leading axis.

    Output shape: ``(num_stack, *single_obs_shape)``.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4) -> None:
        super().__init__(env)
        self._num_stack = num_stack
        self._frames: deque[np.ndarray] = deque(maxlen=num_stack)

        base_space = env.observation_space
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError(
                f"FrameStack requires a Box observation space, got {type(base_space).__name__}",
            )
        low = np.repeat(base_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(base_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=base_space.dtype,  # type: ignore[arg-type]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the underlying env and fill the stack with the initial frame."""
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return np.array(self._frames), info

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the underlying env, append the new frame, and return the stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return np.array(self._frames), reward, terminated, truncated, info


# ======================================================================
# Factory helper
# ======================================================================


def make_wrapped_env(
    scenario: str,
    *,
    resize_shape: tuple[int, int] = (84, 84),
    frame_skip: int = 4,
    num_stack: int = 4,
) -> gym.Env:
    """Build the standard Atari-style preprocessing pipeline for a ViZDoom scenario.

    Chains ``DoomEnv -> ResizeObservation -> SkipFrame -> FrameStack``.  This
    is the wrapper stack used by every training and analysis notebook, so
    centralising it here keeps them in lock-step.
    """
    env: gym.Env = DoomEnv(scenario=scenario)
    env = ResizeObservation(env, shape=resize_shape)
    env = SkipFrame(env, skip=frame_skip)
    env = FrameStack(env, num_stack=num_stack)
    return env
