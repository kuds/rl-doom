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


# Curated action sets per scenario. Each inner list is the set of simultaneously
# pressed buttons for one discrete action. The original ``np.eye`` one-hot
# layout (one button pressed per action) cannot express compound actions that
# are required to play these scenarios well — e.g. "move forward while
# shooting" in Deadly Corridor or "turn while firing" in Defend the Center.
# Without compound actions the PPO policy on Deadly Corridor collapses to a
# deterministic "die faster" strategy because no single-button policy can
# survive the corridor.
#
# Button names map 1:1 to ``vizdoom.Button`` enum member names. Any name that
# is not in the scenario's ``available_buttons`` list raises at init time.
SCENARIO_ACTION_SETS: dict[str, list[list[str]]] = {
    "basic": [
        ["MOVE_LEFT"],
        ["MOVE_RIGHT"],
        ["ATTACK"],
        # Strafe while firing — lets the agent track a moving target.
        ["MOVE_LEFT", "ATTACK"],
        ["MOVE_RIGHT", "ATTACK"],
    ],
    "deadly_corridor": [
        # Pure movement / turning (still useful for navigation + aiming).
        ["MOVE_FORWARD"],
        ["MOVE_BACKWARD"],
        ["TURN_LEFT"],
        ["TURN_RIGHT"],
        ["MOVE_LEFT"],
        ["MOVE_RIGHT"],
        # Stationary attack.
        ["ATTACK"],
        # The compound actions that actually let the agent survive:
        # push forward while firing / strafing / aiming.
        ["MOVE_FORWARD", "ATTACK"],
        ["MOVE_FORWARD", "MOVE_LEFT"],
        ["MOVE_FORWARD", "MOVE_RIGHT"],
        ["MOVE_FORWARD", "TURN_LEFT"],
        ["MOVE_FORWARD", "TURN_RIGHT"],
        # Fire while turning to track imps on either side.
        ["TURN_LEFT", "ATTACK"],
        ["TURN_RIGHT", "ATTACK"],
    ],
    "defend_the_center": [
        ["TURN_LEFT"],
        ["TURN_RIGHT"],
        ["ATTACK"],
        # Killer compound: rotate while firing to sweep the arena.
        ["TURN_LEFT", "ATTACK"],
        ["TURN_RIGHT", "ATTACK"],
    ],
    "deathmatch": [
        # Deathmatch needs both navigation and combat. We restrict to the
        # binary movement/attack buttons and skip the delta + weapon-select
        # buttons in deathmatch.cfg — delta buttons expect continuous values
        # rather than a 0/1 press, and weapon switching adds a large branching
        # factor that's better left for a policy with a more expressive
        # action space. SPEED is included so the agent can run while chasing.
        ["MOVE_FORWARD"],
        ["MOVE_BACKWARD"],
        ["TURN_LEFT"],
        ["TURN_RIGHT"],
        ["MOVE_LEFT"],
        ["MOVE_RIGHT"],
        ["ATTACK"],
        # Fire while moving/turning to track and engage opponents.
        ["MOVE_FORWARD", "ATTACK"],
        ["MOVE_LEFT", "ATTACK"],
        ["MOVE_RIGHT", "ATTACK"],
        ["TURN_LEFT", "ATTACK"],
        ["TURN_RIGHT", "ATTACK"],
        # Navigate while aiming.
        ["MOVE_FORWARD", "TURN_LEFT"],
        ["MOVE_FORWARD", "TURN_RIGHT"],
        # Sprint for closing distance / escaping.
        ["SPEED", "MOVE_FORWARD"],
        ["SPEED", "MOVE_FORWARD", "ATTACK"],
    ],
}


# Per-scenario default difficulty (Doom skill 1..5). ``None``/unset means
# "use whatever ``doom_skill`` is baked into the scenario's .cfg" (typically 3
# — "Hurt me plenty", which is also ViZDoom's built-in DoomGame default).
# Overriding here keeps train / eval / video / analysis envs in lock-step
# without every call site having to pass a kwarg.
SCENARIO_DEFAULT_SKILL: dict[str, int] = {}


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
    doom_skill : int | None
        ViZDoom difficulty (1..5). ``None`` (default) picks the scenario's
        entry in :data:`SCENARIO_DEFAULT_SKILL` if one exists, otherwise
        leaves the cfg's own default untouched.
    num_bots : int
        Number of ZDoom AI-controlled bots to spawn via ``addbot`` at the
        start of each episode. Only meaningful on deathmatch-style maps;
        non-deathmatch scenarios will silently ignore the addbot command.
        Defaults to 0 (no bots).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scenario: str = "basic",
        frame_skip: int = 1,
        render_mode: str | None = None,
        use_compound_actions: bool = True,
        doom_skill: int | None = None,
        num_bots: int = 0,
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

        skill = doom_skill if doom_skill is not None else SCENARIO_DEFAULT_SKILL.get(scenario)
        if skill is not None:
            if not 1 <= skill <= 5:
                raise ValueError(
                    f"doom_skill must be in [1, 5], got {skill!r}"
                )
            self.game.set_doom_skill(skill)
        self._doom_skill = skill

        if num_bots < 0:
            raise ValueError(f"num_bots must be >= 0, got {num_bots!r}")
        self._num_bots = num_bots

        self.game.init()

        # Cache the scenario's episode timeout (in game tics) so we can
        # classify terminations as death/timeout/goal_reached in ``step``.
        # A value of 0 means "no timeout" (agent must reach a terminal
        # state defined by the scenario's ACS script).
        self._episode_timeout = int(self.game.get_episode_timeout())

        self._frame_skip = frame_skip
        n_buttons = self.game.get_available_buttons_size()
        button_names = [b.name for b in self.game.get_available_buttons()]

        # Prefer the curated compound action set for this scenario when one is
        # registered; fall back to the legacy one-hot layout otherwise (or
        # when the caller opts out). Falling back preserves backward
        # compatibility for scenarios we haven't tuned yet (health_gathering,
        # my_way_home, predict_position).
        curated = SCENARIO_ACTION_SETS.get(scenario) if use_compound_actions else None
        if curated is not None:
            name_to_idx = {name: i for i, name in enumerate(button_names)}
            actions: list[list[int]] = []
            for combo in curated:
                vec = [0] * n_buttons
                for name in combo:
                    if name not in name_to_idx:
                        raise ValueError(
                            f"Scenario {scenario!r} compound action {combo!r} "
                            f"references button {name!r} which is not available; "
                            f"available buttons: {button_names}"
                        )
                    vec[name_to_idx[name]] = 1
                actions.append(vec)
            self._actions = actions
        else:
            # Legacy one-hot vectors (one button pressed per action).
            self._actions = np.eye(n_buttons, dtype=np.int32).tolist()

        self.available_actions = self._actions
        self.action_space = gym.spaces.Discrete(len(self._actions))
        # Observation = RGB image from the screen buffer (H, W, 3).
        h, w = self.game.get_screen_height(), self.game.get_screen_width()
        self._obs_shape: tuple[int, int, int] = (h, w, 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self._obs_shape, dtype=np.uint8,
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
        return np.zeros(self._obs_shape, dtype=np.uint8)

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
        for _ in range(self._num_bots):
            self.game.send_game_command("addbot")
        return self._get_obs(), {}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        reward = self.game.make_action(self._actions[action], self._frame_skip)
        terminated = self.game.is_episode_finished()
        obs = self._get_obs()
        info: dict[str, Any] = {}
        if terminated:
            info["termination_reason"] = self._classify_termination()
            info["episode_tics"] = int(self.game.get_episode_time())
            try:
                info["final_health"] = float(
                    self.game.get_game_variable(vizdoom.GameVariable.HEALTH),
                )
            except Exception:  # noqa: BLE001 — game var may be unavailable
                info["final_health"] = None
        return obs, reward, terminated, False, info

    def _classify_termination(self) -> str:
        """Label why the current episode ended.

        * ``"death"``     — the player died (health <= 0 or dead flag set).
        * ``"timeout"``   — the scenario's ``episode_timeout`` elapsed.
        * ``"goal_reached"`` — the ACS script ended the episode any other
          way, e.g. picking up the green vest at the end of Deadly Corridor.
        """
        if self.game.is_player_dead():
            return "death"
        tic = int(self.game.get_episode_time())
        if self._episode_timeout > 0 and tic >= self._episode_timeout:
            return "timeout"
        return "goal_reached"

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
    use_compound_actions: bool = True,
    doom_skill: int | None = None,
    num_bots: int = 0,
) -> gym.Env:
    """Build the standard Atari-style preprocessing pipeline for a ViZDoom scenario.

    Chains ``DoomEnv -> ResizeObservation -> SkipFrame -> FrameStack``.  This
    is the wrapper stack used by every training and analysis notebook, so
    centralising it here keeps them in lock-step.

    ``use_compound_actions`` (default True) expands the one-hot action space
    into the curated compound action set for the scenario — see
    :data:`SCENARIO_ACTION_SETS`. Pass ``False`` to keep the legacy one-per-
    button layout (useful when comparing against older runs).

    ``doom_skill`` (default None) forwards to :class:`DoomEnv`; passing
    ``None`` lets the scenario's :data:`SCENARIO_DEFAULT_SKILL` entry (or
    the cfg default) take effect.

    ``num_bots`` (default 0) forwards to :class:`DoomEnv` — ZDoom AI bots
    are spawned at each ``reset()`` on deathmatch-style maps.
    """
    env: gym.Env = DoomEnv(
        scenario=scenario,
        use_compound_actions=use_compound_actions,
        doom_skill=doom_skill,
        num_bots=num_bots,
    )
    env = ResizeObservation(env, shape=resize_shape)
    env = SkipFrame(env, skip=frame_skip)
    env = FrameStack(env, num_stack=num_stack)
    return env


# ======================================================================
# Stable-Baselines3 integration
# ======================================================================


def make_sb3_env(
    scenario: str,
    *,
    n_envs: int = 1,
    seed: int = 0,
    monitor_dir: str | None = None,
    resize_shape: tuple[int, int] = (84, 84),
    frame_skip: int = 4,
    num_stack: int = 4,
    use_compound_actions: bool = True,
    doom_skill: int | None = None,
    num_bots: int = 0,
):
    """Build a Stable-Baselines3 ``DummyVecEnv`` for a ViZDoom scenario.

    We intentionally always use ``DummyVecEnv`` (sequential in-process rollouts)
    instead of ``SubprocVecEnv``: ViZDoom spawns a native ``vizdoom`` process per
    env under the hood, and Colab's IPython runtime does not play well with
    Python-level ``multiprocessing`` forks/spawns (daemon thread conflicts,
    cleanup errors at shutdown). ``DummyVecEnv`` sidesteps all of that.

    Each underlying env is wrapped in :class:`stable_baselines3.common.monitor.Monitor`
    when ``monitor_dir`` is provided, so per-episode returns/lengths are
    recorded to CSV and surfaced via ``model.ep_info_buffer``.

    Observations are ``(num_stack, H, W)`` ``uint8`` frames (channels-first),
    which SB3's ``CnnPolicy`` consumes directly — no ``VecTransposeImage``
    wrapper needed.
    """
    # Imports are deferred so ``rl_doom.env`` stays importable in environments
    # without Stable-Baselines3 installed (e.g. unit tests for DoomEnv only).
    from pathlib import Path

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    if monitor_dir is not None:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)

    def _thunk(idx: int):
        def _make() -> gym.Env:
            env = make_wrapped_env(
                scenario,
                resize_shape=resize_shape,
                frame_skip=frame_skip,
                num_stack=num_stack,
                use_compound_actions=use_compound_actions,
                doom_skill=doom_skill,
                num_bots=num_bots,
            )
            # Seed the env deterministically per worker.
            env.reset(seed=seed + idx)
            if monitor_dir is not None:
                env = Monitor(
                    env,
                    filename=str(Path(monitor_dir) / f"monitor_{idx}"),
                )
            return env

        return _make

    return DummyVecEnv([_thunk(i) for i in range(n_envs)])
