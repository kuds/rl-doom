"""PettingZoo ``ParallelEnv`` wrapping ViZDoom's native multiplayer.

Each agent controls its own ``vizdoom.DoomGame`` instance; one instance is the
host, the rest ``-join`` it over loopback. In synchronous mode every player
must submit an action per tic before any player advances, so ``reset()`` /
``step()`` fan out across a thread pool and wait for all peers.

Two factory functions build the supported match formats:

* :func:`make_1v1_env` - 2 agents, free-for-all, reward = own-frag-delta minus
  own-death-delta.
* :func:`make_2v2_env` - 4 agents on two teams of two; reward for each agent
  on a team is the team's frag-delta minus the team's death-delta (teammates
  share identical rewards each step).

The raw per-player observation is an RGB screen buffer identical to
:class:`rl_doom.env.DoomEnv`; enabling ``resize_shape`` / ``num_stack`` in the
constructor produces the same ``(num_stack, H, W)`` ``uint8`` tensor that the
single-player pipeline emits, so models trained single-agent can be loaded
directly for self-play.
"""

from __future__ import annotations

import socket
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import vizdoom
from pettingzoo import ParallelEnv

# Imported from ``scenario_limits`` rather than ``rl_doom.env`` on purpose:
# env.py builds a ``vizdoom.GameVariable`` table at module load, so importing
# it here would drag that into every consumer — including the tests that swap
# in a fake ``vizdoom`` module.
from rl_doom.scenario_limits import SCENARIO_ACTION_SETS

# Default compound action set for deathmatch-style multiplayer.
#
# Sourced from the single-player table rather than restated here. This
# module's docstring promises that "models trained single-agent can be loaded
# directly for self-play", and that requires the two envs to expose the *same*
# action space — the same length, and the same index -> button mapping.
#
# Two hand-maintained copies had already drifted: this one had 14 entries
# against env.py's 16 (missing both SPEED compounds), with indices 8/9 and
# 12/13 swapped between "MOVE_x + ATTACK" and "MOVE_FORWARD + TURN_x". A
# single-player checkpoint could not be loaded as an opponent at all
# (Discrete(16) vs Discrete(14)), and had it been reshaped it would have
# strafed where it meant to turn.
#
# Copied element-wise so a caller mutating one table cannot silently reshape
# the other; ``test_deathmatch_action_space_matches_single_player`` asserts the
# two stay equivalent.
DEATHMATCH_ACTIONS: list[list[str]] = [
    list(combo) for combo in SCENARIO_ACTION_SETS["deathmatch"]
]


def _pick_free_port() -> int:
    """Return a currently-unused TCP port on localhost.

    The port can race with other processes between this call and ``init()``,
    but in practice the window is tiny and the cost of a collision is one
    retry at env construction.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_action_table(
    game: vizdoom.DoomGame,
    compound: list[list[str]] | None,
) -> list[list[int]]:
    """Convert a list of button-name combos into ViZDoom button vectors.

    Falls back to the one-hot layout (one action per available button) when
    ``compound`` is ``None``.
    """
    n_buttons = game.get_available_buttons_size()
    names = [b.name for b in game.get_available_buttons()]
    if compound is None:
        return np.eye(n_buttons, dtype=np.int32).tolist()
    name_to_idx = {name: i for i, name in enumerate(names)}
    table: list[list[int]] = []
    for combo in compound:
        vec = [0] * n_buttons
        for name in combo:
            if name not in name_to_idx:
                raise ValueError(
                    f"Compound action {combo!r} references button {name!r} which is "
                    f"not available in this scenario; available: {names}",
                )
            vec[name_to_idx[name]] = 1
        table.append(vec)
    return table


class DoomMultiplayerEnv(ParallelEnv):
    """PettingZoo ``ParallelEnv`` over N ViZDoom players on a shared map.

    Parameters
    ----------
    num_players : int
        Total player count (2 for 1v1, 4 for 2v2).
    teams : list[str] | None
        Parallel to ``agents``. Agents sharing a team-label get the same team
        reward each step. ``None`` means free-for-all (each agent is its own
        team).
    scenario_cfg : str
        ViZDoom ``.cfg`` filename to load from ``vizdoom.scenarios_path``
        (e.g. ``"deathmatch.cfg"`` or ``"cig.cfg"``).
    time_limit : float
        Match timelimit in minutes (passed through as ``+timelimit``).
    frag_limit : int
        Per-player frag cap; 0 disables (passed through as ``+fraglimit``).
    frame_skip : int
        Tics consumed per :meth:`step`.
    resize_shape : tuple[int, int] | None
        If given, screen buffer is resized to ``(H, W)`` grayscale per player.
    num_stack : int
        If > 1, stack this many consecutive frames along a new leading axis
        per agent (only valid when ``resize_shape`` is set).
    compound_actions : list[list[str]] | None
        Action table (see :data:`DEATHMATCH_ACTIONS`). ``None`` = one-hot.
    window_visible : bool
        Render a visible window per player - debugging only.
    port : int | None
        Force a specific host port. Default picks a free ephemeral port so
        multiple envs can coexist on one machine (e.g. SB3 vectorised runs).
    """

    metadata = {"render_modes": ["rgb_array"], "name": "doom_multiplayer_v0"}

    def __init__(
        self,
        *,
        num_players: int = 2,
        teams: list[str] | None = None,
        scenario_cfg: str = "deathmatch.cfg",
        time_limit: float = 10.0,
        frag_limit: int = 0,
        frame_skip: int = 4,
        resize_shape: tuple[int, int] | None = (84, 84),
        num_stack: int = 4,
        compound_actions: list[list[str]] | None = None,
        window_visible: bool = False,
        port: int | None = None,
        respawn_on_death: bool = True,
    ) -> None:
        if num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {num_players}")
        if teams is not None and len(teams) != num_players:
            raise ValueError(
                f"teams must have length {num_players}, got {len(teams)}",
            )
        if num_stack > 1 and resize_shape is None:
            raise ValueError("num_stack > 1 requires resize_shape to be set")

        self._num_players = num_players
        self._teams = teams
        self._scenario_cfg = scenario_cfg
        self._time_limit = time_limit
        self._frag_limit = frag_limit
        self._frame_skip = frame_skip
        self._resize_shape = resize_shape
        self._num_stack = num_stack
        self._compound_actions = (
            compound_actions if compound_actions is not None else DEATHMATCH_ACTIONS
        )
        self._window_visible = window_visible
        self._port = port if port is not None else _pick_free_port()
        self._respawn_on_death = respawn_on_death

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents: list[str] = []

        # Per-agent state, populated by ``_start_games()``.
        self._games: dict[str, vizdoom.DoomGame] = {}
        self._action_tables: dict[str, list[list[int]]] = {}
        self._frame_stacks: dict[str, deque[np.ndarray]] = {}
        self._prev_frags: dict[str, int] = {}
        self._prev_deaths: dict[str, int] = {}

        # Pool sized to the player count so every per-tic fan-out runs in
        # parallel without queueing.
        self._pool = ThreadPoolExecutor(max_workers=num_players)
        self._started = False

        # Build a dummy game just to derive the action/observation shapes
        # without paying for the multiplayer init dance. ``init()`` is skipped.
        probe = vizdoom.DoomGame()
        probe.load_config(f"{vizdoom.scenarios_path}/{scenario_cfg}")
        probe_table = _build_action_table(probe, self._compound_actions)
        probe.close()

        self._num_actions = len(probe_table)
        single_action_space: gym.spaces.Discrete = gym.spaces.Discrete(self._num_actions)
        single_obs_space: gym.spaces.Box = self._build_obs_space()
        # PettingZoo's Parallel API asks for per-agent spaces; they're identical here.
        self._action_spaces: dict[str, gym.spaces.Discrete] = {
            a: single_action_space for a in self.possible_agents
        }
        self._observation_spaces: dict[str, gym.spaces.Box] = {
            a: single_obs_space for a in self.possible_agents
        }

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> gym.spaces.Box:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Discrete:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        # First reset: spin up the networked games. Subsequent resets reuse
        # them and just call ``new_episode()`` on every player in parallel,
        # which is ViZDoom's required protocol for multiplayer.
        if not self._started:
            self._start_games(seed=seed)
        else:
            self._parallel(lambda g, _a: g.new_episode(), None)

        self.agents = list(self.possible_agents)
        self._prev_frags = {a: 0 for a in self.agents}
        self._prev_deaths = {a: 0 for a in self.agents}
        self._frame_stacks = {a: deque(maxlen=self._num_stack) for a in self.agents}

        obs = {a: self._process_obs(a, self._read_screen(self._games[a])) for a in self.agents}
        infos: dict[str, dict[str, Any]] = {a: {} for a in self.agents}
        return obs, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        missing = set(self.agents) - set(actions)
        if missing:
            raise KeyError(f"step() missing actions for agents: {sorted(missing)}")

        # Fan out ``make_action`` across the pool - each player blocks on
        # its peers inside ViZDoom, so we must submit concurrently or deadlock.
        def _act(game: vizdoom.DoomGame, agent: str) -> None:
            if self._respawn_on_death and game.is_player_dead():
                game.respawn_player()
            vec = self._action_tables[agent][int(actions[agent])]
            game.make_action(vec, self._frame_skip)

        self._parallel(_act, actions)

        # Any player seeing ``is_episode_finished()`` means the match is over
        # (timelimit/fraglimit triggers on the host and propagates). We surface
        # that as termination for every agent so PettingZoo empties ``agents``.
        finished = any(g.is_episode_finished() for g in self._games.values())

        obs: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terms: dict[str, bool] = {}
        truncs: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}

        # Per-agent frag/death deltas drive the reward; snapshot first so we
        # can aggregate by team without double-reading game variables.
        frags_now = {a: self._get_frags(self._games[a]) for a in self.agents}
        deaths_now = {a: self._get_deaths(self._games[a]) for a in self.agents}
        frag_delta = {a: frags_now[a] - self._prev_frags[a] for a in self.agents}
        death_delta = {a: deaths_now[a] - self._prev_deaths[a] for a in self.agents}
        self._prev_frags = frags_now
        self._prev_deaths = deaths_now

        team_frag_delta, team_death_delta = self._team_deltas(frag_delta, death_delta)

        for idx, agent in enumerate(self.agents):
            obs[agent] = self._process_obs(agent, self._read_screen(self._games[agent]))
            if self._teams is None:
                rewards[agent] = float(frag_delta[agent] - death_delta[agent])
            else:
                team = self._teams[idx]
                rewards[agent] = float(team_frag_delta[team] - team_death_delta[team])
            terms[agent] = finished
            truncs[agent] = False
            infos[agent] = {
                "frags": frags_now[agent],
                "deaths": deaths_now[agent],
                "team": self._teams[idx] if self._teams is not None else agent,
            }

        if finished:
            # PettingZoo convention: once every agent is terminated or
            # truncated, clear ``agents`` so downstream loops stop calling step.
            self.agents = []

        return obs, rewards, terms, truncs, infos

    def render(self) -> dict[str, np.ndarray]:
        # Return every player's current RGB frame so callers can stitch a
        # grid view for recording. Deviation from the single-agent ``render``
        # signature is intentional - PettingZoo doesn't mandate a shape here.
        return {a: self._read_screen(self._games[a]) for a in self.possible_agents}

    def close(self) -> None:
        for game in self._games.values():
            try:
                game.close()
            except Exception:  # noqa: BLE001 - vizdoom may raise on double-close
                pass
        self._games.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._started = False
        self.agents = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_obs_space(self) -> gym.spaces.Box:
        if self._resize_shape is None:
            # Raw RGB shape falls back to the config's screen resolution; we
            # probe via the same trick as in ``DoomEnv`` by reading the
            # default 320x240 that ViZDoom loads when the cfg doesn't override.
            return gym.spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8)
        shape: tuple[int, ...] = (
            (self._num_stack, *self._resize_shape)
            if self._num_stack > 1
            else self._resize_shape
        )
        return gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def _start_games(self, seed: int | None) -> None:
        """Spin up host + joiners in parallel so ``init()`` handshakes.

        The host's ``init()`` blocks until every joiner has connected, so we
        have to submit joiner inits on the pool *before* waiting on the host.
        """
        host_args = (
            f"-host {self._num_players} -port {self._port} -deathmatch "
            f"+timelimit {self._time_limit} +sv_spawnfarthest 1 "
            f"+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 "
            f"+fraglimit {self._frag_limit}"
        )
        join_args = f"-join 127.0.0.1 -port {self._port}"

        futures = []
        for idx, agent in enumerate(self.possible_agents):
            extra = f"+name {agent} +colorset {idx % 8}"
            args = f"{host_args} {extra}" if idx == 0 else f"{join_args} {extra}"
            futures.append(self._pool.submit(self._init_one, agent, args, seed, idx))
        for fut in futures:
            fut.result()
        self._started = True

    def _init_one(self, agent: str, args: str, seed: int | None, idx: int) -> None:
        game = vizdoom.DoomGame()
        game.load_config(f"{vizdoom.scenarios_path}/{self._scenario_cfg}")
        game.set_window_visible(self._window_visible)
        game.set_screen_format(vizdoom.ScreenFormat.RGB24)
        game.set_mode(vizdoom.Mode.PLAYER)
        # Ensure frag/death tracking is exposed even if the cfg forgot.
        existing = list(game.get_available_game_variables())
        for needed in (vizdoom.GameVariable.FRAGCOUNT, vizdoom.GameVariable.DEATHCOUNT):
            if needed not in existing:
                existing.append(needed)
        game.set_available_game_variables(existing)
        game.add_game_args(args)
        if seed is not None:
            # Only the host's seed matters for the shared map RNG; giving every
            # player the same seed costs nothing and helps reproducibility.
            game.set_seed(seed + idx)
        game.init()
        self._games[agent] = game
        self._action_tables[agent] = _build_action_table(game, self._compound_actions)

    def _parallel(
        self,
        fn,
        actions: dict[str, int] | None,
    ) -> list[Any]:
        """Run ``fn(game, agent)`` across every player in parallel."""
        futures = [
            self._pool.submit(fn, self._games[a], a) for a in self.possible_agents
        ]
        return [f.result() for f in futures]

    def _team_deltas(
        self,
        frag_delta: dict[str, int],
        death_delta: dict[str, int],
    ) -> tuple[dict[str, int], dict[str, int]]:
        if self._teams is None:
            return {}, {}
        tf: dict[str, int] = {}
        td: dict[str, int] = {}
        for idx, agent in enumerate(self.agents):
            team = self._teams[idx]
            tf[team] = tf.get(team, 0) + frag_delta[agent]
            td[team] = td.get(team, 0) + death_delta[agent]
        return tf, td

    @staticmethod
    def _get_frags(game: vizdoom.DoomGame) -> int:
        return int(game.get_game_variable(vizdoom.GameVariable.FRAGCOUNT))

    @staticmethod
    def _get_deaths(game: vizdoom.DoomGame) -> int:
        return int(game.get_game_variable(vizdoom.GameVariable.DEATHCOUNT))

    def _read_screen(self, game: vizdoom.DoomGame) -> np.ndarray:
        state = game.get_state()
        if state is None:
            # Between episodes the state vanishes; hand back a black frame of
            # the native resolution so downstream preprocessing is well-typed.
            h, w = game.get_screen_height(), game.get_screen_width()
            return np.zeros((h, w, 3), dtype=np.uint8)
        buf = state.screen_buffer
        if buf.ndim == 3 and buf.shape[0] == 3 and buf.shape[-1] != 3:
            buf = buf.transpose(1, 2, 0)
        return buf

    def _process_obs(self, agent: str, frame: np.ndarray) -> np.ndarray:
        """Resize/grayscale/stack a raw RGB frame into the configured obs."""
        if self._resize_shape is None:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        resized = cv2.resize(
            gray,
            (self._resize_shape[1], self._resize_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        if self._num_stack <= 1:
            return resized
        stack = self._frame_stacks[agent]
        if not stack:
            for _ in range(self._num_stack):
                stack.append(resized)
        else:
            stack.append(resized)
        return np.array(stack)


# ======================================================================
# Factories
# ======================================================================


def make_1v1_env(
    *,
    scenario_cfg: str = "deathmatch.cfg",
    time_limit: float = 10.0,
    frag_limit: int = 0,
    frame_skip: int = 4,
    resize_shape: tuple[int, int] | None = (84, 84),
    num_stack: int = 4,
    window_visible: bool = False,
    port: int | None = None,
) -> DoomMultiplayerEnv:
    """Build a 2-player free-for-all deathmatch env (1v1 self-play).

    Each agent's reward is its own frag-delta minus its own death-delta per
    step, so opposing policies sum to zero-ish on average (modulo environment
    damage, suicides, etc.).
    """
    return DoomMultiplayerEnv(
        num_players=2,
        teams=None,
        scenario_cfg=scenario_cfg,
        time_limit=time_limit,
        frag_limit=frag_limit,
        frame_skip=frame_skip,
        resize_shape=resize_shape,
        num_stack=num_stack,
        window_visible=window_visible,
        port=port,
    )


def make_2v2_env(
    *,
    scenario_cfg: str = "deathmatch.cfg",
    time_limit: float = 10.0,
    frag_limit: int = 0,
    frame_skip: int = 4,
    resize_shape: tuple[int, int] | None = (84, 84),
    num_stack: int = 4,
    window_visible: bool = False,
    port: int | None = None,
) -> DoomMultiplayerEnv:
    """Build a 4-player team deathmatch env (2v2).

    Teammates share identical rewards each step: team frag-delta minus team
    death-delta. Players 0 and 1 are on team ``"red"``; players 2 and 3 on
    team ``"blue"``.
    """
    return DoomMultiplayerEnv(
        num_players=4,
        teams=["red", "red", "blue", "blue"],
        scenario_cfg=scenario_cfg,
        time_limit=time_limit,
        frag_limit=frag_limit,
        frame_skip=frame_skip,
        resize_shape=resize_shape,
        num_stack=num_stack,
        window_visible=window_visible,
        port=port,
    )
