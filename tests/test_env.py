"""Environment wrapper tests that require the vizdoom binary."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

vizdoom = pytest.importorskip("vizdoom")


def _discrete_n(space: gym.Space) -> int:
    """Narrow a generic ``Space`` to ``Discrete`` and return its size."""
    assert isinstance(space, gym.spaces.Discrete), (
        f"Expected Discrete action space, got {type(space).__name__}"
    )
    return int(space.n)


def test_make_wrapped_env_reset_shape() -> None:
    from rl_doom.env import make_wrapped_env

    env = make_wrapped_env("basic", resize_shape=(84, 84), frame_skip=4, num_stack=4)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
    finally:
        env.close()


def test_make_wrapped_env_step_contract() -> None:
    from rl_doom.env import make_wrapped_env

    env = make_wrapped_env("basic", resize_shape=(84, 84), frame_skip=4, num_stack=4)
    try:
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (4, 84, 84)
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# terminated / truncated semantics
#
# ViZDoom reports "episode finished" for both an absorbing state and a
# scenario-timeout expiry. Gymnasium (and therefore SB3's value-function
# bootstrapping) needs those kept apart: on a truncation the agent's future
# return is not zero, so the target must bootstrap V(s') instead of collapsing
# to the immediate reward.
#
# Both fixtures below are deterministic against the real binary:
#   * ``basic`` has a 300-tic timeout; at frame_skip=4 that is exactly 75 agent
#     steps, and action 0 (MOVE_LEFT) never fires, so the monster cannot die
#     and the episode can only end by timeout.
#   * ``deadly_corridor`` action 0 (MOVE_FORWARD) walks into the shotgunners
#     and dies well inside the timeout.
# ---------------------------------------------------------------------------

BASIC_TIMEOUT_STEPS = 75


def _run_until_done(env: gym.Env, action: int, max_steps: int):
    """Step *action* until the episode ends; return the final transition."""
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
    raise AssertionError(f"episode did not end within {max_steps} steps")


def test_timeout_is_truncated_not_terminated() -> None:
    """A scenario-timeout expiry must set ``truncated``, not ``terminated``."""
    from rl_doom.env import make_wrapped_env

    env = make_wrapped_env("basic", resize_shape=(84, 84), frame_skip=4, num_stack=4)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = _run_until_done(
            env, action=0, max_steps=BASIC_TIMEOUT_STEPS * 2,
        )
    finally:
        env.close()

    assert info["termination_reason"] == "timeout"
    assert truncated is True
    assert terminated is False, (
        "Timeout reported as termination — the value function will bootstrap "
        "V=0 at every timeout instead of V(s')."
    )


def test_death_is_terminated_not_truncated() -> None:
    """A real absorbing state must still set ``terminated``."""
    from rl_doom.env import make_wrapped_env

    env = make_wrapped_env(
        "deadly_corridor", resize_shape=(84, 84), frame_skip=4, num_stack=4, doom_skill=5,
    )
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = _run_until_done(env, action=0, max_steps=600)
    finally:
        env.close()

    assert info["termination_reason"] == "death"
    assert terminated is True
    assert truncated is False


def test_truncating_step_returns_last_real_frame_not_black() -> None:
    """The final observation must be a frame the agent actually saw.

    ViZDoom tears down its state once the episode finishes, so ``get_state()``
    returns ``None`` on the terminal step. Returning a black frame there would
    be harmless while every episode was ``terminated`` (the observation is
    discarded), but on a truncation SB3 feeds it to the value function to
    bootstrap ``V(s')`` — so it has to be real.
    """
    from rl_doom.env import make_wrapped_env

    env = make_wrapped_env("basic", resize_shape=(84, 84), frame_skip=4, num_stack=4)
    try:
        env.reset(seed=0)
        obs, _, _, truncated, _ = _run_until_done(
            env, action=0, max_steps=BASIC_TIMEOUT_STEPS * 2,
        )
    finally:
        env.close()

    assert truncated is True
    assert obs.any(), "terminal observation is an all-black frame"


def test_sb3_vec_env_surfaces_timelimit_truncated() -> None:
    """The truncation flag must survive the SB3 VecEnv boundary.

    This is the integration point that actually drives learning: DummyVecEnv
    translates ``truncated`` into ``info["TimeLimit.truncated"]``, which DQN's
    replay buffer reads via ``handle_timeout_termination`` and PPO reads in
    ``collect_rollouts`` alongside ``terminal_observation``.
    """
    from rl_doom.env import make_sb3_env

    vec_env = make_sb3_env("basic", n_envs=1, seed=0, frame_skip=4, num_stack=4)
    try:
        vec_env.reset()
        for _ in range(BASIC_TIMEOUT_STEPS * 2):
            _, _, dones, infos = vec_env.step(np.array([0]))
            if dones[0]:
                break
        else:
            raise AssertionError("episode did not end")
    finally:
        vec_env.close()

    assert infos[0]["TimeLimit.truncated"] is True
    terminal_obs = infos[0]["terminal_observation"]
    assert terminal_obs.shape == (4, 84, 84)
    assert terminal_obs.any(), "SB3 would bootstrap V(s') from an all-black frame"


def test_unknown_scenario_raises() -> None:
    from rl_doom.env import DoomEnv

    with pytest.raises(ValueError):
        DoomEnv(scenario="not_a_real_scenario")


def test_num_bots_rejects_negative() -> None:
    from rl_doom.env import DoomEnv

    with pytest.raises(ValueError, match=r"num_bots must be in \[0, 8\]"):
        DoomEnv(scenario="deathmatch", num_bots=-1)


def test_num_bots_rejects_above_max() -> None:
    from rl_doom.env import DoomEnv

    with pytest.raises(ValueError, match=r"num_bots must be in \[0, 8\]"):
        DoomEnv(scenario="deathmatch", num_bots=9)


def test_num_bots_accepts_max() -> None:
    from rl_doom.env import MAX_NUM_BOTS, DoomEnv

    assert MAX_NUM_BOTS == 8
    env = DoomEnv(scenario="deathmatch", num_bots=MAX_NUM_BOTS)
    try:
        assert env._num_bots == MAX_NUM_BOTS
    finally:
        env.close()


def test_reset_clears_bots_before_new_episode() -> None:
    """``reset()`` must issue ``removebots`` before ``new_episode`` so that
    ``addbot`` invocations don't accumulate across episodes and exhaust the
    map's player-start slots (raising ``No player N start`` from ZDoom)."""
    from rl_doom.env import DoomEnv

    env = DoomEnv(scenario="deathmatch", num_bots=4)
    calls: list[str] = []

    class _RecordingGame:
        """Delegating proxy that records ``send_game_command`` invocations.

        ``DoomGame`` is a pybind11 type whose methods are read-only, so the
        instance attribute cannot be monkeypatched directly. Swapping
        ``env.game`` for a proxy works because ``DoomEnv`` stores it as a plain
        Python attribute and only ever reaches it through normal attribute
        access.
        """

        def __init__(self, inner: object) -> None:
            self._inner = inner

        def send_game_command(self, cmd: str) -> None:
            calls.append(cmd)
            self._inner.send_game_command(cmd)  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    real_game = env.game
    try:
        env.game = _RecordingGame(real_game)  # type: ignore[assignment]
        env.reset()
        env.reset()
    finally:
        env.game = real_game
        env.close()

    # Every reset with num_bots>0 must call removebots exactly once before
    # the addbot burst; expect that pattern on both resets.
    assert calls.count("removebots") == 2
    assert calls.count("addbot") == 8
    # Order per reset: removebots, then 4x addbot.
    for start in (0, 5):
        assert calls[start] == "removebots"
        assert calls[start + 1 : start + 5] == ["addbot"] * 4


# ---------------------------------------------------------------------------
# Compound action space
# ---------------------------------------------------------------------------


def test_compound_action_space_expands_deadly_corridor() -> None:
    """Deadly Corridor should expose compound actions (e.g. MOVE_FORWARD+ATTACK)
    rather than the 7 one-hot single-button actions."""
    from rl_doom.env import SCENARIO_ACTION_SETS, DoomEnv

    env = DoomEnv(scenario="deadly_corridor")
    try:
        # 7 raw buttons -> curated set has 14 compound actions.
        assert _discrete_n(env.action_space) == len(SCENARIO_ACTION_SETS["deadly_corridor"])
        # At least one action must press two buttons simultaneously.
        multi = [a for a in env.available_actions if sum(a) > 1]
        assert multi, "compound action set did not produce any multi-button actions"
    finally:
        env.close()


def test_compound_action_space_opt_out_returns_one_hot() -> None:
    """With ``use_compound_actions=False`` we fall back to the legacy one-hot
    layout (one action per raw button)."""
    from rl_doom.env import DoomEnv

    env = DoomEnv(scenario="deadly_corridor", use_compound_actions=False)
    try:
        assert _discrete_n(env.action_space) == 7  # 7 raw buttons
        # Every action presses exactly one button.
        assert all(sum(a) == 1 for a in env.available_actions)
    finally:
        env.close()


def test_compound_action_space_basic_and_dtc() -> None:
    """Spot-check the other two curated scenarios."""
    from rl_doom.env import SCENARIO_ACTION_SETS, DoomEnv

    for scenario in ("basic", "defend_the_center"):
        env = DoomEnv(scenario=scenario)
        try:
            assert _discrete_n(env.action_space) == len(SCENARIO_ACTION_SETS[scenario])
            # Each scenario should include at least one "press + ATTACK" compound.
            assert any(sum(a) >= 2 for a in env.available_actions)
        finally:
            env.close()
