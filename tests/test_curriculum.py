"""Unit tests for :mod:`rl_doom.curriculum`.

These tests avoid ViZDoom entirely — the curriculum callback doesn't
need a real env, only a vec-env-like object with ``envs`` holding things
that expose ``game.set_doom_skill``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# SB3 is an optional runtime dep (e.g. on CI runners that don't install
# training deps). Skip the whole module when it's unavailable so the rest
# of the suite still runs.
pytest.importorskip("stable_baselines3")

from rl_doom.curriculum import (  # noqa: E402 — import after skip guard
    CurriculumStage,
    SkillCurriculumCallback,
    _iter_inner_envs,
    _set_skill_on_vec_env,
    parse_curriculum_config,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeGame:
    """Stand-in for ``vizdoom.DoomGame`` that just records ``set_doom_skill``."""

    def __init__(self) -> None:
        self.skill: int | None = None
        self.calls: list[int] = []

    def set_doom_skill(self, skill: int) -> None:
        self.skill = skill
        self.calls.append(skill)


class _FakeDoomEnv:
    def __init__(self) -> None:
        self.game = _FakeGame()
        self._doom_skill: int | None = None


class _FakeWrapper:
    """One-level Gym-style wrapper so the callback has to walk ``.env``."""

    def __init__(self, inner: object) -> None:
        self.env = inner


class _FakeVecEnv:
    def __init__(self, envs: list[object]) -> None:
        self.envs = envs


class _FakeVecNormalize:
    """Mimics SB3 ``VecNormalize`` — wraps another VecEnv on ``.venv``."""

    def __init__(self, venv: _FakeVecEnv) -> None:
        self.venv = venv


class _FakeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, object]] = []

    def record(self, key: str, value: object) -> None:
        self.records.append((key, value))


class _FakeEvalCallback:
    """Minimal :class:`stable_baselines3.common.callbacks.EvalCallback` stub."""

    def __init__(self, eval_env: object | None = None) -> None:
        self.evaluations_timesteps: list[int] = []
        self.last_mean_reward: float = 0.0
        self.eval_env = eval_env

    def push_eval(self, timestep: int, mean_reward: float) -> None:
        self.evaluations_timesteps.append(timestep)
        self.last_mean_reward = mean_reward


def _prime(
    callback: SkillCurriculumCallback,
    *,
    vec_env: object,
    num_timesteps: int = 0,
) -> None:
    """Attach the minimum state SB3 would normally inject.

    In SB3 2.x both ``BaseCallback.training_env`` and
    ``BaseCallback.logger`` are properties that delegate to the attached
    model, so we can't set them directly on the callback — the fake
    model here exposes ``get_env`` and ``logger`` which the properties
    return.
    """
    callback.model = SimpleNamespace(  # type: ignore[assignment]
        num_timesteps=num_timesteps,
        get_env=lambda: vec_env,
        logger=_FakeLogger(),
    )


# ---------------------------------------------------------------------------
# parse_curriculum_config
# ---------------------------------------------------------------------------


def test_parse_curriculum_returns_none_when_absent() -> None:
    assert parse_curriculum_config(None) is None
    assert parse_curriculum_config({}) is None


def test_parse_curriculum_explicit_disable() -> None:
    cfg = {"enabled": False, "stages": [{"skill": 1, "promote_at": 10}]}
    assert parse_curriculum_config(cfg) is None


def test_parse_curriculum_happy_path() -> None:
    cfg = {
        "stages": [
            {"skill": 1, "promote_at": 40.0},
            {"skill": 2, "promote_at": 60.0},
            {"skill": 3, "promote_at": None},
        ],
    }
    stages = parse_curriculum_config(cfg)
    assert stages is not None
    assert len(stages) == 3
    assert stages[0] == CurriculumStage(skill=1, promote_at=40.0)
    assert stages[-1] == CurriculumStage(skill=3, promote_at=None)


def test_parse_curriculum_requires_stages() -> None:
    with pytest.raises(ValueError, match="non-empty 'stages'"):
        parse_curriculum_config({"enabled": True})


def test_parse_curriculum_requires_skill_field() -> None:
    with pytest.raises(ValueError, match="missing required 'skill'"):
        parse_curriculum_config({"stages": [{"promote_at": 10}]})


def test_parse_curriculum_non_terminal_promote_at_required() -> None:
    cfg = {
        "stages": [
            {"skill": 1, "promote_at": None},
            {"skill": 2, "promote_at": None},
        ],
    }
    with pytest.raises(ValueError, match="non-terminal"):
        parse_curriculum_config(cfg)


def test_curriculum_stage_rejects_out_of_range_skill() -> None:
    with pytest.raises(ValueError):
        CurriculumStage(skill=0, promote_at=1.0)
    with pytest.raises(ValueError):
        CurriculumStage(skill=6, promote_at=1.0)


# ---------------------------------------------------------------------------
# _iter_inner_envs / _set_skill_on_vec_env
# ---------------------------------------------------------------------------


def test_iter_inner_envs_unwraps_vec_normalize() -> None:
    e1, e2 = _FakeDoomEnv(), _FakeDoomEnv()
    vec = _FakeVecEnv([e1, e2])
    wrapped = _FakeVecNormalize(vec)
    assert list(_iter_inner_envs(wrapped)) == [e1, e2]


def test_set_skill_walks_gym_wrapper_chain() -> None:
    raw = _FakeDoomEnv()
    wrapped_once = _FakeWrapper(raw)
    wrapped_twice = _FakeWrapper(wrapped_once)
    vec = _FakeVecEnv([wrapped_twice])

    _set_skill_on_vec_env(vec, 2)
    assert raw.game.skill == 2
    assert raw._doom_skill == 2


# ---------------------------------------------------------------------------
# SkillCurriculumCallback
# ---------------------------------------------------------------------------


def _build_callback(
    stages: list[CurriculumStage],
    *,
    min_gap: int = 1,
    sync_eval: bool = True,
) -> tuple[SkillCurriculumCallback, _FakeEvalCallback, _FakeDoomEnv, _FakeDoomEnv]:
    train_env = _FakeDoomEnv()
    eval_env = _FakeDoomEnv()
    eval_cb = _FakeEvalCallback(eval_env=_FakeVecEnv([eval_env]))
    cb = SkillCurriculumCallback(
        eval_cb,  # type: ignore[arg-type]
        stages,
        min_evals_between_promotions=min_gap,
        sync_eval_env=sync_eval,
        verbose=0,
    )
    _prime(cb, vec_env=_FakeVecEnv([train_env]))
    return cb, eval_cb, train_env, eval_env


def test_on_training_start_applies_initial_skill_to_train_and_eval() -> None:
    stages = [
        CurriculumStage(skill=1, promote_at=40.0),
        CurriculumStage(skill=2, promote_at=60.0),
        CurriculumStage(skill=3, promote_at=None),
    ]
    cb, eval_cb, train_env, eval_env = _build_callback(stages)
    cb._on_training_start()

    assert train_env.game.skill == 1
    assert eval_env.game.skill == 1
    assert cb.promotions == [
        {"step": 0, "skill": 1, "trigger": "initial", "eval_mean_reward": None},
    ]


def test_promotion_fires_when_eval_reward_exceeds_threshold() -> None:
    stages = [
        CurriculumStage(skill=1, promote_at=40.0),
        CurriculumStage(skill=2, promote_at=60.0),
        CurriculumStage(skill=3, promote_at=None),
    ]
    cb, eval_cb, train_env, eval_env = _build_callback(stages)
    cb._on_training_start()

    # First eval below threshold: no promotion.
    eval_cb.push_eval(timestep=1000, mean_reward=20.0)
    cb.model.num_timesteps = 1000
    cb._on_step()
    assert cb.current_skill == 1

    # Second eval above threshold: promote to stage 2 (skill 2).
    eval_cb.push_eval(timestep=2000, mean_reward=55.0)
    cb.model.num_timesteps = 2000
    cb._on_step()
    assert cb.current_skill == 2
    assert train_env.game.skill == 2
    assert eval_env.game.skill == 2
    assert cb.promotions[-1]["trigger"] == "promotion"
    assert cb.promotions[-1]["skill"] == 2


def test_min_evals_between_promotions_blocks_rapid_double_promotion() -> None:
    stages = [
        CurriculumStage(skill=1, promote_at=40.0),
        CurriculumStage(skill=2, promote_at=60.0),
        CurriculumStage(skill=3, promote_at=None),
    ]
    cb, eval_cb, *_ = _build_callback(stages, min_gap=2)
    cb._on_training_start()

    # First eval clears stage 1's threshold — but min_gap=2 means we need
    # two evals on the current rung before allowing any promotion, so
    # this one just counts and the skill stays at 1.
    eval_cb.push_eval(timestep=1000, mean_reward=50.0)
    cb.model.num_timesteps = 1000
    cb._on_step()
    assert cb.current_skill == 1

    # Second eval: now we have two evals and we're over the threshold,
    # so promotion happens.
    eval_cb.push_eval(timestep=2000, mean_reward=50.0)
    cb.model.num_timesteps = 2000
    cb._on_step()
    assert cb.current_skill == 2


def test_terminal_stage_never_promotes() -> None:
    stages = [
        CurriculumStage(skill=1, promote_at=10.0),
        CurriculumStage(skill=2, promote_at=None),
    ]
    cb, eval_cb, train_env, _ = _build_callback(stages)
    cb._on_training_start()

    eval_cb.push_eval(timestep=1000, mean_reward=500.0)
    cb.model.num_timesteps = 1000
    cb._on_step()
    assert cb.current_skill == 2

    # Pushing a second very-high eval should be a no-op; we're already
    # at the terminal stage.
    eval_cb.push_eval(timestep=2000, mean_reward=5000.0)
    cb.model.num_timesteps = 2000
    cb._on_step()
    assert cb.current_skill == 2
    assert cb.is_terminal is True
    # Exactly one "promotion" trigger was recorded, plus the initial.
    promotions = [p for p in cb.promotions if p["trigger"] == "promotion"]
    assert len(promotions) == 1


def test_sync_eval_env_false_leaves_eval_on_initial_skill() -> None:
    stages = [
        CurriculumStage(skill=1, promote_at=40.0),
        CurriculumStage(skill=2, promote_at=None),
    ]
    cb, eval_cb, train_env, eval_env = _build_callback(stages, sync_eval=False)
    cb._on_training_start()
    assert eval_env.game.skill == 1  # initial still applied

    eval_cb.push_eval(timestep=1000, mean_reward=100.0)
    cb.model.num_timesteps = 1000
    cb._on_step()
    # Training env promoted, eval env stayed behind.
    assert train_env.game.skill == 2
    assert eval_env.game.skill == 1
    assert cb.current_skill == 2


def test_on_step_no_op_when_eval_has_not_run() -> None:
    stages = [
        CurriculumStage(skill=1, promote_at=40.0),
        CurriculumStage(skill=2, promote_at=None),
    ]
    cb, _, train_env, _ = _build_callback(stages)
    cb._on_training_start()
    # No eval pushed yet.
    assert cb._on_step() is True
    assert cb.current_skill == 1


def test_same_eval_timestep_processed_once() -> None:
    stages = [
        CurriculumStage(skill=1, promote_at=40.0),
        CurriculumStage(skill=2, promote_at=None),
    ]
    cb, eval_cb, *_ = _build_callback(stages)
    cb._on_training_start()

    eval_cb.push_eval(timestep=1000, mean_reward=10.0)
    cb.model.num_timesteps = 1000
    cb._on_step()
    cb._on_step()  # same timestep — should not double-count against min_gap

    # One evaluation only; below threshold; still on stage 1.
    assert cb._evals_since_promotion == 1
    assert cb.current_skill == 1
