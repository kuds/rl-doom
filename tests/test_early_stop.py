"""Eval-plateau early stopping, including its curriculum interaction.

Motivated by the 2026-04 runs, which all peaked well before their budget and
then declined — DQN/deadly_corridor lost 40% of its eval reward over the 2.1M
steps after its peak at 400k.

The subtle requirement is the curriculum one: promoting a stage makes the task
harder, so eval reward legitimately drops. A naive "no improvement in N evals"
counter reads that as a plateau and stops training exactly when the agent has
been handed the harder problem it was being prepared for.
"""

from __future__ import annotations

from typing import Any

import pytest

from rl_doom.early_stop import StopOnEvalPlateau


class _FakeEvalCallback:
    """Stands in for EvalCallback's two consumed attributes."""

    def __init__(self) -> None:
        self.evaluations_timesteps: list[int] = []
        self.last_mean_reward: float = 0.0

    def report(self, step: int, reward: float) -> None:
        self.evaluations_timesteps.append(step)
        self.last_mean_reward = reward


class _FakeCurriculum:
    """Exposes only the ``promotions`` list the stopper reads."""

    def __init__(self) -> None:
        self.promotions: list[dict[str, Any]] = [{"trigger": "initial"}]

    def promote(self) -> None:
        self.promotions.append({"trigger": "promotion"})


def _drive(
    stopper: StopOnEvalPlateau,
    eval_cb: _FakeEvalCallback,
    rewards: list[float],
    *,
    curriculum: _FakeCurriculum | None = None,
    promote_at_index: int | None = None,
) -> int:
    """Feed *rewards* as successive evals; return how many were consumed."""
    step = 0
    for i, reward in enumerate(rewards):
        if promote_at_index is not None and i == promote_at_index:
            assert curriculum is not None
            curriculum.promote()
        step += 1000
        eval_cb.report(step, reward)
        stopper.num_timesteps = step
        if not stopper._on_step():
            return i + 1
    return len(rewards)


def _make(eval_cb: _FakeEvalCallback, **kw: Any) -> StopOnEvalPlateau:
    stopper = StopOnEvalPlateau(eval_cb, verbose=0, **kw)  # type: ignore[arg-type]
    stopper.num_timesteps = 0
    stopper._on_training_start()
    return stopper


def test_rejects_nonsense_patience() -> None:
    with pytest.raises(ValueError, match="patience must be >= 1"):
        StopOnEvalPlateau(_FakeEvalCallback(), patience=0)  # type: ignore[arg-type]


def test_improving_run_is_never_stopped() -> None:
    eval_cb = _FakeEvalCallback()
    stopper = _make(eval_cb, patience=3, min_evals=2)
    consumed = _drive(stopper, eval_cb, [float(i) for i in range(30)])
    assert consumed == 30
    assert stopper.stopped_early is False


def test_stops_after_patience_evals_without_improvement() -> None:
    eval_cb = _FakeEvalCallback()
    stopper = _make(eval_cb, patience=3, min_evals=2)
    # Best is 10.0 at index 0, then five worse evals; stop on the 3rd.
    consumed = _drive(stopper, eval_cb, [10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    assert stopper.stopped_early is True
    assert consumed == 4  # the best eval plus three without improvement
    assert stopper.stopped_at_step == 4000


def test_min_evals_defers_stopping_through_a_slow_start() -> None:
    eval_cb = _FakeEvalCallback()
    stopper = _make(eval_cb, patience=2, min_evals=6)
    consumed = _drive(stopper, eval_cb, [5.0] + [1.0] * 9)
    # Patience alone would have fired at eval 3; min_evals holds it to 6.
    assert stopper.stopped_early is True
    assert consumed == 6


def test_min_improvement_ignores_noise() -> None:
    """Tiny gains must not keep resetting patience forever."""
    eval_cb = _FakeEvalCallback()
    stopper = _make(eval_cb, patience=3, min_evals=2, min_improvement=1.0)
    # Each eval improves, but by less than min_improvement.
    consumed = _drive(stopper, eval_cb, [10.0, 10.1, 10.2, 10.3, 10.4, 10.5])
    assert stopper.stopped_early is True
    assert consumed == 4


def test_curriculum_promotion_resets_patience() -> None:
    """The regression this class exists for.

    Reward drops after a promotion because the task got harder. Without a
    reset, the pre-promotion best keeps the counter running and training stops
    on the harder rung it was being prepared for.
    """
    eval_cb = _FakeEvalCallback()
    curriculum = _FakeCurriculum()
    stopper = _make(eval_cb, patience=3, min_evals=2, curriculum_cb=curriculum)

    # Climb to 100 on the easy rung, promote, then score lower but improving.
    rewards = [50.0, 80.0, 100.0, 40.0, 45.0, 50.0, 55.0, 60.0]
    consumed = _drive(
        stopper, eval_cb, rewards, curriculum=curriculum, promote_at_index=3,
    )
    assert consumed == len(rewards), "stopped after a promotion dropped the reward"
    assert stopper.stopped_early is False


def test_still_stops_on_a_plateau_after_a_promotion() -> None:
    """The reset must not disable stopping permanently."""
    eval_cb = _FakeEvalCallback()
    curriculum = _FakeCurriculum()
    stopper = _make(eval_cb, patience=3, min_evals=2, curriculum_cb=curriculum)

    rewards = [50.0, 100.0, 40.0, 39.0, 38.0, 37.0, 36.0, 35.0]
    consumed = _drive(
        stopper, eval_cb, rewards, curriculum=curriculum, promote_at_index=2,
    )
    assert stopper.stopped_early is True
    assert consumed < len(rewards)


def test_ignores_repeated_reads_of_the_same_eval() -> None:
    """_on_step runs every step; only a new eval timestep counts."""
    eval_cb = _FakeEvalCallback()
    stopper = _make(eval_cb, patience=2, min_evals=1)
    eval_cb.report(1000, 10.0)
    stopper.num_timesteps = 1000
    for _ in range(50):
        assert stopper._on_step() is True
    assert stopper.stopped_early is False


def test_no_evals_yet_is_a_noop() -> None:
    eval_cb = _FakeEvalCallback()
    stopper = _make(eval_cb, patience=1, min_evals=1)
    assert stopper._on_step() is True
