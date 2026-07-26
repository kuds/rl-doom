"""Stop a run once evaluation performance has stopped improving.

Why this exists
---------------

Every completed run in this project peaked well before its budget and then
got worse. From the 2026-04 runs:

    DQN  / deadly_corridor    best 2153.19 @   400,000 -> final 1282.70 (-40%)
    PPO  / deathmatch         best  193.90 @   250,000 -> final  166.50 (-14%)
    PPO  / deadly_corridor    best 2281.30 @ 1,675,000 -> final 2161.11 ( -5%)

The first two spent 84% and 90% of their budget making the policy worse — for
DQN that is roughly 3.7 of 4.4 GPU-hours. Results were never at risk
(``EvalCallback`` keeps the best checkpoint), but the compute was, and a
24-cell matrix multiplies it.

Curriculum interaction
----------------------

The obvious implementation — SB3's ``StopTrainingOnNoModelImprovement`` — is
wrong here. Promoting a curriculum stage makes the task harder, so eval reward
*should* drop, and a plain "no improvement in N evals" counter reads that as a
plateau and stops training exactly when the agent has been handed the harder
problem it was being prepared for. This stopper watches the curriculum and
resets its baseline on every promotion, so patience is always measured within
a single rung.
"""

from __future__ import annotations

from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class StopOnEvalPlateau(BaseCallback):
    """End training after ``patience`` consecutive evals without improvement.

    Parameters
    ----------
    eval_cb :
        The :class:`EvalCallback` already registered on the model. Read for
        ``evaluations_timesteps`` and ``last_mean_reward``, mirroring how
        :class:`~rl_doom.curriculum.SkillCurriculumCallback` consumes it.
    patience :
        Evals without a new best before stopping.
    min_evals :
        Never stop before this many evals have happened on the current rung,
        so a slow start is not mistaken for a plateau.
    min_improvement :
        How much better than the running best counts as improvement. Guards
        against noise ratcheting the bar up by fractions.
    curriculum_cb :
        Optional :class:`~rl_doom.curriculum.SkillCurriculumCallback`. When
        given, a promotion resets the baseline and the counter — a harder rung
        legitimately scores lower, and that must not read as a plateau.
    """

    def __init__(
        self,
        eval_cb: EvalCallback,
        *,
        patience: int = 20,
        min_evals: int = 10,
        min_improvement: float = 0.0,
        curriculum_cb: Any | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience!r}")
        self._eval_cb = eval_cb
        self._patience = int(patience)
        self._min_evals = int(min_evals)
        self._min_improvement = float(min_improvement)
        self._curriculum_cb = curriculum_cb

        self._last_seen_eval_ts: int = -1
        self._best: float | None = None
        self._evals_without_improvement = 0
        self._evals_on_rung = 0
        self._promotions_seen = 0
        # Public, for the run summary / tests.
        self.stopped_early = False
        self.stopped_at_step: int | None = None

    def _promotion_count(self) -> int:
        if self._curriculum_cb is None:
            return 0
        return len(getattr(self._curriculum_cb, "promotions", ()) or ())

    def _reset_rung(self) -> None:
        self._best = None
        self._evals_without_improvement = 0
        self._evals_on_rung = 0

    def _on_training_start(self) -> None:
        # ``promotions`` gets its "initial" entry at training start; count it
        # now so it is not mistaken for a promotion on the first eval.
        self._promotions_seen = self._promotion_count()

    def _on_step(self) -> bool:
        evals = getattr(self._eval_cb, "evaluations_timesteps", None)
        if not evals:
            return True
        latest_ts = int(evals[-1])
        if latest_ts == self._last_seen_eval_ts:
            return True
        self._last_seen_eval_ts = latest_ts

        # A promotion since the last eval means the task just changed. Start
        # measuring the new rung from scratch rather than against a best score
        # earned on an easier one.
        promotions = self._promotion_count()
        if promotions != self._promotions_seen:
            self._promotions_seen = promotions
            self._reset_rung()
            if self.verbose:
                print("[early-stop] curriculum promoted — patience reset")

        mean_reward = float(self._eval_cb.last_mean_reward)
        self._evals_on_rung += 1

        if self._best is None or mean_reward > self._best + self._min_improvement:
            self._best = mean_reward
            self._evals_without_improvement = 0
            return True

        self._evals_without_improvement += 1
        if (
            self._evals_on_rung >= self._min_evals
            and self._evals_without_improvement >= self._patience
        ):
            self.stopped_early = True
            self.stopped_at_step = int(self.num_timesteps)
            if self.verbose:
                print(
                    f"[early-stop] no improvement over {self._patience} evals "
                    f"(best {self._best:.2f}, latest {mean_reward:.2f}); "
                    f"stopping at step {self.num_timesteps:,}. The best "
                    f"checkpoint is already saved by EvalCallback.",
                )
            return False  # SB3 ends training when a callback returns False
        return True
