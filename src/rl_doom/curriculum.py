"""Skill-based curriculum learning for ViZDoom scenarios.

Provides :class:`SkillCurriculumCallback`, a Stable-Baselines3 callback that
monitors an :class:`~stable_baselines3.common.callbacks.EvalCallback` and
promotes the training envs' ViZDoom ``doom_skill`` each time the eval mean
reward clears a per-stage threshold.

Why this helps Deadly Corridor (the headline use case):

* The scenario's reward is dominated by the death penalty; on skill 3 the
  imps kill the agent before it can discover the "push forward + shoot"
  gradient.
* Starting on skill 1 lets the agent survive long enough to learn the
  distance-to-vest shaping, then ramping the difficulty back up fine-tunes
  combat without losing the navigation prior.

Design notes:

* Uses ``DoomGame.set_doom_skill`` which takes effect on the *next*
  ``new_episode()`` — no forced reset required, the active episode finishes
  at the old difficulty.
* Relies on the public ``EvalCallback`` attributes
  ``evaluations_timesteps`` and ``last_mean_reward``, so the callback must
  be registered **after** the ``EvalCallback`` in the callback list (SB3
  fires callbacks in registration order, and we need the eval result from
  the current step to be available before we inspect it).
* Propagates skill changes to both the training vec env and the eval vec
  env (via the attached ``EvalCallback``) so the promotion threshold is
  measured on the same difficulty the agent is training on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


@dataclass(frozen=True)
class CurriculumStage:
    """One rung of a skill curriculum.

    ``promote_at`` is the eval-reward threshold that triggers promotion to
    the *next* stage. ``None`` marks a terminal stage — once the curriculum
    reaches it, no further promotions occur even if the threshold would be
    exceeded.
    """

    skill: int
    promote_at: float | None

    def __post_init__(self) -> None:
        if not 1 <= self.skill <= 5:
            raise ValueError(
                f"CurriculumStage.skill must be in [1, 5], got {self.skill!r}",
            )


def parse_curriculum_config(cfg: dict[str, Any] | None) -> list[CurriculumStage] | None:
    """Translate a YAML ``curriculum:`` block into :class:`CurriculumStage` rungs.

    Expected shape::

        curriculum:
          stages:
            - {skill: 1, promote_at: 50.0}
            - {skill: 2, promote_at: 80.0}
            - {skill: 3, promote_at: null}   # terminal

    Returns ``None`` when the block is absent or explicitly disabled
    (``enabled: false``), so callers can use truthiness to decide whether
    to attach the callback.
    """
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None
    raw_stages = cfg.get("stages")
    if not raw_stages:
        raise ValueError(
            "curriculum block must define a non-empty 'stages' list",
        )
    stages: list[CurriculumStage] = []
    for i, item in enumerate(raw_stages):
        if "skill" not in item:
            raise ValueError(
                f"curriculum.stages[{i}] missing required 'skill' field",
            )
        promote_at = item.get("promote_at")
        stages.append(
            CurriculumStage(
                skill=int(item["skill"]),
                promote_at=None if promote_at is None else float(promote_at),
            ),
        )
    # The terminal stage's promote_at is ignored, so we don't require it to
    # be None — but warn-style guard: non-terminal stages must set a
    # threshold, otherwise the curriculum would be stuck forever.
    for i, stage in enumerate(stages[:-1]):
        if stage.promote_at is None:
            raise ValueError(
                f"curriculum.stages[{i}] is non-terminal and must set 'promote_at'",
            )
    return stages


class SkillCurriculumCallback(BaseCallback):
    """Promote ``doom_skill`` on all training/eval envs based on eval reward.

    Parameters
    ----------
    eval_cb :
        The :class:`EvalCallback` already registered on the model. The
        curriculum reads its ``evaluations_timesteps`` and
        ``last_mean_reward`` to decide when to promote.
    stages :
        Ordered list of :class:`CurriculumStage`. The first stage's skill
        is applied to every env at ``_on_training_start``; subsequent
        stages are activated when the previous stage's ``promote_at`` is
        exceeded.
    min_evals_between_promotions :
        Require at least this many successful evaluations on the current
        rung before allowing a promotion, to avoid jumping multiple stages
        on a single lucky rollout.
    sync_eval_env :
        When ``True`` (default) the eval env's skill is kept in lock-step
        with the training env. When ``False`` the eval env stays on the
        initial skill — useful when you want every checkpoint evaluated on
        a fixed difficulty for cross-run comparison.
    """

    def __init__(
        self,
        eval_cb: EvalCallback,
        stages: list[CurriculumStage],
        *,
        min_evals_between_promotions: int = 1,
        sync_eval_env: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        if not stages:
            raise ValueError("stages must be non-empty")
        self._eval_cb = eval_cb
        self._stages = stages
        self._idx = 0
        self._min_gap = max(1, int(min_evals_between_promotions))
        self._sync_eval_env = sync_eval_env
        self._last_seen_eval_ts: int = -1
        self._evals_since_promotion = 0
        # Populated at _on_training_start so tests (and post-hoc summaries)
        # can inspect the full promotion timeline.
        self.promotions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public introspection helpers
    # ------------------------------------------------------------------

    @property
    def current_skill(self) -> int:
        return self._stages[self._idx].skill

    @property
    def current_stage_index(self) -> int:
        return self._idx

    @property
    def is_terminal(self) -> bool:
        return self._idx >= len(self._stages) - 1

    # ------------------------------------------------------------------
    # SB3 callback protocol
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        # Always seed the eval env with the initial skill even when
        # ``sync_eval_env=False``; that flag only gates future promotions.
        self._apply_skill(self.current_skill, sync_eval=True)
        self.promotions.append(
            {
                "step": int(self.num_timesteps),
                "skill": self.current_skill,
                "trigger": "initial",
                "eval_mean_reward": None,
            },
        )
        if self.verbose:
            print(
                f"[curriculum] start skill={self.current_skill} "
                f"(stage 1 / {len(self._stages)})",
            )

    def _on_step(self) -> bool:
        evals = getattr(self._eval_cb, "evaluations_timesteps", None)
        if not evals:
            return True
        latest_ts = int(evals[-1])
        if latest_ts == self._last_seen_eval_ts:
            return True
        # A fresh eval result is available.
        self._last_seen_eval_ts = latest_ts
        self._evals_since_promotion += 1
        self.logger.record("curriculum/skill", self.current_skill)
        self.logger.record("curriculum/stage_index", self._idx)

        if self.is_terminal:
            return True
        threshold = self._stages[self._idx].promote_at
        assert threshold is not None  # non-terminal stages are validated to have one
        mean_r = float(self._eval_cb.last_mean_reward)
        if mean_r >= threshold and self._evals_since_promotion >= self._min_gap:
            self._idx += 1
            new_skill = self.current_skill
            self._apply_skill(new_skill)
            self._evals_since_promotion = 0
            self.promotions.append(
                {
                    "step": int(self.num_timesteps),
                    "skill": new_skill,
                    "trigger": "promotion",
                    "eval_mean_reward": mean_r,
                },
            )
            if self.verbose:
                print(
                    f"[curriculum] step={self.num_timesteps} "
                    f"eval_mean={mean_r:.2f} >= {threshold:.2f} "
                    f"-> promote to skill {new_skill} "
                    f"(stage {self._idx + 1} / {len(self._stages)})",
                )
            self.logger.record("curriculum/skill", new_skill)
            self.logger.record("curriculum/stage_index", self._idx)
            self.logger.record("curriculum/promotion_step", self.num_timesteps)
        return True

    # ------------------------------------------------------------------
    # Env-walking helpers
    # ------------------------------------------------------------------

    def _apply_skill(self, skill: int, *, sync_eval: bool | None = None) -> None:
        """Set ``doom_skill`` on every underlying DoomEnv (train + eval).

        ``sync_eval`` overrides ``self._sync_eval_env`` for this call — the
        initial application at ``_on_training_start`` always propagates to
        the eval env regardless of the instance flag.
        """
        _set_skill_on_vec_env(self.training_env, skill)
        propagate = self._sync_eval_env if sync_eval is None else sync_eval
        if propagate:
            eval_env = getattr(self._eval_cb, "eval_env", None)
            if eval_env is not None:
                _set_skill_on_vec_env(eval_env, skill)


def _iter_inner_envs(vec_env: Any) -> Iterable[Any]:
    """Yield the per-worker envs held by a (possibly wrapped) vec env.

    Handles the two wrapper stacks used in :mod:`rl_doom.sb3_utils`:

    * Plain ``DummyVecEnv`` — the envs live on ``.envs``.
    * ``VecNormalize`` wrapping ``DummyVecEnv`` — unwrap to ``.venv`` first.

    Anything else is returned as a single-element iterable so the caller
    can still walk the wrapper chain down to ``DoomEnv``.
    """
    inner = vec_env
    # Peel VecNormalize (and any future VecEnvWrapper) by walking .venv.
    while hasattr(inner, "venv") and not hasattr(inner, "envs"):
        inner = inner.venv
    envs = getattr(inner, "envs", None)
    if envs is None:
        # Fallback: treat the wrapper itself as the sole env. This keeps
        # the helper usable under unit-test doubles that don't implement
        # the full VecEnv API.
        yield inner
        return
    yield from envs


def _set_skill_on_vec_env(vec_env: Any, skill: int) -> None:
    """Walk each worker env down to its ``DoomEnv`` and set the skill."""
    for env in _iter_inner_envs(vec_env):
        base = env
        # Gym wrappers expose ``.env``; the chain terminates at the raw
        # DoomEnv which carries ``game`` (a ``vizdoom.DoomGame``).
        while hasattr(base, "env") and not hasattr(base, "game"):
            base = base.env
        game = getattr(base, "game", None)
        if game is None:
            continue  # e.g. a test double that isn't a real DoomEnv
        set_skill = getattr(game, "set_doom_skill", None)
        if set_skill is None:
            continue
        set_skill(skill)
        # Mirror the stored skill so DoomEnv.__repr__/introspection sees it.
        if hasattr(base, "_doom_skill"):
            base._doom_skill = skill
