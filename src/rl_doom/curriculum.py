"""Curriculum learning for ViZDoom scenarios.

Provides :class:`SkillCurriculumCallback`, a Stable-Baselines3 callback that
monitors an :class:`~stable_baselines3.common.callbacks.EvalCallback` and
promotes one or both of two difficulty knobs whenever the eval mean reward
clears a per-stage threshold:

* ``doom_skill`` — ViZDoom's 1..5 monster-AI aggressiveness setting.
  Used by scenarios where enemies are pre-placed monsters
  (deadly_corridor, defend_the_center).
* ``num_bots`` — count of ZDoom AI bots spawned via ``addbot`` at each
  ``reset()``. Used by deathmatch-style maps where there are no
  pre-placed monsters and opponents are player-style bots.

Why this helps — two headline use cases:

* **Deadly Corridor** (skill curriculum). The reward is dominated by
  the death penalty; on skill 3 the imps kill the agent before it can
  discover the "push forward + shoot" gradient. Starting on skill 1
  lets the agent survive long enough to learn the distance-to-vest
  shaping, then ramping the difficulty back up fine-tunes combat
  without losing the navigation prior.
* **Deathmatch** (bot-count curriculum). An 8-bot free-for-all with
  no reward shaping is extremely sparse for a fresh policy. Starting
  with 2 bots gives the agent frequent combat encounters, then
  scaling up to 4 and 8 grows the difficulty as the policy hardens.

Design notes:

* ``DoomGame.set_doom_skill`` takes effect on the next
  ``new_episode()`` — no forced reset required, the active episode
  finishes at the old difficulty.
* ``num_bots`` changes take effect on the next ``reset()`` because
  :class:`rl_doom.env.DoomEnv` re-issues ``addbot`` commands after
  each ``new_episode``; the active episode keeps its original count.
* Relies on the public ``EvalCallback`` attributes
  ``evaluations_timesteps`` and ``last_mean_reward``, so the callback
  must be registered **after** the ``EvalCallback`` in the callback
  list (SB3 fires callbacks in registration order).
* Propagates changes to both the training vec env and the eval vec
  env (via the attached ``EvalCallback``) so the promotion threshold
  is measured on the same difficulty the agent is training on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from rl_doom.scenario_limits import MAX_NUM_BOTS


@dataclass(frozen=True)
class CurriculumStage:
    """One rung of a curriculum.

    A stage applies one or both of two difficulty knobs when it becomes
    active:

    * ``skill`` — ViZDoom's ``doom_skill`` (1..5), controls monster AI
      aggressiveness. Used by deadly_corridor / defend_the_center.
    * ``num_bots`` — count of ZDoom AI bots spawned via ``addbot`` at
      each ``reset()``, capped at :data:`~rl_doom.scenario_limits.MAX_NUM_BOTS`.
      Used by deathmatch-style maps where the enemies are player-style
      bots, not scenario monsters.

    At least one of the two must be set. ``promote_at`` is the
    eval-reward threshold that triggers promotion to the *next* stage;
    ``None`` marks a terminal stage.
    """

    skill: int | None = None
    num_bots: int | None = None
    promote_at: float | None = None

    def __post_init__(self) -> None:
        if self.skill is None and self.num_bots is None:
            raise ValueError(
                "CurriculumStage must set at least one of 'skill' or 'num_bots'",
            )
        if self.skill is not None and not 1 <= self.skill <= 5:
            raise ValueError(
                f"CurriculumStage.skill must be in [1, 5], got {self.skill!r}",
            )
        if self.num_bots is not None and not 0 <= self.num_bots <= MAX_NUM_BOTS:
            raise ValueError(
                f"CurriculumStage.num_bots must be in [0, {MAX_NUM_BOTS}], "
                f"got {self.num_bots!r}",
            )


def parse_curriculum_config(cfg: dict[str, Any] | None) -> list[CurriculumStage] | None:
    """Translate a YAML ``curriculum:`` block into :class:`CurriculumStage` rungs.

    Expected shapes (pick one or mix within a single curriculum)::

        # Skill curriculum (deadly_corridor):
        curriculum:
          stages:
            - {skill: 1, promote_at: 50.0}
            - {skill: 2, promote_at: 80.0}
            - {skill: 3, promote_at: null}

        # Bot-count curriculum (deathmatch):
        curriculum:
          stages:
            - {num_bots: 2, promote_at: 3.0}
            - {num_bots: 4, promote_at: 5.0}
            - {num_bots: 8, promote_at: null}

    Stages may also set both fields at once if the scenario benefits
    from ramping both knobs together. Returns ``None`` when the block
    is absent or explicitly disabled (``enabled: false``) so callers
    can use truthiness to decide whether to attach the callback.
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
        if "skill" not in item and "num_bots" not in item:
            raise ValueError(
                f"curriculum.stages[{i}] must set 'skill' and/or 'num_bots'",
            )
        promote_at = item.get("promote_at")
        stages.append(
            CurriculumStage(
                skill=int(item["skill"]) if "skill" in item else None,
                num_bots=int(item["num_bots"]) if "num_bots" in item else None,
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
    def current_stage(self) -> CurriculumStage:
        return self._stages[self._idx]

    @property
    def current_skill(self) -> int | None:
        return self.current_stage.skill

    @property
    def current_num_bots(self) -> int | None:
        return self.current_stage.num_bots

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
        # Always seed the eval env with the initial stage even when
        # ``sync_eval_env=False``; that flag only gates future promotions.
        self._apply_stage(self.current_stage, sync_eval=True)
        self.promotions.append(self._promotion_entry(trigger="initial", mean_r=None))
        if self.verbose:
            print(
                f"[curriculum] start {self._describe_stage(self.current_stage)} "
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
        self._log_current_stage()

        if self.is_terminal:
            return True
        threshold = self._stages[self._idx].promote_at
        assert threshold is not None  # non-terminal stages are validated to have one
        mean_r = float(self._eval_cb.last_mean_reward)
        if mean_r >= threshold and self._evals_since_promotion >= self._min_gap:
            self._idx += 1
            new_stage = self.current_stage
            self._apply_stage(new_stage)
            self._evals_since_promotion = 0
            self.promotions.append(
                self._promotion_entry(trigger="promotion", mean_r=mean_r),
            )
            if self.verbose:
                print(
                    f"[curriculum] step={self.num_timesteps} "
                    f"eval_mean={mean_r:.2f} >= {threshold:.2f} "
                    f"-> promote to {self._describe_stage(new_stage)} "
                    f"(stage {self._idx + 1} / {len(self._stages)})",
                )
            self._log_current_stage()
            self.logger.record("curriculum/promotion_step", self.num_timesteps)
        return True

    # ------------------------------------------------------------------
    # Env-walking helpers
    # ------------------------------------------------------------------

    def _apply_stage(
        self, stage: CurriculumStage, *, sync_eval: bool | None = None,
    ) -> None:
        """Apply the stage's knobs (``skill`` and/or ``num_bots``) to all envs.

        ``sync_eval`` overrides ``self._sync_eval_env`` for this call — the
        initial application at ``_on_training_start`` always propagates to
        the eval env regardless of the instance flag.
        """
        propagate = self._sync_eval_env if sync_eval is None else sync_eval
        eval_env = getattr(self._eval_cb, "eval_env", None) if propagate else None
        if stage.skill is not None:
            _set_skill_on_vec_env(self.training_env, stage.skill)
            if eval_env is not None:
                _set_skill_on_vec_env(eval_env, stage.skill)
        if stage.num_bots is not None:
            _set_num_bots_on_vec_env(self.training_env, stage.num_bots)
            if eval_env is not None:
                _set_num_bots_on_vec_env(eval_env, stage.num_bots)

    def _promotion_entry(
        self, *, trigger: str, mean_r: float | None,
    ) -> dict[str, Any]:
        stage = self.current_stage
        return {
            "step": int(self.num_timesteps),
            "skill": stage.skill,
            "num_bots": stage.num_bots,
            "trigger": trigger,
            "eval_mean_reward": mean_r,
        }

    def _log_current_stage(self) -> None:
        stage = self.current_stage
        if stage.skill is not None:
            self.logger.record("curriculum/skill", stage.skill)
        if stage.num_bots is not None:
            self.logger.record("curriculum/num_bots", stage.num_bots)
        self.logger.record("curriculum/stage_index", self._idx)

    @staticmethod
    def _describe_stage(stage: CurriculumStage) -> str:
        parts: list[str] = []
        if stage.skill is not None:
            parts.append(f"skill={stage.skill}")
        if stage.num_bots is not None:
            parts.append(f"num_bots={stage.num_bots}")
        return " ".join(parts) or "<empty stage>"


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


def _set_num_bots_on_vec_env(vec_env: Any, num_bots: int) -> None:
    """Update each worker env's ``_num_bots`` so the next ``reset()`` spawns
    that many ZDoom AI bots.

    Unlike ``doom_skill`` (which DoomGame applies immediately), bot count
    only takes effect on the *next* episode because ``DoomEnv.reset``
    re-issues ``addbot`` commands after ``new_episode``. The active
    episode keeps the previous bot count.
    """
    for env in _iter_inner_envs(vec_env):
        base = env
        while hasattr(base, "env") and not hasattr(base, "_num_bots"):
            base = base.env
        if hasattr(base, "_num_bots"):
            base._num_bots = int(num_bots)
