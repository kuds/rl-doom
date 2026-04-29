"""Curriculum learning for ViZDoom scenarios.

Provides three layers, smallest first:

* :class:`CurriculumStage` — one rung of the schedule (skill / num_bots /
  promote_at threshold).
* :class:`CurriculumController` — framework-agnostic state machine that
  owns the stage list, the "evals since last promotion" counter, and the
  promotion timeline. Used directly by :func:`rl_doom.agents.dreamer.train_dreamer`.
* :class:`SkillCurriculumCallback` — Stable-Baselines3 callback that wraps a
  ``CurriculumController`` and plumbs its decisions into SB3's
  ``EvalCallback`` lifecycle (used by DQN / PPO / Recurrent PPO).

Two difficulty knobs are supported:

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
* SB3 path: relies on the public ``EvalCallback`` attributes
  ``evaluations_timesteps`` and ``last_mean_reward``, so the callback
  must be registered **after** the ``EvalCallback`` in the callback
  list (SB3 fires callbacks in registration order).
* Both paths propagate changes to the eval env in lock-step with the
  training env (controlled by ``sync_eval_env``) so the promotion
  threshold is measured on the same difficulty the agent is training on.
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


class CurriculumController:
    """Framework-agnostic curriculum state machine.

    Owns the stage list, the current rung, the per-rung eval counter, and
    the promotion timeline. Knows nothing about envs or training frameworks
    — callers wire env mutations themselves (via :func:`apply_stage_to_doom_env`
    for single envs or the SB3 vec-env walkers below).

    Two-step usage:

    1. Call :meth:`record_initial` once before training to take the starting
       stage and append the ``trigger="initial"`` entry to ``promotions``.
    2. Call :meth:`maybe_promote` after each completed eval pass; if the
       returned stage is non-None, apply it to your envs.

    The state machine is entirely deterministic: each ``maybe_promote`` call
    consumes one eval, increments the rung counter, and may advance the
    stage index by at most one. ``min_evals_between_promotions`` blocks
    rapid double-promotion on a single lucky rollout.
    """

    def __init__(
        self,
        stages: list[CurriculumStage],
        *,
        min_evals_between_promotions: int = 1,
    ) -> None:
        if not stages:
            raise ValueError("stages must be non-empty")
        self._stages = stages
        self._idx = 0
        self._min_gap = max(1, int(min_evals_between_promotions))
        self._evals_since_promotion = 0
        # Public, append-only — written for the JSON timeline that
        # ``rl_doom.summary`` and the matrix CSV consume.
        self.promotions: list[dict[str, Any]] = []

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
    def num_stages(self) -> int:
        return len(self._stages)

    @property
    def is_terminal(self) -> bool:
        return self._idx >= len(self._stages) - 1

    @property
    def evals_since_promotion(self) -> int:
        return self._evals_since_promotion

    def record_initial(self) -> CurriculumStage:
        """Append the initial entry to ``promotions`` and return the starting stage.

        Callers should immediately apply the returned stage to both the
        training and eval envs (initial application always propagates to
        the eval env regardless of ``sync_eval_env``).
        """
        self.promotions.append(self._promotion_entry(trigger="initial", mean_r=None, step=0))
        return self.current_stage

    def maybe_promote(
        self, *, current_step: int, eval_mean_reward: float,
    ) -> CurriculumStage | None:
        """Process a fresh eval result.

        Returns the new stage if a promotion fired, otherwise ``None``.
        Callers are responsible for applying the returned stage to envs
        (and for deciding whether to sync the eval env in non-initial
        promotions, via their own ``sync_eval_env`` semantics).
        """
        self._evals_since_promotion += 1
        if self.is_terminal:
            return None
        threshold = self._stages[self._idx].promote_at
        # ``parse_curriculum_config`` enforces that non-terminal stages
        # set a threshold, but guard anyway so callers passing raw
        # CurriculumStage lists don't trip an assertion error.
        if threshold is None:
            return None
        if eval_mean_reward >= threshold and self._evals_since_promotion >= self._min_gap:
            self._idx += 1
            self._evals_since_promotion = 0
            self.promotions.append(
                self._promotion_entry(
                    trigger="promotion",
                    mean_r=float(eval_mean_reward),
                    step=int(current_step),
                ),
            )
            return self.current_stage
        return None

    def describe_stage(self, stage: CurriculumStage | None = None) -> str:
        """Format ``skill=X num_bots=Y`` for log lines."""
        stage = stage if stage is not None else self.current_stage
        parts: list[str] = []
        if stage.skill is not None:
            parts.append(f"skill={stage.skill}")
        if stage.num_bots is not None:
            parts.append(f"num_bots={stage.num_bots}")
        return " ".join(parts) or "<empty stage>"

    def _promotion_entry(
        self, *, trigger: str, mean_r: float | None, step: int,
    ) -> dict[str, Any]:
        stage = self.current_stage
        return {
            "step": int(step),
            "skill": stage.skill,
            "num_bots": stage.num_bots,
            "trigger": trigger,
            "eval_mean_reward": mean_r,
        }


def apply_stage_to_doom_env(doom_env: Any, stage: CurriculumStage) -> None:
    """Apply ``stage``'s knobs to a single DoomEnv (or a Gym wrapper around one).

    Walks ``.env`` until it hits something exposing ``game`` (skill path)
    or ``_num_bots`` (bot-count path). No-ops on stage knobs the env
    doesn't support — e.g. applying a ``num_bots`` stage to a non-deathmatch
    DoomEnv just leaves the field unset.

    Used by :func:`rl_doom.agents.dreamer.train_dreamer` to update the wrapped
    DreamerDoomEnv (and its eval twin) without going through SB3's vec-env
    machinery.
    """
    base = doom_env
    # Walk Gym/Gymnasium wrapper chain; stop when we find either knob's
    # backing attribute (skill needs ``game``, num_bots needs ``_num_bots``).
    while hasattr(base, "env") and not hasattr(base, "game") and not hasattr(base, "_num_bots"):
        base = base.env
    if stage.skill is not None:
        # Skill might live one wrapper deeper than ``_num_bots``; do a
        # short secondary descent if the current ``base`` lacks ``game``.
        skill_base = base
        while hasattr(skill_base, "env") and not hasattr(skill_base, "game"):
            skill_base = skill_base.env
        game = getattr(skill_base, "game", None)
        if game is not None:
            set_skill = getattr(game, "set_doom_skill", None)
            if set_skill is not None:
                set_skill(stage.skill)
            if hasattr(skill_base, "_doom_skill"):
                skill_base._doom_skill = stage.skill
    if stage.num_bots is not None:
        bots_base = base
        while hasattr(bots_base, "env") and not hasattr(bots_base, "_num_bots"):
            bots_base = bots_base.env
        if hasattr(bots_base, "_num_bots"):
            bots_base._num_bots = int(stage.num_bots)


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
        self._eval_cb = eval_cb
        self._state = CurriculumController(
            stages,
            min_evals_between_promotions=min_evals_between_promotions,
        )
        self._sync_eval_env = sync_eval_env
        self._last_seen_eval_ts: int = -1
        # The pre-controller field name. Tests and downstream code may
        # poke at ``promotions`` directly — forward to the controller's
        # list (same object) so legacy access stays valid.
        # Populated as soon as ``_on_training_start`` runs.

    # ------------------------------------------------------------------
    # Public introspection helpers — forwarded to the controller so
    # external code that imported these stays binary-compatible across
    # the controller refactor.
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> CurriculumStage:
        return self._state.current_stage

    @property
    def current_skill(self) -> int | None:
        return self._state.current_skill

    @property
    def current_num_bots(self) -> int | None:
        return self._state.current_num_bots

    @property
    def current_stage_index(self) -> int:
        return self._state.current_stage_index

    @property
    def is_terminal(self) -> bool:
        return self._state.is_terminal

    @property
    def promotions(self) -> list[dict[str, Any]]:
        return self._state.promotions

    @property
    def _evals_since_promotion(self) -> int:
        # Test-only accessor — tests poke at this internal counter to
        # verify min_gap behaviour. Forwards to the controller.
        return self._state.evals_since_promotion

    @property
    def _stages(self) -> list[CurriculumStage]:
        # Legacy accessor used by older downstream code.
        return self._state._stages

    @property
    def _idx(self) -> int:
        # Legacy accessor.
        return self._state.current_stage_index

    # ------------------------------------------------------------------
    # SB3 callback protocol
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        initial_stage = self._state.record_initial()
        # Always seed the eval env with the initial stage even when
        # ``sync_eval_env=False``; that flag only gates future promotions.
        self._apply_stage(initial_stage, sync_eval=True)
        if self.verbose:
            print(
                f"[curriculum] start {self._state.describe_stage()} "
                f"(stage 1 / {self._state.num_stages})",
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
        mean_r = float(self._eval_cb.last_mean_reward)

        new_stage = self._state.maybe_promote(
            current_step=int(self.num_timesteps),
            eval_mean_reward=mean_r,
        )
        # Log current stage on every fresh eval (whether or not we promoted)
        # so a flat training run still has a non-empty curriculum/* series.
        self._log_current_stage()
        if new_stage is not None:
            self._apply_stage(new_stage)
            if self.verbose:
                threshold = self._state._stages[
                    self._state.current_stage_index - 1
                ].promote_at
                print(
                    f"[curriculum] step={self.num_timesteps} "
                    f"eval_mean={mean_r:.2f} >= {threshold:.2f} "
                    f"-> promote to {self._state.describe_stage(new_stage)} "
                    f"(stage {self._state.current_stage_index + 1} / {self._state.num_stages})",
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

    def _log_current_stage(self) -> None:
        stage = self._state.current_stage
        if stage.skill is not None:
            self.logger.record("curriculum/skill", stage.skill)
        if stage.num_bots is not None:
            self.logger.record("curriculum/num_bots", stage.num_bots)
        self.logger.record("curriculum/stage_index", self._state.current_stage_index)


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
