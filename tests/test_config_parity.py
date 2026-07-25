"""Guards on the shipped configs themselves.

``configs/matrix/deadly_corridor_curriculum.yaml`` runs each algorithm's
baseline against its curriculum sibling to answer "does the skill curriculum
help?". That only measures the curriculum if the two arms are otherwise
identical. They were not: both Dreamer curriculum configs gave the curriculum
arm 1.5-2x the baseline's budget, so its row of the headline table was
measuring compute.

Nothing in the repo parsed a shipped config, so the drift was invisible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# Resolved from this file, not cwd — an empty parametrize list collects zero
# tests and reports green.
CURRICULUM_CONFIGS = sorted(CONFIG_DIR.glob("*_curriculum.yaml"))
ALL_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
assert CURRICULUM_CONFIGS, f"no curriculum configs found under {CONFIG_DIR}"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _baseline_for(curriculum_path: Path) -> Path:
    return curriculum_path.with_name(
        curriculum_path.name.replace("_curriculum.yaml", ".yaml"),
    )


@pytest.mark.parametrize("path", CURRICULUM_CONFIGS, ids=lambda p: p.name)
def test_curriculum_matches_its_baseline_budget(path: Path) -> None:
    """Both arms of the comparison must train for the same number of steps."""
    baseline_path = _baseline_for(path)
    assert baseline_path.exists(), f"{path.name} has no baseline sibling"

    curriculum = _load(path)["training"]
    baseline = _load(baseline_path)["training"]

    assert curriculum["total_timesteps"] == baseline["total_timesteps"], (
        f"{path.name} trains for {curriculum['total_timesteps']:,} steps but "
        f"{baseline_path.name} trains for {baseline['total_timesteps']:,}. "
        f"The matrix compares these two directly, so unequal budgets make the "
        f"result measure compute rather than the curriculum."
    )
    assert curriculum["checkpoint_freq"] == baseline["checkpoint_freq"]


@pytest.mark.parametrize("path", CURRICULUM_CONFIGS, ids=lambda p: p.name)
def test_curriculum_matches_its_baseline_hyperparams(path: Path) -> None:
    """Only the curriculum block (and the knobs it owns) may differ."""
    curriculum = _load(path)
    baseline = _load(_baseline_for(path))

    assert curriculum["hyperparams"] == baseline["hyperparams"], (
        f"{path.name} and its baseline disagree on hyperparams"
    )
    assert curriculum["eval"] == baseline["eval"]
    assert curriculum["seed"] == baseline["seed"]

    # The env block may differ only on the knobs the curriculum ramps: those
    # are owned by the stages, so the baseline pins them and the curriculum
    # config leaves them out.
    ramped = {"doom_skill", "num_bots"}
    c_env = {k: v for k, v in curriculum["env"].items() if k not in ramped}
    b_env = {k: v for k, v in baseline["env"].items() if k not in ramped}
    assert c_env == b_env, f"{path.name} env differs from its baseline beyond {ramped}"


@pytest.mark.parametrize("path", CURRICULUM_CONFIGS, ids=lambda p: p.name)
def test_curriculum_stages_ramp_something(path: Path) -> None:
    """A stage list that sets neither knob would be a no-op curriculum."""
    stages = _load(path)["curriculum"]["stages"]
    assert len(stages) >= 2, f"{path.name} has fewer than two stages"
    for stage in stages:
        assert "skill" in stage or "num_bots" in stage, (
            f"{path.name} has a stage that ramps neither skill nor num_bots: {stage}"
        )


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_every_config_declares_its_identity(path: Path) -> None:
    """Catch a config that cannot be dispatched at all."""
    from rl_doom.sb3_utils import resolve_algo_class

    cfg = _load(path)
    for key in ("scenario", "algorithm", "seed", "env", "hyperparams", "training", "eval"):
        assert key in cfg, f"{path.name} is missing the '{key}' block"

    algo = cfg["algorithm"]
    if algo != "dreamer":  # dreamer has its own driver, not an SB3 class
        resolve_algo_class(algo)


# ---------------------------------------------------------------------------
# Stage-0 env settings
#
# train_sb3 folds the curriculum's first stage into the env settings before
# building the envs. It used to do so unconditionally, which broke the
# bot-ramping deathmatch curricula: their stages set only num_bots, so
# `doom_skill = stages[0].skill` wrote None over the config's `doom_skill: 3`
# and the arm silently trained at the scenario cfg's default difficulty while
# its baseline sibling trained at skill 3.
# ---------------------------------------------------------------------------


def test_stage0_skill_ramp_overrides_skill_and_leaves_bots() -> None:
    from rl_doom.curriculum import CurriculumStage
    from rl_doom.sb3_utils import apply_stage0_env_settings

    stage0 = CurriculumStage(skill=1, num_bots=None, promote_at=1500.0)
    skill, bots = apply_stage0_env_settings(stage0, doom_skill=3, num_bots=4, verbose=False)
    assert skill == 1
    assert bots == 4


def test_stage0_bot_ramp_preserves_configured_skill() -> None:
    """A bot-only curriculum must not blank out doom_skill."""
    from rl_doom.curriculum import CurriculumStage
    from rl_doom.sb3_utils import apply_stage0_env_settings

    stage0 = CurriculumStage(skill=None, num_bots=2, promote_at=3.0)
    skill, bots = apply_stage0_env_settings(stage0, doom_skill=3, num_bots=0, verbose=False)
    assert skill == 3, "bot-only curriculum discarded the configured doom_skill"
    assert bots == 2, "stage-0 num_bots did not reach env construction"


def test_stage0_ramping_both_overrides_both() -> None:
    from rl_doom.curriculum import CurriculumStage
    from rl_doom.sb3_utils import apply_stage0_env_settings

    stage0 = CurriculumStage(skill=2, num_bots=6, promote_at=10.0)
    skill, bots = apply_stage0_env_settings(stage0, doom_skill=5, num_bots=1, verbose=False)
    assert (skill, bots) == (2, 6)


@pytest.mark.parametrize("path", CURRICULUM_CONFIGS, ids=lambda p: p.name)
def test_shipped_curricula_keep_their_baseline_env_settings(path: Path) -> None:
    """End-to-end over the real configs: no knob is silently lost.

    Whatever the baseline pins, the curriculum arm must end up with — either
    because its stages ramp it, or because the config's value survives.
    """
    from rl_doom.curriculum import parse_curriculum_config
    from rl_doom.sb3_utils import apply_stage0_env_settings

    curriculum_cfg = _load(path)
    baseline_env = _load(_baseline_for(path))["env"]

    stages = parse_curriculum_config(curriculum_cfg["curriculum"])
    assert stages, f"{path.name} parsed to an empty stage list"
    skill, bots = apply_stage0_env_settings(
        stages[0],
        doom_skill=curriculum_cfg["env"].get("doom_skill"),
        num_bots=int(curriculum_cfg["env"].get("num_bots", 0)),
        verbose=False,
    )

    if stages[0].skill is None:
        assert skill == baseline_env.get("doom_skill"), (
            f"{path.name} ramps bots, not skill, so it must keep the baseline's "
            f"doom_skill={baseline_env.get('doom_skill')} — got {skill}"
        )
    if stages[0].num_bots is None:
        assert bots == int(baseline_env.get("num_bots", 0))
