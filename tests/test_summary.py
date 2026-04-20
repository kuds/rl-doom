"""Unit tests for :mod:`rl_doom.summary`.

Covers the ``Curriculum`` section rendering added to
:func:`rl_doom.summary.generate_run_summary` so a regression in the
per-run ``stage_summary.txt`` output is caught at CI time rather than
in a notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_doom.summary import generate_run_summary


def _minimal_run_dir(
    tmp_path: Path,
    *,
    curriculum: dict | None = None,
    terminations: dict | None = None,
) -> Path:
    """Write the files ``generate_run_summary`` reads.

    The function only needs ``config.json`` + ``metrics/training.npz`` to
    produce a non-empty summary; ``curriculum.json`` and
    ``termination_counts.json`` are optional sidecars.
    """
    run_dir = tmp_path / "run"
    (run_dir / "metrics").mkdir(parents=True)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "env": "deadly_corridor",
                "algo": "ppo",
                "seed": 42,
                "hyperparams": {
                    "lr": 2.5e-4,
                    "total_timesteps": 3_000_000,
                    "n_envs": 8,
                },
            },
        ),
    )
    np.savez(
        run_dir / "metrics" / "training.npz",
        eval_rewards=np.array([[1_000_000, 1730.46, 845.01, 50.0, 19.2]]),
        episode_rewards=np.array([100.0] * 20),
        wall_time_seconds=8592.0,
        fps=233,
    )
    if curriculum is not None:
        (run_dir / "metrics" / "curriculum.json").write_text(
            json.dumps(curriculum),
        )
    if terminations is not None:
        (run_dir / "metrics" / "termination_counts.json").write_text(
            json.dumps(terminations),
        )
    return run_dir


def test_summary_without_curriculum_has_no_curriculum_section(
    tmp_path: Path,
) -> None:
    """Baseline (non-curriculum) runs must not grow a curriculum section."""
    run_dir = _minimal_run_dir(tmp_path)
    out = generate_run_summary(run_dir)
    assert "Curriculum" not in out
    assert "Final skill" not in out


def test_summary_renders_curriculum_final_skill_and_promotions(
    tmp_path: Path,
) -> None:
    """The curriculum section must surface final skill + each promotion."""
    curriculum = {
        "stages": [
            {"skill": 1, "num_bots": None, "promote_at": 1500.0},
            {"skill": 2, "num_bots": None, "promote_at": 1500.0},
            {"skill": 3, "num_bots": None, "promote_at": None},
        ],
        "promotions": [
            {
                "step": 0,
                "skill": 1,
                "num_bots": None,
                "trigger": "initial",
                "eval_mean_reward": None,
            },
            {
                "step": 500_000,
                "skill": 2,
                "num_bots": None,
                "trigger": "promotion",
                "eval_mean_reward": 1523.4,
            },
            {
                "step": 1_250_000,
                "skill": 3,
                "num_bots": None,
                "trigger": "promotion",
                "eval_mean_reward": 1612.9,
            },
        ],
        "final_stage_index": 2,
        "final_skill": 3,
        "final_num_bots": None,
    }
    out = generate_run_summary(_minimal_run_dir(tmp_path, curriculum=curriculum))
    assert "Curriculum" in out
    # Final skill includes the human-readable stage position.
    assert "Final skill:" in out
    assert "3" in out
    assert "stage 3 / 3" in out
    # Promotion table lists both non-initial promotions with step + knob +
    # eval mean. Step number is comma-formatted by ``_fmt_number``.
    assert "500,000" in out
    assert "1,250,000" in out
    assert "-> skill 2" in out
    assert "-> skill 3" in out
    assert "1523.40" in out
    assert "1612.90" in out
    # The "initial" rung is not counted as a promotion.
    assert "Promotions:     2" in out


def test_summary_handles_curriculum_with_only_initial_stage(
    tmp_path: Path,
) -> None:
    """An aborted-early run (never promoted) still renders the final skill."""
    curriculum = {
        "stages": [
            {"skill": 1, "promote_at": 1500.0},
            {"skill": 2, "promote_at": None},
        ],
        "promotions": [
            {"step": 0, "skill": 1, "trigger": "initial", "eval_mean_reward": None},
        ],
        "final_stage_index": 0,
        "final_skill": 1,
    }
    out = generate_run_summary(_minimal_run_dir(tmp_path, curriculum=curriculum))
    assert "Curriculum" in out
    assert "Final skill:" in out
    # Stage index is 0 -> "stage 1 / 2". Promotion list should be omitted
    # entirely since no promotions (only the initial entry) occurred.
    assert "stage 1 / 2" in out
    assert "Promotions:" not in out
