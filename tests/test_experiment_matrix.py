"""Unit tests for ``scripts/run_experiment_matrix.py``.

Only exercises the pure-Python plumbing (config merging and variant
expansion). The actual ``train_sb3`` call is not invoked — it's covered
by the existing SB3 training tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


def _load_matrix_module():
    """Import ``scripts/run_experiment_matrix.py`` as a module.

    The script lives under ``scripts/`` which isn't a Python package, so
    we load it by path. The script's own ``sys.path`` mutation to import
    ``rl_doom`` is a no-op here because the test runner already has the
    src tree on the path via the editable install (or pyproject config).
    """
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("_matrix_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_matrix_under_test"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_leaves_base_untouched() -> None:
    matrix = _load_matrix_module()
    base: dict = {"hyperparams": {"lr": 1e-4, "gamma": 0.99}, "env": {"n_envs": 1}}
    overrides: dict = {"hyperparams": {"lr": 5e-4}}
    merged = matrix._deep_merge(base, overrides)

    # Returned dict has the override applied.
    assert merged["hyperparams"]["lr"] == 5e-4
    # Non-overridden keys survive.
    assert merged["hyperparams"]["gamma"] == 0.99
    # Original base dict is untouched (no aliasing).
    assert base["hyperparams"]["lr"] == 1e-4


def test_deep_merge_adds_new_keys() -> None:
    matrix = _load_matrix_module()
    merged = matrix._deep_merge(
        {"a": 1},
        {"b": {"c": 2}},
    )
    assert merged == {"a": 1, "b": {"c": 2}}


def test_deep_merge_replaces_non_dict_value() -> None:
    matrix = _load_matrix_module()
    merged = matrix._deep_merge({"x": [1, 2, 3]}, {"x": [9]})
    assert merged == {"x": [9]}


# ---------------------------------------------------------------------------
# _expand_variants
# ---------------------------------------------------------------------------


def _write_base_config(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_expand_variants_cross_with_seeds(tmp_path: Path) -> None:
    matrix = _load_matrix_module()
    base_path = _write_base_config(
        tmp_path,
        "base",
        {
            "scenario": "deadly_corridor",
            "algorithm": "ppo",
            "seed": 0,
            "hyperparams": {"lr": 1e-4},
            "training": {"total_timesteps": 1000},
        },
    )
    cfg = {
        "name": "m",
        "seeds": [42, 123],
        "variants": [
            {"name": "baseline", "base_config": str(base_path)},
            {
                "name": "big_lr",
                "base_config": str(base_path),
                "overrides": {"hyperparams": {"lr": 5e-4}},
            },
        ],
    }
    runs = matrix._expand_variants(cfg)
    # 2 variants × 2 seeds = 4 runs.
    assert len(runs) == 4

    # Variant + seed pairs are all represented.
    pairs = {(r["_variant_name"], r["seed"]) for r in runs}
    assert pairs == {
        ("baseline", 42),
        ("baseline", 123),
        ("big_lr", 42),
        ("big_lr", 123),
    }

    # Override reached the merged dict; baseline is untouched.
    big_lr = next(r for r in runs if r["_variant_name"] == "big_lr")
    baseline = next(r for r in runs if r["_variant_name"] == "baseline")
    assert big_lr["hyperparams"]["lr"] == 5e-4
    assert baseline["hyperparams"]["lr"] == 1e-4


def test_expand_variants_rejects_empty_variants() -> None:
    matrix = _load_matrix_module()
    with pytest.raises(ValueError, match="non-empty 'variants'"):
        matrix._expand_variants({"seeds": [1], "variants": []})


def test_expand_variants_requires_base_config(tmp_path: Path) -> None:
    matrix = _load_matrix_module()
    with pytest.raises(ValueError, match="missing 'base_config'"):
        matrix._expand_variants(
            {
                "seeds": [1],
                "variants": [{"name": "nope"}],
            },
        )


# ---------------------------------------------------------------------------
# _resolve_dreamer_port_path — env var + fallback chain
# ---------------------------------------------------------------------------


def test_resolve_dreamer_port_path_uses_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DREAMERV3_TORCH_PATH`` should win over the ``/content`` and ``~`` fallbacks."""
    matrix = _load_matrix_module()
    explicit = tmp_path / "custom_port"
    explicit.mkdir()
    monkeypatch.setenv("DREAMERV3_TORCH_PATH", str(explicit))
    assert matrix._resolve_dreamer_port_path() == explicit


def test_resolve_dreamer_port_path_falls_back_when_nothing_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When no candidate exists, return the last candidate so train_dreamer can
    surface a friendly clone-this-repo error rather than a confusing path error."""
    matrix = _load_matrix_module()
    # Point env var at a non-existent dir; ``/content`` and ``~/dreamerv3-torch``
    # are also expected not to exist on a typical CI runner.
    monkeypatch.setenv("DREAMERV3_TORCH_PATH", str(tmp_path / "nope"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    result = matrix._resolve_dreamer_port_path()
    # Last-resort fallback is ``Path.home() / "dreamerv3-torch"``.
    assert result == tmp_path / "dreamerv3-torch"


# ---------------------------------------------------------------------------
# Failure isolation, incremental summary, and --resume
#
# The loop used to be `rows.append(_run_one(spec))` with no try, and the
# summary CSV was written once after the last cell. A crash in cell 3 of 24
# skipped cells 4-24 *and* discarded the rows for cells 1 and 2 — hours to days
# of compute producing no aggregate output at all.
# ---------------------------------------------------------------------------


@pytest.fixture()
def matrix_env(tmp_path, monkeypatch):
    """Matrix module with training stubbed and artifacts redirected to tmp."""
    module = _load_matrix_module()
    monkeypatch.setattr(module, "TRAINING_JOBS", tmp_path / "training_jobs")

    matrix_cfg = {
        "name": "testmatrix",
        "seeds": [1, 2],
        "variants": [
            {"name": "alpha", "base_config": "ppo_basic"},
            {"name": "beta", "base_config": "ppo_basic"},
        ],
    }
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix_cfg))
    return module, matrix_path


def _stub_run_one(module, monkeypatch, fail_on=()):
    """Replace _run_one with a stub that records calls and fails on demand."""
    calls: list[tuple[str, int]] = []

    def _fake_run_one(spec, *, matrix_name, total_timesteps_override, device):
        cell = (spec["_variant_name"], int(spec["seed"]))
        calls.append(cell)
        if cell in fail_on:
            raise RuntimeError(f"simulated failure in {cell[0]} seed={cell[1]}")
        return {
            "matrix": matrix_name,
            "variant": cell[0],
            "scenario": spec["scenario"],
            "algo": spec["algorithm"],
            "seed": cell[1],
            "mean_eval_reward": 1.0,
            "run_dir": f"/fake/{cell[0]}_{cell[1]}",
        }

    monkeypatch.setattr(module, "_run_one", _fake_run_one)
    return calls


def _summary_rows(module, name="testmatrix"):
    import csv

    path = module.matrix_summary_path(name)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_one_failed_cell_does_not_abort_the_matrix(matrix_env, monkeypatch) -> None:
    module, matrix_path = matrix_env
    calls = _stub_run_one(module, monkeypatch, fail_on={("alpha", 2)})

    exit_code = module.main(["--matrix", str(matrix_path)])

    assert len(calls) == 4, "a failed cell stopped the remaining runs"
    assert exit_code == 1, "matrix with a failure should exit non-zero"


def test_failed_cell_is_recorded_with_its_error(matrix_env, monkeypatch) -> None:
    module, matrix_path = matrix_env
    _stub_run_one(module, monkeypatch, fail_on={("alpha", 2)})
    module.main(["--matrix", str(matrix_path)])

    rows = _summary_rows(module)
    assert len(rows) == 4, "every cell should appear in the summary"
    failed = [r for r in rows if r["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["variant"] == "alpha" and failed[0]["seed"] == "2"
    assert "simulated failure" in failed[0]["error"]


def test_summary_survives_the_process_dying_midway(matrix_env, monkeypatch) -> None:
    """Rows are appended as cells finish, not batched to the end."""
    module, matrix_path = matrix_env

    def _die_on_third(spec, *, matrix_name, total_timesteps_override, device):
        cell = (spec["_variant_name"], int(spec["seed"]))
        if cell == ("beta", 1):
            raise KeyboardInterrupt  # not caught by the per-cell except
        return {
            "matrix": matrix_name, "variant": cell[0], "scenario": spec["scenario"],
            "algo": spec["algorithm"], "seed": cell[1], "mean_eval_reward": 1.0,
            "run_dir": "/fake",
        }

    monkeypatch.setattr(module, "_run_one", _die_on_third)
    with pytest.raises(KeyboardInterrupt):
        module.main(["--matrix", str(matrix_path)])

    rows = _summary_rows(module)
    assert len(rows) == 2, (
        "cells completed before the kill were lost; the CSV must be appended "
        "to as it goes"
    )
    assert all(r["status"] == "completed" for r in rows)


def test_resume_skips_completed_cells_and_retries_failures(
    matrix_env, monkeypatch,
) -> None:
    module, matrix_path = matrix_env

    _stub_run_one(module, monkeypatch, fail_on={("alpha", 2)})
    module.main(["--matrix", str(matrix_path)])

    second_calls = _stub_run_one(module, monkeypatch, fail_on=set())
    exit_code = module.main(["--matrix", str(matrix_path), "--resume"])

    assert second_calls == [("alpha", 2)], (
        f"--resume should retry only the failed cell, ran {second_calls}"
    )
    assert exit_code == 0
    statuses = {(r["variant"], r["seed"]): r["status"] for r in _summary_rows(module)}
    assert statuses[("alpha", "2")] == "completed"


def test_fail_fast_stops_at_the_first_failure(matrix_env, monkeypatch) -> None:
    module, matrix_path = matrix_env
    calls = _stub_run_one(module, monkeypatch, fail_on={("alpha", 2)})

    exit_code = module.main(["--matrix", str(matrix_path), "--fail-fast"])

    assert exit_code == 1
    assert calls == [("alpha", 1), ("alpha", 2)], "should stop after the failure"
    assert len(_summary_rows(module)) == 2


def test_rerun_without_resume_starts_the_summary_over(matrix_env, monkeypatch) -> None:
    """Otherwise a second run would stack rows on top of the first."""
    module, matrix_path = matrix_env
    _stub_run_one(module, monkeypatch)
    module.main(["--matrix", str(matrix_path)])
    assert len(_summary_rows(module)) == 4

    _stub_run_one(module, monkeypatch)
    module.main(["--matrix", str(matrix_path)])
    assert len(_summary_rows(module)) == 4
