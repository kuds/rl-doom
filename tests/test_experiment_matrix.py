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
