"""Unit tests for :mod:`rl_doom.paths`.

Focused on the metadata that downstream tooling (notebooks,
``generate_summary``, cross-run analyses) depends on being present in
``config.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from rl_doom import __version__ as PACKAGE_VERSION
from rl_doom.paths import write_config


def test_write_config_records_package_version(tmp_path: Path) -> None:
    """``config.json`` must carry the codebase version that produced the run."""
    write_config(
        tmp_path,
        env="basic",
        algo="ppo",
        seed=0,
        hyperparams={"lr": 1e-4},
    )
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["version"] == PACKAGE_VERSION
