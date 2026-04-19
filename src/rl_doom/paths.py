"""Experiment path layout helpers.

Centralises the directory conventions used by the training and analysis
notebooks so that every notebook writes to the same predictable structure:

    <repo>/
    ├── training_jobs/<env>/<algo>/
    │   ├── runs/<run_id>/{checkpoints,metrics,tensorboard,figures,media}/
    │   ├── sweeps/<sweep_name>/<variant>/{checkpoints,metrics,tensorboard,figures}/
    │   └── latest -> runs/<most_recent_run_id>
    └── analysis/
        ├── per_experiment/<env>/<algo>/{runs_aggregated,sweeps/<sweep_name>}/
        ├── cross_experiment/
        └── saliency/

When running on Colab with Google Drive mounted, ``training_jobs`` and
``analysis`` are expected to be symlinks into Drive; the helpers here only
touch paths relative to the repo root and do not care whether a given
directory is a real directory or a symlink.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root = two parents above this file (src/rl_doom/paths.py -> rl_doom/).
REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_JOBS = REPO_ROOT / "training_jobs"
ANALYSIS = REPO_ROOT / "analysis"

_RUN_SUBDIRS = ("checkpoints", "metrics", "tensorboard", "figures", "media")
_SWEEP_VARIANT_SUBDIRS = ("checkpoints", "metrics", "tensorboard", "figures")


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def experiment_dir(env: str, algo: str) -> Path:
    """Return ``training_jobs/<env>/<algo>/``, creating it if missing."""
    d = TRAINING_JOBS / env / algo
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_run_dir(
    env: str,
    algo: str,
    seed: int,
    tag: str | None = None,
) -> Path:
    """Create a fresh run directory and return its path.

    The run ID is ``<UTC timestamp>_seed<N>[_<tag>]`` so runs sort
    chronologically and never collide.
    """
    run_id = f"{_utc_stamp()}_seed{seed}"
    if tag:
        run_id = f"{run_id}_{tag}"
    run_dir = experiment_dir(env, algo) / "runs" / run_id
    for sub in _RUN_SUBDIRS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def list_runs(env: str, algo: str) -> list[Path]:
    """Return all run directories for an experiment, sorted chronologically."""
    runs_dir = experiment_dir(env, algo) / "runs"
    if not runs_dir.exists():
        return []
    return sorted(p for p in runs_dir.iterdir() if p.is_dir())


def latest_run(env: str, algo: str) -> Path:
    """Return the most recent run directory.

    Prefers the ``latest`` pointer (symlink or plain-text fallback file) if
    present; otherwise falls back to the last entry in sorted run IDs.
    Raises ``FileNotFoundError`` if no runs exist.
    """
    parent = experiment_dir(env, algo)
    latest = parent / "latest"
    if latest.is_symlink():
        resolved = latest.resolve()
        if resolved.exists():
            return resolved
    elif latest.is_file():
        # Plain-text pointer file (used when the filesystem does not
        # support symlinks, e.g. Google Drive mounted on Colab).
        target_rel = latest.read_text().strip()
        if target_rel:
            resolved = (parent / target_rel).resolve()
            if resolved.exists():
                return resolved
    runs = list_runs(env, algo)
    if not runs:
        raise FileNotFoundError(f"No runs found for {env}/{algo}")
    return runs[-1]


def update_latest_symlink(env: str, algo: str, run_dir: Path) -> None:
    """Atomically point ``training_jobs/<env>/<algo>/latest`` at ``run_dir``.

    On filesystems that support symlinks this creates ``latest`` as a
    symlink. On filesystems that do not (e.g. Google Drive's FUSE mount on
    Colab, which fails with ``OSError`` errno 95 "Operation not supported"),
    it falls back to writing a plain-text pointer file containing the
    relative run path. ``latest_run`` understands both forms.
    """
    parent = experiment_dir(env, algo)
    latest = parent / "latest"
    # Use a path relative to the symlink's parent so the link stays valid
    # even if the whole tree is moved (e.g. Drive remount).
    target = Path("runs") / run_dir.name
    tmp = parent / ".latest.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    try:
        tmp.symlink_to(target)
    except OSError:
        # Filesystem refuses symlinks; write a text pointer instead. The
        # failed symlink_to call does not create ``tmp``, so it is safe to
        # write to the same path directly.
        tmp.write_text(str(target))
    os.replace(tmp, latest)


# ---------------------------------------------------------------------------
# Run config / status
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def write_config(
    run_dir: Path,
    *,
    env: str,
    algo: str,
    seed: int,
    hyperparams: dict[str, Any],
    env_settings: dict[str, Any] | None = None,
    policy_kwargs: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Write ``config.json`` at the root of a run directory.

    ``env_settings`` captures the ViZDoom environment configuration
    (``resize_shape``, ``frame_skip``, ``num_stack``, ``n_envs``,
    ``doom_skill``, ``num_bots``, …) so the exact wrapper stack used by a
    run is reproducible from ``config.json`` alone.

    ``policy_kwargs`` captures the network architecture (``features_dim``,
    ``net_arch``, any additional SB3 ``policy_kwargs`` overrides) so the
    trained weights can be paired with the exact network definition that
    produced them.
    """
    cfg = {
        "env": env,
        "algo": algo,
        "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "hyperparams": hyperparams,
        "env_settings": env_settings or {},
        "policy_kwargs": policy_kwargs or {},
        "status": "running",
        **extra,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))


def mark_run_status(run_dir: Path, status: str, **extra: Any) -> None:
    """Update ``status`` and append ``finished_at`` / extra fields to config.json."""
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg["status"] = status
    cfg["finished_at"] = datetime.now(timezone.utc).isoformat()
    for k, v in extra.items():
        cfg[k] = v
    cfg_path.write_text(json.dumps(cfg, indent=2, default=str))


def load_config(run_dir: Path) -> dict[str, Any]:
    """Load ``config.json`` from a run directory."""
    return json.loads((run_dir / "config.json").read_text())


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def new_sweep_dir(env: str, algo: str, sweep_name: str) -> Path:
    """Create a sweep directory and return its path."""
    sweep_dir = experiment_dir(env, algo) / "sweeps" / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    return sweep_dir


def new_sweep_variant_dir(sweep_dir: Path, variant_name: str) -> Path:
    """Create one variant directory inside a sweep."""
    variant_dir = sweep_dir / variant_name
    for sub in _SWEEP_VARIANT_SUBDIRS:
        (variant_dir / sub).mkdir(parents=True, exist_ok=True)
    return variant_dir


# ---------------------------------------------------------------------------
# Analysis directories (notebook 04 outputs)
# ---------------------------------------------------------------------------


def analysis_root() -> Path:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    return ANALYSIS


def per_experiment_analysis_dir(env: str, algo: str) -> Path:
    d = ANALYSIS / "per_experiment" / env / algo
    d.mkdir(parents=True, exist_ok=True)
    return d


def cross_experiment_analysis_dir() -> Path:
    d = ANALYSIS / "cross_experiment"
    for sub in ("tables", "figures"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def saliency_analysis_dir() -> Path:
    d = ANALYSIS / "saliency"
    d.mkdir(parents=True, exist_ok=True)
    return d
