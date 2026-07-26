"""Shared utilities for training artifact generation.

Centralizes plotting, the training timer, checkpoint saving, and Google Drive
persistence so notebooks stay concise and consistent. Run-level metadata
(seed, git SHA, hyperparams, GPU info) is owned by ``rl_doom.paths.write_config``
and ``rl_doom.sb3_utils.gpu_info`` \u2014 this module does not duplicate it.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_doom.paths import REPO_ROOT

# ---------------------------------------------------------------------------
# Google Drive persistence
# ---------------------------------------------------------------------------

def in_colab() -> bool:
    """True when running inside a Google Colab runtime.

    Checked by importing ``google.colab`` rather than probing for
    ``/content``, which also exists on other hosted runtimes.
    """
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


# Artifact trees that ``rl_doom.paths`` writes into, and therefore the ones
# worth persisting to Drive across Colab sessions.
DRIVE_SUBDIRS: tuple[str, ...] = ("training_jobs", "analysis")


def setup_google_drive(
    drive_root: str = "/content/drive/MyDrive/rl-doom",
    subdirs: Sequence[str] | None = None,
    repo_root: str | Path | None = None,
) -> str | None:
    """Mount Google Drive and symlink the artifact trees into it.

    Call at the top of a notebook. **Off Colab this really is a no-op**: it
    returns ``None`` without touching the filesystem. (The previous version
    documented that behaviour but did not implement it — the bare
    ``from google.colab import drive`` raised ``ModuleNotFoundError``.)

    Symlinks ``<repo>/training_jobs`` and ``<repo>/analysis`` at
    ``<drive_root>/...`` so runs survive a runtime being recycled. Those are
    the trees ``rl_doom.paths`` actually writes; the old default named a flat
    ``checkpoints/logs/figures/media/runs`` layout that nothing produces.

    Returns the drive root, or ``None`` when not on Colab.
    """
    if not in_colab():
        return None

    from google.colab import drive

    names = tuple(subdirs) if subdirs is not None else DRIVE_SUBDIRS
    root = Path(repo_root) if repo_root is not None else REPO_ROOT

    drive.mount("/content/drive")
    os.makedirs(drive_root, exist_ok=True)

    for subdir in names:
        drive_dir = os.path.join(drive_root, subdir)
        local_dir = str(root / subdir)
        os.makedirs(drive_dir, exist_ok=True)

        # Refresh a stale symlink from a previous session.
        if os.path.islink(local_dir):
            os.remove(local_dir)

        # Migrate anything written locally before the mount.
        if os.path.isdir(local_dir):
            for name in os.listdir(local_dir):
                src = os.path.join(local_dir, name)
                dst = os.path.join(drive_dir, name)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(local_dir)

        os.symlink(drive_dir, local_dir)

    print(f"Google Drive mounted. Artifacts will persist at: {drive_root}")
    return drive_root


def setup_colab(
    repo_url: str = "https://github.com/kuds/rl-doom.git",
    checkout: str = "/content/rl-doom",
    extras: str = "notebooks",
) -> bool:
    """Clone + install the repo when running on Colab; no-op otherwise.

    Notebooks previously carried this inline under a comment reading
    "Uncomment the block below when running on Google Colab" — above code that
    was already uncommented. Run locally, they ``os.chdir`` into
    ``/content/rl-doom`` and raise. Guarding at runtime means one cell works in
    both places, with nothing to edit.

    Returns True if Colab setup ran.
    """
    if not in_colab():
        return False

    import subprocess

    if not os.path.exists(checkout):
        subprocess.run(["git", "clone", repo_url, checkout], check=True)
    os.chdir(f"{checkout}/notebooks")
    subprocess.run(
        ["pip", "install", "-q", "-e", f"{checkout}[{extras}]"], check=True,
    )
    return True


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_with_smoothing(
    ax: plt.Axes,
    data: np.ndarray | list,
    *,
    window: int = 20,
    raw_alpha: float = 0.3,
    raw_color: str | None = None,
    smooth_color: str | None = None,
    xlabel: str = "Episode",
    ylabel: str = "Value",
    title: str = "",
    raw_label: str = "Raw",
    smooth_label: str | None = None,
) -> None:
    """Plot a time series with optional moving-average smoothing."""
    data = np.asarray(data)
    if smooth_label is None:
        smooth_label = f"MA-{window}"

    kwargs: dict[str, Any] = {"alpha": raw_alpha, "label": raw_label}
    if raw_color:
        kwargs["color"] = raw_color
    ax.plot(data, **kwargs)

    if len(data) >= window:
        sm = np.convolve(data, np.ones(window) / window, mode="valid")
        sm_kwargs: dict[str, Any] = {"label": smooth_label}
        if smooth_color:
            sm_kwargs["color"] = smooth_color
        ax.plot(range(window - 1, window - 1 + len(sm)), sm, **sm_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()


def plot_eval_curve(
    ax: plt.Axes,
    eval_log: np.ndarray,
    *,
    col_step: int = 0,
    col_mean: int = 1,
    col_std: int = 2,
    color: str = "tab:blue",
    xlabel: str = "Training Step",
    ylabel: str = "Eval Reward",
    title: str = "Evaluation Performance",
) -> None:
    """Plot evaluation curve with error band from an eval_log array."""
    if len(eval_log) == 0:
        ax.text(0.5, 0.5, "No eval data", ha="center", va="center",
                transform=ax.transAxes)
        return
    steps = eval_log[:, col_step]
    means = eval_log[:, col_mean]
    stds = eval_log[:, col_std]
    ax.plot(steps, means, marker="o", color=color)
    ax.fill_between(steps, means - stds, means + stds, alpha=0.2, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Training timer
# ---------------------------------------------------------------------------

class TrainingTimer:
    """Simple wall-clock timer that tracks FPS."""

    def __init__(self) -> None:
        self._start = time.time()
        self.total_steps = 0

    def step(self, n: int = 1) -> None:
        self.total_steps += n

    @property
    def elapsed(self) -> float:
        return time.time() - self._start

    @property
    def fps(self) -> float:
        e = self.elapsed
        return self.total_steps / e if e > 0 else 0.0

    def summary(self) -> str:
        return f"Wall time: {self.elapsed:.1f}s | FPS: {self.fps:.0f}"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_full_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    config: dict,
    step: int,
    wall_time: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint with model weights + training metadata."""
    ckpt: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "training_step": step,
    }
    if wall_time is not None:
        ckpt["wall_time_seconds"] = wall_time
    if extra:
        ckpt.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Artifact directory setup
# ---------------------------------------------------------------------------

def ensure_artifact_dirs(root: str = "..") -> None:
    """Create standard artifact directories if they don't exist."""
    for subdir in ["checkpoints", "logs", "figures", "media", "runs"]:
        os.makedirs(f"{root}/{subdir}", exist_ok=True)
