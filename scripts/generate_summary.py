#!/usr/bin/env python3
"""Generate a plain-text training summary for each completed run.

Reads ``training_jobs/<env>/<algo>/runs/*/`` and produces a
``stage_summary.txt`` in each run directory (and one combined summary
at the repo root).

Usage (from repo root or Colab):
    python scripts/generate_summary.py

Or to summarise a single run:
    python scripts/generate_summary.py training_jobs/basic/dqn/runs/<run_id>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_JOBS = REPO_ROOT / "training_jobs"

SCENARIO_LABELS = {
    "basic": "Basic",
    "deadly_corridor": "Deadly Corridor",
    "defend_the_center": "Defend the Center",
    "deathmatch": "Deathmatch",
    "health_gathering": "Health Gathering",
    "my_way_home": "My Way Home",
    "predict_position": "Predict Position",
}

ALGO_LABELS = {"dqn": "Double DQN", "ppo": "PPO"}


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _fmt_number(n) -> str:
    return f"{int(n):,}"


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def _load_metrics(run_dir: Path) -> dict:
    npz_path = run_dir / "metrics" / "training.npz"
    if npz_path.exists():
        return dict(np.load(npz_path, allow_pickle=True))
    return {}


def _best_eval(eval_arr: np.ndarray) -> tuple:
    """Return (step, mean, std, mean_len, std_len) for the best eval checkpoint."""
    if eval_arr.ndim != 2 or eval_arr.shape[0] == 0:
        return None
    best_idx = int(eval_arr[:, 1].argmax())
    return tuple(eval_arr[best_idx])


def generate_run_summary(run_dir: Path) -> str:
    cfg = _load_config(run_dir)
    metrics = _load_metrics(run_dir)

    env_name = cfg.get("env", run_dir.parents[1].name)
    algo_name = cfg.get("algo", run_dir.parents[0].name)
    env_label = SCENARIO_LABELS.get(env_name, env_name)
    algo_label = ALGO_LABELS.get(algo_name, algo_name.upper())
    hp = cfg.get("hyperparams", {})
    gpu = cfg.get("gpu_setup", {})
    status = cfg.get("status", "unknown")

    started_at = cfg.get("started_at", "")
    if started_at:
        try:
            dt = datetime.fromisoformat(started_at)
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            date_str = started_at
    else:
        date_str = "N/A"

    total_steps = hp.get("total_steps", hp.get("total_timesteps", "N/A"))
    wall_time = cfg.get("wall_time_seconds", float(metrics.get("wall_time_seconds", 0)))
    fps = float(metrics.get("fps", 0))

    ep_rewards = metrics.get("episode_rewards", np.array([]))
    ep_lengths = metrics.get("episode_lengths", np.array([]))

    eval_rewards = metrics.get("eval_rewards", metrics.get("eval_log", np.array([])))
    if eval_rewards.ndim == 1 and eval_rewards.size == 0:
        eval_rewards = np.empty((0, 5))

    # Final eval
    if eval_rewards.shape[0] > 0:
        final_eval = eval_rewards[-1]
        final_mean, final_std = final_eval[1], final_eval[2]
        final_len_mean, final_len_std = final_eval[3], final_eval[4]
        final_eval_str = f"{final_mean:.2f} +/- {final_std:.2f}"
        final_len_str = f"{final_len_mean:.1f} +/- {final_len_std:.1f} steps"
    else:
        final_eval_str = "N/A"
        final_len_str = "N/A"

    # Best eval
    best = _best_eval(eval_rewards)
    if best is not None:
        best_step, best_mean, best_std, best_len_mean, best_len_std = best
        best_eval_str = f"{best_mean:.2f} +/- {best_std:.2f} (at {_fmt_number(best_step)} steps)"
        best_len_str = f"{best_len_mean:.1f} +/- {best_len_std:.1f} steps"
    else:
        best_eval_str = "N/A"
        best_len_str = "N/A"

    # Recent training reward (last 20 episodes)
    if ep_rewards.size >= 20:
        recent_mean = float(np.mean(ep_rewards[-20:]))
        recent_std = float(np.std(ep_rewards[-20:]))
        recent_str = f"{recent_mean:.2f} +/- {recent_std:.2f}"
    elif ep_rewards.size > 0:
        recent_str = f"{float(np.mean(ep_rewards)):.2f} +/- {float(np.std(ep_rewards)):.2f}"
    else:
        recent_str = "N/A"

    # Build the summary
    title = f"rl-doom: {algo_label} on {env_label}"
    lines = [
        title,
        "=" * max(50, len(title)),
        "",
        f"Project:        rl-doom",
        f"Environment:    ViZDoom — {env_label}",
        f"Algorithm:      {algo_label}",
        f"Seed:           {cfg.get('seed', 'N/A')}",
        f"Date:           {date_str}",
        f"Status:         {status}",
        f"Git SHA:        {cfg.get('git_sha', 'N/A')}",
        f"Timesteps:      {_fmt_number(total_steps) if isinstance(total_steps, (int, float)) else total_steps}",
    ]

    if wall_time > 0:
        lines.append(f"Duration:       {_fmt_duration(wall_time)}")
        if fps > 0:
            lines.append(f"Throughput:     {fps:.0f} FPS")

    lines += [
        f"Final eval:     {final_eval_str}",
        f"Avg ep length:  {final_len_str}",
        f"Best eval:      {best_eval_str}",
        f"Recent train:   {recent_str} (last 20 episodes)",
    ]

    # GPU / device info
    if gpu:
        lines += [
            "",
            "Device",
            "-" * 40,
            f"  Device:         {gpu.get('device', 'N/A')}",
        ]
        if "gpu_name" in gpu:
            lines.append(f"  GPU:            {gpu['gpu_name']}")
        if "gpu_memory_total_mb" in gpu:
            lines.append(f"  VRAM:           {gpu['gpu_memory_total_mb']} MB")
        if "cuda_version" in gpu:
            lines.append(f"  CUDA:           {gpu['cuda_version']}")
        if "torch_version" in gpu:
            lines.append(f"  PyTorch:        {gpu['torch_version']}")

    # Hyperparameters
    if hp:
        lines += [
            "",
            "Hyperparameters",
            "-" * 40,
        ]
        for k, v in hp.items():
            lines.append(f"  {k:20s} {v}")

    # Best-checkpoint evaluation block (mirror of the example)
    if best is not None:
        lines += [
            "",
            f"Best Checkpoint Evaluation (step {_fmt_number(best_step)})",
            "-" * 40,
            f"  Reward:         {best_mean:.2f} +/- {best_std:.2f}",
            f"  Ep length:      {best_len_str}",
        ]

    lines.append("")
    return "\n".join(lines)


def discover_runs(root: Path) -> list[Path]:
    """Find all run directories under training_jobs/."""
    runs = []
    if not root.exists():
        return runs
    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir():
            continue
        for algo_dir in sorted(env_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            runs_dir = algo_dir / "runs"
            if runs_dir.exists():
                for run_dir in sorted(runs_dir.iterdir()):
                    if run_dir.is_dir() and (run_dir / "config.json").exists():
                        runs.append(run_dir)
    return runs


def main():
    if len(sys.argv) > 1:
        run_dirs = [Path(p).resolve() for p in sys.argv[1:]]
    else:
        run_dirs = discover_runs(TRAINING_JOBS)

    if not run_dirs:
        print("No training runs found. Run a training notebook first.")
        sys.exit(0)

    summaries = []
    for run_dir in run_dirs:
        summary = generate_run_summary(run_dir)
        summaries.append(summary)

        out_path = run_dir / "stage_summary.txt"
        out_path.write_text(summary)
        print(f"Wrote {out_path}")

    combined = "\n\n".join(summaries)
    combined_path = REPO_ROOT / "stage_summary.txt"
    combined_path.write_text(combined + "\n")
    print(f"\nCombined summary written to {combined_path}")
    print()
    print(combined)


if __name__ == "__main__":
    main()
