"""Generate plain-text training summaries from run artifacts.

The CLI wrapper lives at ``scripts/generate_summary.py`` and simply forwards
to :func:`generate_run_summary` here, so notebooks can call it in-process
(``from rl_doom.summary import generate_run_summary``) without spawning a
subprocess — important on Colab where subprocesses fight the runtime for
resources and shutdown hooks.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rl_doom import __version__ as _PACKAGE_VERSION
from rl_doom.scenario_limits import EPISODE_METRIC_KEYS, EPISODE_METRIC_LABELS

SCENARIO_LABELS: dict[str, str] = {
    "basic": "Basic",
    "deadly_corridor": "Deadly Corridor",
    "defend_the_center": "Defend the Center",
    "deathmatch": "Deathmatch",
    "health_gathering": "Health Gathering",
    "my_way_home": "My Way Home",
    "predict_position": "Predict Position",
}

ALGO_LABELS: dict[str, str] = {
    "dqn": "Double DQN",
    "ppo": "PPO",
    "recurrent_ppo": "Recurrent PPO (LSTM)",
    "dreamer": "DreamerV3",
}


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _fmt_number(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _load_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    npz_path = run_dir / "metrics" / "training.npz"
    if npz_path.exists():
        return dict(np.load(npz_path, allow_pickle=True))
    return {}


def _load_termination_report(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics" / "termination_counts.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            return {}
    return {}


def _load_curriculum_report(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics" / "curriculum.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            return {}
    return {}


def _load_video_episodes(run_dir: Path) -> list[dict[str, Any]]:
    """Return per-episode stats from any ``media/*_episodes.json`` sidecar.

    Training writes a single sidecar per best-checkpoint video run
    (``<algo>_<scenario>_episodes.json``) containing one entry per greedy
    rollout — reward, length, termination, and (since we started recording
    KILLCOUNT) enemy kills. We aggregate across whatever sidecars exist so
    the summary can report best-model kill averages.
    """
    media_dir = run_dir / "media"
    if not media_dir.exists():
        return []
    episodes: list[dict[str, Any]] = []
    for path in sorted(media_dir.glob("*_episodes.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        entries = data.get("episodes") if isinstance(data, dict) else None
        if isinstance(entries, list):
            episodes.extend(e for e in entries if isinstance(e, dict))
    return episodes


def _best_eval(eval_arr: np.ndarray) -> tuple | None:
    if eval_arr.ndim != 2 or eval_arr.shape[0] == 0:
        return None
    best_idx = int(eval_arr[:, 1].argmax())
    return tuple(eval_arr[best_idx])


def generate_run_summary(run_dir: Path) -> str:
    """Build the human-readable summary string for a single run directory."""
    run_dir = Path(run_dir)
    cfg = _load_config(run_dir)
    metrics = _load_metrics(run_dir)
    terminations = _load_termination_report(run_dir)
    curriculum = _load_curriculum_report(run_dir)
    video_episodes = _load_video_episodes(run_dir)

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
    if isinstance(total_steps, (int, float)):
        total_steps_str = _fmt_number(total_steps)
    else:
        total_steps_str = str(total_steps)
    # Runtime outcomes live in metrics/training.npz; config.json only carries
    # identity + hardware + hyperparameters.
    wall_time = float(metrics.get("wall_time_seconds", 0) or 0)
    fps = float(metrics.get("fps", 0) or 0)

    ep_rewards = metrics.get("episode_rewards", np.array([]))
    ep_metrics: dict[str, np.ndarray] = {
        key: metrics.get(f"episode_{key}", np.array([]))
        for key in EPISODE_METRIC_KEYS
    }
    eval_rewards = metrics.get("eval_rewards", metrics.get("eval_log", np.array([])))
    if getattr(eval_rewards, "ndim", 0) == 1 and eval_rewards.size == 0:
        eval_rewards = np.empty((0, 5))

    if eval_rewards.shape[0] > 0:
        final_eval = eval_rewards[-1]
        final_mean, final_std = final_eval[1], final_eval[2]
        final_len_mean, final_len_std = final_eval[3], final_eval[4]
        final_eval_str = f"{final_mean:.2f} +/- {final_std:.2f}"
        final_len_str = f"{final_len_mean:.1f} +/- {final_len_std:.1f} steps"
    else:
        final_eval_str = "N/A"
        final_len_str = "N/A"

    best = _best_eval(eval_rewards)
    if best is not None:
        best_step, best_mean, best_std, best_len_mean, best_len_std = best
        best_eval_str = f"{best_mean:.2f} +/- {best_std:.2f} (at {_fmt_number(best_step)} steps)"
        best_len_str = f"{best_len_mean:.1f} +/- {best_len_std:.1f} steps"
    else:
        best_eval_str = "N/A"
        best_len_str = "N/A"

    if ep_rewards.size >= 20:
        recent_mean = float(np.mean(ep_rewards[-20:]))
        recent_std = float(np.std(ep_rewards[-20:]))
        recent_str = f"{recent_mean:.2f} +/- {recent_std:.2f}"
    elif ep_rewards.size > 0:
        recent_str = f"{float(np.mean(ep_rewards)):.2f} +/- {float(np.std(ep_rewards)):.2f}"
    else:
        recent_str = "N/A"

    # Prefer the version recorded in config.json (captures the code that
    # produced the artefacts); fall back to the installed package version for
    # runs written before the field was added.
    version_str = cfg.get("version") or _PACKAGE_VERSION

    title = f"rl-doom: {algo_label} on {env_label}"
    lines = [
        title,
        "=" * max(50, len(title)),
        "",
        "Project:        rl-doom",
        f"Version:        {version_str}",
        f"Environment:    ViZDoom — {env_label}",
        f"Algorithm:      {algo_label}",
        f"Seed:           {cfg.get('seed', 'N/A')}",
        f"Date:           {date_str}",
        f"Status:         {status}",
        f"Git SHA:        {cfg.get('git_sha', 'N/A')}",
        f"Timesteps:      {total_steps_str}",
    ]

    # Make a shortened run legible: without this, "Timesteps: 900,000" against
    # a config saying 2,500,000 reads as a crash rather than a plateau.
    requested = cfg.get("hyperparams", {}).get("total_timesteps")
    if cfg.get("stopped_early") and requested:
        lines.append(
            f"Early stop:     yes — plateaued at "
            f"{_fmt_number(cfg.get('stopped_at_step'))} of "
            f"{_fmt_number(requested)} requested",
        )

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

    # Headline success rate (goal-reached fraction across all training
    # episodes). This is the "did the agent actually solve the task?" metric,
    # distinct from reward plateaus driven by kill-and-die local optima.
    counts_for_rate = terminations.get("counts") or {}
    total_for_rate = terminations.get("total_episodes", sum(counts_for_rate.values())) or 0
    if total_for_rate:
        goal_n = int(counts_for_rate.get("goal_reached", 0))
        lines.append(
            f"Success rate:   {goal_n / total_for_rate * 100:.1f}% "
            f"({goal_n:,} / {total_for_rate:,} episodes)",
        )

    # Per-episode combat/exploration metrics come from ViZDoom game
    # variables (KILLCOUNT, DAMAGECOUNT, etc.), captured in the Monitor
    # CSVs. Scenarios that don't expose a given metric (e.g. SECRETCOUNT
    # in ``basic``) see an all-zero array; gate on ``any()`` so the block
    # stays tidy instead of listing uniformly-zero lines.
    nonzero_metrics = [
        (key, arr)
        for key, arr in ep_metrics.items()
        if getattr(arr, "size", 0) > 0 and bool(np.any(arr))
    ]
    if nonzero_metrics:
        lines += ["", "Training Episode Metrics", "-" * 40]
        for key, arr in nonzero_metrics:
            label = EPISODE_METRIC_LABELS.get(key, key)
            total = int(np.sum(arr))
            mean = float(np.mean(arr))
            line = (
                f"  {label + ':':16s}{_fmt_number(total):>10s} total  "
                f"({mean:.2f} avg / episode)"
            )
            if arr.size >= 20:
                recent_mean = float(np.mean(arr[-20:]))
                recent_std = float(np.std(arr[-20:]))
                line += (
                    f"  |  last 20: {recent_mean:.2f} +/- {recent_std:.2f}"
                )
            lines.append(line)

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
        if "gpu_capability" in gpu:
            lines.append(f"  Compute cap:    {gpu['gpu_capability']}")
        if "cuda_version" in gpu:
            lines.append(f"  CUDA:           {gpu['cuda_version']}")
        if "cudnn_version" in gpu:
            lines.append(f"  cuDNN:          {gpu['cudnn_version']}")
        if "torch_version" in gpu:
            lines.append(f"  PyTorch:        {gpu['torch_version']}")

    if hp:
        lines += ["", "Hyperparameters", "-" * 40]
        for k, v in hp.items():
            lines.append(f"  {k:20s} {v}")

    if best is not None:
        lines += [
            "",
            f"Best Checkpoint Evaluation (step {_fmt_number(best_step)})",
            "-" * 40,
            f"  Reward:         {best_mean:.2f} +/- {best_std:.2f}",
            f"  Ep length:      {best_len_str}",
        ]
        # Per-episode combat/exploration metrics from the best-checkpoint
        # video rollouts, when a playback JSON is available. The videos
        # are recorded with the best eval checkpoint, so these averages
        # are the headline "what the shipped model actually does in an
        # episode" numbers. Skip metrics that stayed at 0 across every
        # rollout so scenario-irrelevant fields don't clutter the block.
        n_rollouts = len(video_episodes)
        if n_rollouts:
            for key in EPISODE_METRIC_KEYS:
                values = [
                    int(e[key])
                    for e in video_episodes
                    if isinstance(e.get(key), (int, float))
                ]
                if not values or not any(values):
                    continue
                arr = np.asarray(values, dtype=np.int64)
                label = EPISODE_METRIC_LABELS.get(key, key)
                lines.append(
                    f"  Avg {label.lower():12s} {float(arr.mean()):.2f} "
                    f"(over {arr.size} rollout"
                    f"{'s' if arr.size != 1 else ''})",
                )

    counts = terminations.get("counts") or {}
    if counts:
        total = terminations.get("total_episodes", sum(counts.values())) or 0
        lines += [
            "",
            "Episode Terminations",
            "-" * 40,
            f"  Total:          {_fmt_number(total)}",
        ]
        # Stable ordering: goal_reached first (the "win"), then death,
        # timeout, then anything else alphabetically.
        priority = {"goal_reached": 0, "death": 1, "timeout": 2}
        ordered = sorted(
            counts.items(), key=lambda kv: (priority.get(kv[0], 99), kv[0]),
        )
        for reason, count in ordered:
            frac = (count / total) if total else 0.0
            lines.append(
                f"  {reason:14s} {count:>6}  ({frac * 100:5.1f}%)",
            )

    if curriculum:
        stages = curriculum.get("stages") or []
        promotions = curriculum.get("promotions") or []
        final_skill = curriculum.get("final_skill")
        final_num_bots = curriculum.get("final_num_bots")
        final_stage_index = curriculum.get("final_stage_index")
        # Only render the section when the callback actually ran, i.e.
        # at least one of the knobs ended with a non-None value.
        if final_skill is not None or final_num_bots is not None:
            lines += ["", "Curriculum", "-" * 40]
            stage_str = (
                f" (stage {int(final_stage_index) + 1} / {len(stages)})"
                if final_stage_index is not None and stages
                else ""
            )
            if final_skill is not None:
                lines.append(f"  Final skill:    {final_skill}{stage_str}")
            if final_num_bots is not None:
                # If we already printed the stage position on the skill line,
                # omit it here to avoid duplicating.
                suffix = "" if final_skill is not None else stage_str
                lines.append(f"  Final num_bots: {final_num_bots}{suffix}")
            # Promotion timeline: each entry is
            # {step, skill, num_bots, trigger, eval_mean_reward}. The first
            # entry is always the initial stage; skip it so we only list the
            # actual promotions.
            promoted = [p for p in promotions if p.get("trigger") == "promotion"]
            if promoted:
                lines.append(f"  Promotions:     {len(promoted)}")
                for p in promoted:
                    step = _fmt_number(p.get("step", 0))
                    mean_r = p.get("eval_mean_reward")
                    mean_str = (
                        f"{mean_r:.2f}" if isinstance(mean_r, (int, float)) else "N/A"
                    )
                    # Show whichever knob(s) the stage carries. The promotion
                    # entry records both fields; ``None`` means "not used".
                    parts: list[str] = []
                    if p.get("skill") is not None:
                        parts.append(f"skill {p['skill']}")
                    if p.get("num_bots") is not None:
                        parts.append(f"num_bots {p['num_bots']}")
                    knob_str = " + ".join(parts) if parts else "?"
                    lines.append(
                        f"    step {step:>12s} -> {knob_str}  (eval_mean={mean_str})",
                    )

    lines.append("")
    return "\n".join(lines)


def discover_runs(root: Path) -> list[Path]:
    """Find every run directory under ``training_jobs/``."""
    runs: list[Path] = []
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
