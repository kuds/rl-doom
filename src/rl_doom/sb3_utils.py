"""Stable-Baselines3 training glue for rl-doom.

Centralises the boilerplate that every training notebook needs:
* GPU / environment metadata capture (written into ``config.json``)
* A single ``train_sb3`` entry point that runs SB3 training end-to-end and
  persists *all* artifacts (plots, metrics, video, checkpoint, summary)
  **per scenario**, before moving on to the next scenario.

Design choices:
* Colab-friendly: we never use ``SubprocVecEnv``; all parallelism uses
  ``DummyVecEnv`` (see :func:`rl_doom.env.make_sb3_env`).
* Artifact-local: the caller passes a ``run_dir`` and every side effect
  writes under that dir. This avoids coupling to a specific notebook.
* Logger: we configure SB3's logger with ``stdout + tensorboard + csv``,
  then parse the CSV after training to build summary plots. No running
  in-memory accumulators needed.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

# ``rl_doom.env`` imports ``vizdoom`` at module load time, but the ViZDoom
# binary isn't always available where ``sb3_utils`` is needed (e.g. lean
# test runners that only exercise the YAML -> policy_kwargs translation).
# Defer that import to the two functions that actually build envs.
from rl_doom.curriculum import (
    CurriculumStage,
    SkillCurriculumCallback,
    parse_curriculum_config,
)
from rl_doom.early_stop import StopOnEvalPlateau
from rl_doom.paths import mark_run_status, update_latest_symlink
from rl_doom.scenario_limits import EPISODE_METRIC_KEYS

# Algorithms that use a Generalized Advantage Estimation rollout buffer and
# benefit from VecNormalize reward scaling. RecurrentPPO is the LSTM variant
# of PPO from sb3-contrib; it shares PPO's hyperparameters and adds an
# ``lstm_hidden_size`` knob via policy_kwargs.
_PPO_FAMILY = {"ppo", "recurrent_ppo"}


def _recurrent_ppo_class() -> type[BaseAlgorithm]:
    """Import ``RecurrentPPO`` on demand.

    ``sb3-contrib`` is an optional dependency; deferring the import keeps
    PPO/DQN-only installs working.
    """
    from sb3_contrib import RecurrentPPO

    return RecurrentPPO


# Algorithm name -> SB3 class, behind lazy factories so ``sb3_contrib`` is only
# imported when it is actually asked for.
#
# One registry rather than a dispatch per call site. There used to be three,
# and two of them ended in a silent ``else: DQN`` catch-all, so any name that
# was not exactly "ppo" got loaded through ``DQN.load``. That is not
# hypothetical: notebook 05 analyses ``["dqn", "ppo", "recurrent_ppo"]`` and
# every recurrent_ppo run failed to load, while ``_record_video``'s copy would
# have mis-loaded at the very end of a multi-hour run, after training had
# already succeeded.
_ALGO_CLASSES: dict[str, Callable[[], type[BaseAlgorithm]]] = {
    "ppo": lambda: PPO,
    "dqn": lambda: DQN,
    "recurrent_ppo": _recurrent_ppo_class,
}


def _format_gib(n_bytes: float) -> str:
    return f"{n_bytes / 2**30:.1f} GiB"


def estimate_replay_bytes(buffer_size: int, obs_space: Any, n_envs: int = 1) -> int:
    """Bytes SB3's ``ReplayBuffer`` will allocate for observations.

    SB3 keeps ``observations`` *and* ``next_observations`` as separate arrays,
    so the cost is two full copies of the frame stack per transition. The
    ``optimize_memory_usage=True`` layout that would halve this is not usable
    here: SB3 rejects it together with ``handle_timeout_termination=True``,
    which is what makes a truncated episode bootstrap ``V(s')`` instead of
    treating a scenario timeout as absorbing (see ``DoomEnv.step``). Correct
    value targets are worth more than the memory.

    Non-observation arrays (actions, rewards, dones, timeouts) are a few bytes
    per transition and are ignored.
    """
    obs_bytes = int(np.prod(obs_space.shape)) * np.dtype(obs_space.dtype).itemsize
    per_env = max(buffer_size // max(n_envs, 1), 1)
    return per_env * max(n_envs, 1) * obs_bytes * 2


def check_replay_buffer_fits(
    buffer_size: int, obs_space: Any, n_envs: int = 1, *, headroom: float = 0.9,
) -> None:
    """Fail fast when a DQN replay buffer cannot fit in available memory.

    SB3 has its own check, but it only warns, only *after* the allocation is
    attempted, and only when ``psutil`` is importable — which it silently is
    not in a default install. The observed failure without this guard is a bare
    ``numpy._core._exceptions._ArrayMemoryError`` partway into a run that has
    already spent minutes building envs.
    """
    needed = estimate_replay_bytes(buffer_size, obs_space, n_envs)
    try:
        import psutil
    except ImportError:
        # Without psutil there is nothing to compare against; the estimate is
        # still worth printing so an OOM later is at least attributable.
        print(
            f"[train_sb3] replay buffer needs ~{_format_gib(needed)} "
            f"(buffer_size={buffer_size:,}); install psutil to have this "
            f"checked against available memory before allocating.",
        )
        return

    available = psutil.virtual_memory().available
    if needed > available * headroom:
        per_transition = max(estimate_replay_bytes(1, obs_space, n_envs), 1)
        fits = int(available * headroom / per_transition)
        raise MemoryError(
            f"DQN replay buffer needs ~{_format_gib(needed)} "
            f"(buffer_size={buffer_size:,} x {obs_space.shape} uint8 x2 for "
            f"next_obs) but only {_format_gib(available)} is available.\n"
            f"Lower `hyperparams.buffer_size` in the config: "
            f"~{fits:,} transitions fit in the memory you have.\n"
            f"Note that `optimize_memory_usage=True` would halve this but SB3 "
            f"rejects it alongside `handle_timeout_termination=True`, which "
            f"this pipeline needs for correct value targets on timeouts.",
        )


def apply_stage0_env_settings(
    stage0: CurriculumStage,
    *,
    doom_skill: int | None,
    num_bots: int,
    verbose: bool = True,
) -> tuple[int | None, int]:
    """Fold a curriculum's first stage into the env settings it owns.

    Returns the ``(doom_skill, num_bots)`` the training and eval envs should be
    built with. A stage only overrides the knobs it actually sets — the rest
    stay as the caller configured them.

    That distinction is the whole point. Not every curriculum ramps difficulty:
    the deathmatch curricula ramp *bots*, so their stages carry only
    ``num_bots`` and leave ``skill`` as ``None``. Overriding unconditionally
    replaced the config's ``doom_skill: 3`` with ``None``, which falls through
    ``SCENARIO_DEFAULT_SKILL`` (empty) to the scenario cfg's own default — so
    the curriculum arm trained at a different difficulty from its baseline
    sibling, confounding the comparison the matrix exists to make.
    ``apply_stage_to_doom_env`` guards on ``skill is not None`` too, so the
    callback could not recover it either.
    """
    if stage0.skill is not None:
        if verbose and doom_skill is not None and doom_skill != stage0.skill:
            print(
                f"[train_sb3] curriculum enabled: overriding doom_skill="
                f"{doom_skill} with stage-0 skill={stage0.skill}",
            )
        doom_skill = stage0.skill
    if stage0.num_bots is not None:
        if verbose and num_bots and num_bots != stage0.num_bots:
            print(
                f"[train_sb3] curriculum enabled: overriding num_bots="
                f"{num_bots} with stage-0 num_bots={stage0.num_bots}",
            )
        # Seed at construction as well as through the callback. The callback
        # applies stage 0 in ``_on_training_start``, which is early enough for
        # training, but building the envs with the right value keeps pre- and
        # post-callback state consistent — and the video env at the end reads
        # these same variables.
        num_bots = stage0.num_bots
    return doom_skill, num_bots


def resolve_algo_class(algo: str) -> type[BaseAlgorithm]:
    """Return the SB3 class for *algo*, raising on anything unrecognised.

    Raising matters more than it looks: the failure mode this replaces was
    loading a checkpoint through the wrong class, which surfaces as a confusing
    shape/observation-space error far from the actual mistake — or, worse, does
    not surface at all.
    """
    factory = _ALGO_CLASSES.get(algo.lower())
    if factory is None:
        raise ValueError(
            f"Unknown algo {algo!r}; expected one of {sorted(_ALGO_CLASSES)}.",
        )
    return factory()

# ---------------------------------------------------------------------------
# GPU / environment metadata
# ---------------------------------------------------------------------------


def gpu_info() -> dict[str, Any]:
    """Capture GPU + runtime metadata for ``config.json``.

    Safe to call on CPU-only machines — the ``gpu_*`` fields are simply
    omitted when no CUDA device is present.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_mb"] = round(props.total_memory / 1024**2)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
        info["gpu_multi_processor_count"] = props.multi_processor_count
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        )
    return info


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _build_model(
    algo: str,
    vec_env,
    *,
    hyperparams: dict[str, Any],
    tensorboard_log: str | Path,
    device: str | torch.device,
    seed: int,
    verbose: int = 0,
    policy_kwargs: dict[str, Any] | None = None,
) -> BaseAlgorithm:
    """Instantiate a PPO or DQN model from a flat hyperparam dict.

    ``policy_kwargs`` is passed straight through to SB3. Pass ``None`` (the
    default) to keep SB3's baked-in policy defaults (NatureCNN with
    features_dim=512 and no extra MLP layers).
    """
    algo_norm = algo.lower()
    pk = policy_kwargs or {}
    # Validate against the registry up front, so an unknown name fails here
    # rather than further downstream. The branches below still name concrete
    # classes: each algorithm takes a different constructor signature, and
    # going through the registry's ``type[BaseAlgorithm]`` would erase the
    # types that let mypy catch a misspelled hyperparameter kwarg.
    resolve_algo_class(algo_norm)
    if algo_norm in _PPO_FAMILY:
        ppo_kwargs = dict(
            learning_rate=hyperparams["lr"],
            n_steps=hyperparams["n_steps"],
            batch_size=hyperparams["batch_size"],
            n_epochs=hyperparams["n_epochs"],
            gamma=hyperparams["gamma"],
            gae_lambda=hyperparams["gae_lambda"],
            clip_range=hyperparams["clip_eps"],
            ent_coef=hyperparams["entropy_coef"],
            vf_coef=hyperparams["value_coef"],
            max_grad_norm=hyperparams["max_grad_norm"],
            tensorboard_log=str(tensorboard_log),
            device=device,
            seed=seed,
            verbose=verbose,
            policy_kwargs=pk,
        )
        if algo_norm == "recurrent_ppo":
            return _recurrent_ppo_class()("CnnLstmPolicy", vec_env, **ppo_kwargs)
        return PPO("CnnPolicy", vec_env, **ppo_kwargs)
    if algo_norm == "dqn":
        # Check before constructing: SB3 allocates the whole buffer eagerly in
        # DQN.__init__, so an oversized buffer_size otherwise surfaces as a
        # numpy MemoryError after the envs are already up.
        check_replay_buffer_fits(
            int(hyperparams["buffer_size"]),
            vec_env.observation_space,
            n_envs=getattr(vec_env, "num_envs", 1),
        )
        return DQN(
            "CnnPolicy",
            vec_env,
            learning_rate=hyperparams["lr"],
            buffer_size=hyperparams["buffer_size"],
            learning_starts=hyperparams["learning_starts"],
            batch_size=hyperparams["batch_size"],
            tau=hyperparams.get("tau", 1.0),
            gamma=hyperparams["gamma"],
            train_freq=hyperparams["train_freq"],
            gradient_steps=hyperparams.get("gradient_steps", 1),
            target_update_interval=hyperparams["target_update_interval"],
            exploration_fraction=hyperparams["exploration_fraction"],
            exploration_initial_eps=hyperparams["eps_start"],
            exploration_final_eps=hyperparams["eps_end"],
            max_grad_norm=hyperparams.get("max_grad_norm", 10.0),
            tensorboard_log=str(tensorboard_log),
            device=device,
            seed=seed,
            verbose=verbose,
            policy_kwargs=pk,
        )
    # Unreachable: resolve_algo_class above rejects anything not in the
    # registry. Kept so adding a registry entry without a construction branch
    # here fails loudly instead of falling off the end returning None.
    raise ValueError(
        f"Algo {algo!r} is registered but has no construction branch in "
        f"_build_model.",
    )


def policy_kwargs_from_config(policy_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Translate a YAML ``policy:`` block into SB3 ``policy_kwargs``.

    The YAML exposes two ergonomic fields — ``features_dim`` (NatureCNN final
    FC width) and ``net_arch`` (extra MLP layers after the feature extractor)
    — and this helper reshapes them into the nested dict SB3 actually
    consumes. Unset fields are omitted so SB3's own defaults win.

    ``policy_cfg`` may also contain already-nested keys (e.g.
    ``features_extractor_kwargs``, ``activation_fn``) that pass through
    unchanged, letting callers reach less-common knobs without another
    round of plumbing.
    """
    if not policy_cfg:
        return {}
    out: dict[str, Any] = {}
    # Flat ergonomic keys.
    features_dim = policy_cfg.get("features_dim")
    if features_dim is not None:
        out.setdefault("features_extractor_kwargs", {})["features_dim"] = int(features_dim)
    if "net_arch" in policy_cfg and policy_cfg["net_arch"] is not None:
        out["net_arch"] = policy_cfg["net_arch"]
    # Recurrent-only knobs. Harmless on non-recurrent policies because they
    # won't be present in those YAML files; passing them to a non-LSTM policy
    # would raise, so guard at the YAML layer (only recurrent_ppo configs set
    # these keys).
    if "lstm_hidden_size" in policy_cfg and policy_cfg["lstm_hidden_size"] is not None:
        out["lstm_hidden_size"] = int(policy_cfg["lstm_hidden_size"])
    if "n_lstm_layers" in policy_cfg and policy_cfg["n_lstm_layers"] is not None:
        out["n_lstm_layers"] = int(policy_cfg["n_lstm_layers"])
    # Pass-through for anything the caller nested themselves.
    for k, v in policy_cfg.items():
        if k in {"features_dim", "net_arch", "lstm_hidden_size", "n_lstm_layers"}:
            continue
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Termination tracking
# ---------------------------------------------------------------------------


class TerminationTracker(BaseCallback):
    """Aggregate per-episode termination reasons across the vec env.

    ``DoomEnv`` writes ``termination_reason``/``episode_tics``/``final_health``
    into the ``info`` dict when an episode ends. SB3's ``Monitor`` wrapper
    preserves the raw info when the sub-env finishes (it only adds an
    ``episode`` key), so we can read ``info["termination_reason"]`` directly
    on the step where ``dones[i]`` is True.

    The callback maintains running counts plus a compact history of the
    per-episode metadata (termination reason, length in tics, final health),
    so downstream code can build a termination timeline or pivot by
    reason × step.
    """

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.counts: dict[str, int] = {}
        self.episodes: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for done, info in zip(dones, infos):
            if not done:
                continue
            reason = info.get("termination_reason", "unknown")
            self.counts[reason] = self.counts.get(reason, 0) + 1
            episode: dict[str, Any] = {
                "step": int(self.num_timesteps),
                "reason": reason,
                "episode_tics": info.get("episode_tics"),
                "final_health": info.get("final_health"),
            }
            for key in EPISODE_METRIC_KEYS:
                episode[key] = info.get(key)
            self.episodes.append(episode)
        return True


def _write_termination_report(
    tracker: TerminationTracker, metrics_dir: Path,
) -> None:
    """Persist termination counts + per-episode records to ``metrics/``."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    total = sum(tracker.counts.values())
    report: dict[str, Any] = {
        "total_episodes": total,
        "counts": tracker.counts,
        "fractions": {
            k: (v / total if total else 0.0) for k, v in tracker.counts.items()
        },
    }
    (metrics_dir / "termination_counts.json").write_text(
        json.dumps(report, indent=2),
    )
    if tracker.episodes:
        with (metrics_dir / "termination_episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step", "reason", "episode_tics", "final_health",
                    *EPISODE_METRIC_KEYS,
                ],
            )
            writer.writeheader()
            for row in tracker.episodes:
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------


def _load_progress_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """Load SB3's ``progress.csv`` into a dict of numpy arrays.

    Returns ``{}`` if the file is missing (e.g. training finished before the
    first log flush on a tiny run).
    """
    if not csv_path.exists():
        return {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    out: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            try:
                out[k].append(float(v))
            except (TypeError, ValueError):
                out[k].append(float("nan"))
    return {k: np.array(v, dtype=np.float32) for k, v in out.items()}


def _load_evaluations(run_dir: Path) -> dict[str, np.ndarray]:
    """Load SB3 ``EvalCallback`` output if present."""
    eval_npz = run_dir / "metrics" / "evaluations.npz"
    if not eval_npz.exists():
        return {}
    with np.load(eval_npz) as data:
        return {k: data[k] for k in data.files}


def _plot_learning_curves(
    algo: str,
    scenario: str,
    progress: dict[str, np.ndarray],
    episodes: dict[str, np.ndarray],
    evaluations: dict[str, np.ndarray],
    *,
    out_path: Path,
) -> None:
    """Render a 2×3 learning-curves grid matching the legacy notebook layout."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ep_rewards = episodes.get("rewards", np.array([]))
    ep_lengths = episodes.get("lengths", np.array([]))

    # (0,0) Episode rewards
    ax = axes[0, 0]
    if ep_rewards.size:
        ax.plot(ep_rewards, alpha=0.3, label="Raw")
        if ep_rewards.size >= 20:
            sm = np.convolve(ep_rewards, np.ones(20) / 20, mode="valid")
            ax.plot(range(19, 19 + len(sm)), sm, label="MA-20")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No episode data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"Episode Rewards — {scenario}")

    # (0,1) Episode lengths
    ax = axes[0, 1]
    if ep_lengths.size:
        ax.plot(ep_lengths, alpha=0.3, color="green", label="Raw")
        if ep_lengths.size >= 20:
            sm = np.convolve(ep_lengths, np.ones(20) / 20, mode="valid")
            ax.plot(range(19, 19 + len(sm)), sm, color="darkgreen", label="MA-20")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Lengths")

    # Pick algorithm-specific panels. SB3 key names are stable.
    # RecurrentPPO logs the same train/* keys as PPO, so they share a panel set.
    algo_norm = algo.lower()
    panel_defs: list[tuple[tuple[int, int], str, str, str]]
    if algo_norm in _PPO_FAMILY:
        panel_defs = [
            ((0, 2), "train/policy_gradient_loss", "Policy Loss", "PPO Policy Loss"),
            ((1, 0), "train/value_loss", "Value Loss", "Value Function Loss"),
            ((1, 1), "train/entropy_loss", "Entropy (negated)", "Policy Entropy Loss"),
            ((1, 2), "train/clip_fraction", "Clip Fraction", "PPO Clip Fraction"),
        ]
    else:  # dqn
        panel_defs = [
            ((0, 2), "train/loss", "Q Loss", "DQN Loss"),
            ((1, 0), "rollout/exploration_rate", "Epsilon", "Exploration ε"),
            ((1, 1), "train/learning_rate", "LR", "Learning Rate"),
            ((1, 2), "rollout/ep_rew_mean", "Mean Ep Reward", "Rollout Reward (mean)"),
        ]

    x_key = "time/total_timesteps"
    xs = progress.get(x_key)
    for (r, c), key, ylabel, title in panel_defs:
        ax = axes[r, c]
        ys = progress.get(key)
        if ys is not None and xs is not None and len(ys) == len(xs):
            mask = ~np.isnan(ys)
            ax.plot(xs[mask], ys[mask])
        elif ys is not None:
            ax.plot(ys)
        else:
            ax.text(0.5, 0.5, f"{key} not logged", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        ax.set_xlabel("Timesteps" if xs is not None else "Log step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Eval curve as a second, simpler figure.
    if evaluations:
        eval_out = out_path.with_name("eval_performance.png")
        _plot_eval_performance(algo, scenario, evaluations, eval_out)


def _plot_eval_performance(
    algo: str, scenario: str, evaluations: dict[str, np.ndarray], out_path: Path,
) -> None:
    timesteps = evaluations.get("timesteps")
    results = evaluations.get("results")  # (n_evals, n_eval_episodes)
    ep_lengths = evaluations.get("ep_lengths")
    if timesteps is None or results is None or timesteps.size == 0:
        return

    means = results.mean(axis=1)
    stds = results.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(timesteps, means, marker="o")
    ax.fill_between(timesteps, means - stds, means + stds, alpha=0.2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Eval Reward")
    ax.set_title(f"{algo.upper()} — Eval Reward ({scenario})")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Derive both series inside the guard. They used to be computed above as a
    # correlated pair of Optionals and only ``len_means`` was checked here, so
    # the band arithmetic read a value the guard never covered. Unreachable in
    # practice — both came from the same ``ep_lengths is not None`` — but the
    # invariant was implicit, and editing one line would have broken it
    # silently. numpy 2.5's stricter stubs flagged it.
    if ep_lengths is not None:
        len_means = ep_lengths.mean(axis=1)
        len_stds = ep_lengths.std(axis=1)
        ax.plot(timesteps, len_means, marker="o", color="green")
        ax.fill_between(
            timesteps, len_means - len_stds, len_means + len_stds,
            alpha=0.2, color="green",
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Eval Episode Length")
    ax.set_title(f"{algo.upper()} — Eval Episode Length ({scenario})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _record_video(
    model: BaseAlgorithm,
    scenario: str,
    out_path: Path,
    *,
    fps: int = 20,
    max_steps: int = 5_000,
    n_episodes: int = 1,
    resize_shape: tuple[int, int] = (84, 84),
    frame_skip: int = 4,
    num_stack: int = 4,
    use_compound_actions: bool = True,
    doom_skill: int | None = None,
    num_bots: int = 0,
) -> list[Path]:
    """Record ``n_episodes`` greedy rollouts as separate video files.

    For each of the ``n_episodes`` rollouts we write a standalone video
    file named ``<stem>_ep<N>.mp4`` next to ``out_path``; e.g.
    ``media/ppo_deadly_corridor_ep1.mp4``. A single ``_episodes.json``
    sidecar lists the per-episode reward, length (steps), and
    termination reason in playback order, alongside the env-config
    (``doom_skill``, ``num_bots``, and wrapper settings) the clips were
    recorded under so downstream tooling can tell at a glance which
    difficulty the videos actually show.

    ``max_steps`` is per-episode, not total. Returns the list of video
    paths actually written (empty if encoding failed for every episode).
    """
    import imageio

    from rl_doom.env import make_wrapped_env

    env = make_wrapped_env(
        scenario,
        resize_shape=resize_shape,
        frame_skip=frame_skip,
        num_stack=num_stack,
        use_compound_actions=use_compound_actions,
        doom_skill=doom_skill,
        num_bots=num_bots,
    )
    # Walk down wrappers to grab raw RGB frames from DoomEnv.render().
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    def _frame() -> np.ndarray:
        # Gymnasium's ``render`` signature is ``RenderFrame | list | None``;
        # DoomEnv.render always produces a single HWC ndarray, so narrow.
        frame: Any = base_env.render()
        assert isinstance(frame, np.ndarray), (
            f"Expected ndarray frame from DoomEnv.render, got {type(frame).__name__}"
        )
        return frame

    def _write_clip(frames: list[np.ndarray], path: Path) -> Path:
        imgs = cast(Any, frames)
        try:
            imageio.mimsave(path, imgs, fps=fps)
            return path
        except Exception as exc:  # noqa: BLE001
            fallback = path.with_suffix(".gif")
            imageio.mimsave(fallback, imgs, fps=fps, loop=0)
            print(
                f"[video] mp4 encode failed ({exc}); wrote GIF fallback: {fallback}"
            )
            return fallback

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Recurrent policies (RecurrentPPO) thread an LSTM hidden state across
    # ``predict`` calls and need an ``episode_start`` flag so they can zero
    # out the state at episode boundaries. SB3's ``predict`` API accepts these
    # arguments on every policy — they're just no-ops for non-recurrent ones —
    # so we always pass them and keep one code path.
    episode_stats: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    for ep in range(max(n_episodes, 1)):
        frames: list[np.ndarray] = []
        obs, _ = env.reset()
        frames.append(_frame())
        done = False
        step = 0
        total_reward = 0.0
        last_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        lstm_states: Any = None
        episode_start = np.array([True])
        while not done and step < max_steps:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            obs, reward, terminated, truncated, last_info = env.step(int(action))
            total_reward += float(reward)
            frames.append(_frame())
            done = terminated or truncated
            episode_start = np.array([False])
            step += 1
        # ``DoomEnv`` labels every episode it ends — including a scenario
        # timeout, which it reports as ``truncated`` — so prefer its reason
        # whenever the env is what stopped us. ``"truncated"`` is reserved for
        # the recorder hitting its own ``max_steps`` cap, which is a different
        # thing and worth telling apart in the episode JSON.
        if terminated or truncated:
            termination = str(last_info.get("termination_reason", "unknown"))
        elif step >= max_steps:
            termination = "truncated"
        else:
            termination = "unknown"
        clip_path = out_path.with_name(f"{out_path.stem}_ep{ep + 1}{out_path.suffix}")
        written = _write_clip(frames, clip_path)
        video_paths.append(written)
        # Per-episode combat/exploration metrics come straight from the
        # terminal-step info dict (populated by ``DoomEnv.step``); default
        # to 0 when the episode got truncated before termination so the
        # JSON schema stays stable across rollouts.
        entry: dict[str, Any] = {
            "episode": ep + 1,
            "video": written.name,
            "reward": round(total_reward, 4),
            "length_steps": step,
            "termination": termination,
        }
        for key in EPISODE_METRIC_KEYS:
            entry[key] = int(last_info.get(key) or 0) if terminated else 0
        episode_stats.append(entry)
    env.close()

    # Single JSON summary describing every per-episode clip. Kept alongside
    # the videos so downstream tooling can map clip -> reward/termination
    # without re-running the model. ``env_config`` records the exact env
    # settings the clips were recorded under — most importantly ``doom_skill``
    # and ``num_bots``, which must match the eval env for the reported
    # rewards/terminations to agree with ``stage_summary.txt``.
    stats_path = out_path.with_name(out_path.stem + "_episodes.json")
    stats_path.write_text(
        json.dumps(
            {
                "fps": fps,
                "env_config": {
                    "scenario": scenario,
                    "doom_skill": doom_skill,
                    "num_bots": num_bots,
                    "resize_shape": list(resize_shape),
                    "frame_skip": frame_skip,
                    "num_stack": num_stack,
                    "use_compound_actions": use_compound_actions,
                },
                "episodes": episode_stats,
            },
            indent=4,
        )
        + "\n",
    )
    return video_paths


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------


def train_sb3(
    *,
    algo: str,
    scenario: str,
    run_dir: Path,
    hyperparams: dict[str, Any],
    seed: int,
    total_timesteps: int,
    n_envs: int = 1,
    eval_freq: int = 10_000,
    eval_episodes: int = 10,
    checkpoint_freq: int = 50_000,
    record_video: bool = True,
    video_fps: int = 20,
    video_episodes: int = 5,
    device: str | torch.device | None = None,
    verbose: int = 0,
    on_complete: Callable[[Path], None] | None = None,
    doom_skill: int | None = None,
    num_bots: int = 0,
    resize_shape: tuple[int, int] = (84, 84),
    frame_skip: int = 4,
    num_stack: int = 4,
    use_compound_actions: bool = True,
    policy_kwargs: dict[str, Any] | None = None,
    curriculum: dict[str, Any] | list[CurriculumStage] | None = None,
    extra_callbacks: list[BaseCallback] | None = None,
    early_stop_patience: int | None = None,
    early_stop_min_evals: int = 10,
) -> dict[str, Any]:
    """Train a single SB3 model end-to-end and write *all* artifacts now.

    On return, ``run_dir`` contains:
    * ``checkpoints/final.zip`` + ``checkpoints/best/`` (if eval ran)
    * ``metrics/progress.csv`` + ``metrics/training.npz`` + ``metrics/evaluations.npz``
    * ``figures/learning_curves.png`` + ``figures/eval_performance.png``
    * ``media/<algo>_<scenario>_ep<N>.mp4`` (one per playthrough, with
      ``.gif`` fallback) plus ``media/<algo>_<scenario>_episodes.json``
    * ``tensorboard/`` event files
    * ``stage_summary.txt`` (via ``scripts.generate_summary``)
    * ``config.json`` with ``status=completed``, wall-time, FPS, GPU info

    The function is **self-contained per scenario**: callers training multiple
    scenarios in a loop get a fully-written artifact set for scenario N before
    scenario N+1 starts, which means Colab preemption or an error on scenario
    N+1 still leaves scenario N shippable.

    Parameters
    ----------
    verbose :
        Forwarded to the SB3 model and its callbacks. Defaults to ``0``
        (silent) — set to ``1`` for rollout/eval logs, ``2`` for debug.
    on_complete :
        Optional hook invoked with ``run_dir`` after all artifacts are written
        (e.g. to display the video in a Jupyter cell).
    extra_callbacks :
        Additional SB3 callbacks appended after the built-in eval /
        checkpoint / termination / curriculum ones. Useful for run-specific
        instrumentation without forking this function.
    early_stop_patience :
        Stop once this many consecutive evals pass without a new best eval
        reward. ``None`` (default) trains the full budget. Patience is
        measured within a curriculum rung — a promotion resets it, since a
        harder rung legitimately scores lower. The best checkpoint is saved by
        ``EvalCallback`` regardless, so stopping early costs nothing but the
        steps it skips.
    early_stop_min_evals :
        Minimum evals on a rung before ``early_stop_patience`` can fire, so a
        slow start is not mistaken for a plateau.
    curriculum :
        Optional skill-curriculum specification. Accepts either a raw YAML
        dict (``{"stages": [...], "min_evals_between_promotions": ...}``) or
        a pre-parsed list of :class:`rl_doom.curriculum.CurriculumStage`.
        When supplied, the curriculum's first-stage skill overrides
        ``doom_skill`` at training start and a
        :class:`~rl_doom.curriculum.SkillCurriculumCallback` is appended to
        the callback list.
    """
    run_dir = Path(run_dir)
    tb_dir = run_dir / "tensorboard"
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    media_dir = run_dir / "media"
    ckpt_dir = run_dir / "checkpoints"
    for d in (tb_dir, metrics_dir, figures_dir, media_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- curriculum (parse early so the first stage can override skill) -
    curriculum_stages: list[CurriculumStage] | None
    curriculum_kwargs: dict[str, Any] = {}
    if curriculum is None:
        curriculum_stages = None
    elif isinstance(curriculum, list):
        curriculum_stages = curriculum
    else:
        curriculum_stages = parse_curriculum_config(curriculum)
        curriculum_kwargs = {
            k: curriculum[k]
            for k in ("min_evals_between_promotions", "sync_eval_env")
            if k in curriculum
        }
    if curriculum_stages:
        doom_skill, num_bots = apply_stage0_env_settings(
            curriculum_stages[0], doom_skill=doom_skill, num_bots=num_bots,
        )

    # --- envs ---------------------------------------------------------
    from rl_doom.env import make_sb3_env

    monitor_dir = metrics_dir / "monitor"
    vec_env = make_sb3_env(
        scenario,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(monitor_dir),
        resize_shape=resize_shape,
        frame_skip=frame_skip,
        num_stack=num_stack,
        use_compound_actions=use_compound_actions,
        doom_skill=doom_skill,
        num_bots=num_bots,
    )
    # Eval env is kept small (n_envs=1) and offset-seeded so eval rollouts
    # aren't duplicates of training rollouts.
    eval_env = make_sb3_env(
        scenario,
        n_envs=1,
        seed=seed + 10_000,
        monitor_dir=None,
        resize_shape=resize_shape,
        frame_skip=frame_skip,
        num_stack=num_stack,
        use_compound_actions=use_compound_actions,
        doom_skill=doom_skill,
        num_bots=num_bots,
    )

    # Reward normalization for PPO: ViZDoom scenarios (especially
    # deadly_corridor) have reward spikes in the thousands that dwarf the
    # entropy bonus and inflate value loss. Wrapping the training env in
    # VecNormalize with norm_reward=True scales the discounted returns by a
    # running std estimate so the advantage/entropy scales stay comparable.
    # DQN is left alone — Q-learning's Bellman target assumes un-normalized
    # rewards, and SB3's VecNormalize would bias the replay buffer.
    # The eval env is wrapped (training=False, norm_reward=False) only to
    # satisfy SB3's ``sync_envs_normalization`` invariant; its reported
    # rewards remain in the original raw scale.
    if algo.lower() in _PPO_FAMILY:
        vec_env = VecNormalize(
            vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0,
        )
        eval_env = VecNormalize(
            eval_env, norm_obs=False, norm_reward=False, training=False,
        )

    # --- model + logger ----------------------------------------------
    model = _build_model(
        algo,
        vec_env,
        hyperparams=hyperparams,
        tensorboard_log=tb_dir,
        device=device,
        seed=seed,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
    )
    # Force CSV + TensorBoard output so we can rebuild plots post-hoc. SB3's
    # ``configure`` writes both formats into the same folder, so we route
    # everything into ``tensorboard/`` and read ``progress.csv`` back from
    # there when building learning-curve figures.
    model.set_logger(configure(str(tb_dir), ["stdout", "csv", "tensorboard"]))

    termination_tracker = TerminationTracker()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir / "best"),
        log_path=str(metrics_dir),
        eval_freq=max(eval_freq // max(n_envs, 1), 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    callbacks: list[BaseCallback] = [
        eval_cb,
        CheckpointCallback(
            save_freq=max(checkpoint_freq // max(n_envs, 1), 1),
            save_path=str(ckpt_dir),
            name_prefix="step",
            verbose=verbose,
        ),
        termination_tracker,
    ]
    # Registered AFTER the eval callback so ``evaluations_timesteps`` /
    # ``last_mean_reward`` are populated before the curriculum reads them.
    curriculum_cb: SkillCurriculumCallback | None = None
    if curriculum_stages:
        curriculum_cb = SkillCurriculumCallback(
            eval_cb, curriculum_stages, verbose=verbose, **curriculum_kwargs,
        )
        callbacks.append(curriculum_cb)
    # After the curriculum callback so that on any given step it sees the
    # promotion that step produced, rather than reacting one eval late.
    early_stop_cb: StopOnEvalPlateau | None = None
    if early_stop_patience:
        early_stop_cb = StopOnEvalPlateau(
            eval_cb,
            patience=early_stop_patience,
            min_evals=early_stop_min_evals,
            curriculum_cb=curriculum_cb,
            verbose=1,
        )
        callbacks.append(early_stop_cb)
    # Appended last so a caller's callback sees the state the built-in ones
    # have already updated.
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # --- train --------------------------------------------------------
    #
    # Everything that makes a run useful — the model, metrics, figures, video,
    # config.json, stage_summary.txt — is written *after* learn() returns, so
    # an unguarded exception here discarded all of it. On a 2.5M-step Colab run
    # that meant a failure at step 2.4M left nothing but periodic checkpoint
    # zips that nothing in the repo reads.
    #
    # On failure we still save the weights and write whatever metrics exist,
    # mark the run failed, and re-raise. A partial run that can be inspected
    # beats a clean traceback over an empty directory.
    t_start = time.time()
    training_error: BaseException | None = None
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
    except BaseException as exc:  # noqa: BLE001 — re-raised below
        # BaseException, not Exception: KeyboardInterrupt is the *expected* way
        # a human stops an overlong run, and losing the run to Ctrl-C is the
        # most annoying version of this bug.
        training_error = exc
        print(
            f"[train_sb3] training raised {type(exc).__name__}: {exc}\n"
            f"[train_sb3] salvaging artifacts into {run_dir} before re-raising.",
        )
    finally:
        wall_time = time.time() - t_start
        # Steps actually completed, which is not total_timesteps on failure.
        steps_done = int(getattr(model, "num_timesteps", 0))
        fps = steps_done / wall_time if wall_time > 0 else 0.0

        # --- persist --------------------------------------------------
        # Saving runs on both paths: on failure these are the only weights
        # that will exist.
        try:
            model.save(str(ckpt_dir / "final.zip"))
            # Save VecNormalize running stats so a future resumed run can pick
            # up with the same reward-scale estimate (inference/video doesn't
            # need it since we only normalise rewards, not observations).
            if isinstance(vec_env, VecNormalize):
                vec_env.save(str(ckpt_dir / "vec_normalize.pkl"))
        except Exception as save_exc:  # noqa: BLE001 — best-effort salvage
            print(f"[train_sb3] could not save final checkpoint: {save_exc}")

        # Teardown belongs in the finally: ViZDoom spawns a native process per
        # env, and on the old success-only path a failed run leaked every one
        # of them.
        for env_name, env_to_close in (("train", vec_env), ("eval", eval_env)):
            try:
                env_to_close.close()
            except Exception as close_exc:  # noqa: BLE001 — best-effort
                # Was `except Exception: pass` around both closes together, so
                # a failure closing the train env also skipped the eval env.
                print(f"[train_sb3] error closing {env_name} env: {close_exc}")

    if training_error is not None:
        _write_termination_report(termination_tracker, metrics_dir)
        mark_run_status(
            run_dir,
            status="failed",
            error=f"{type(training_error).__name__}: {training_error}",
            completed_timesteps=steps_done,
            requested_timesteps=total_timesteps,
            wall_time_seconds=wall_time,
        )
        raise training_error

    progress = _load_progress_csv(tb_dir / "progress.csv")
    episodes = _collect_episode_stats(monitor_dir)
    evaluations = _load_evaluations(run_dir)
    _write_termination_report(termination_tracker, metrics_dir)
    if curriculum_cb is not None:
        (metrics_dir / "curriculum.json").write_text(
            json.dumps(
                {
                    "stages": [
                        {
                            "skill": s.skill,
                            "num_bots": s.num_bots,
                            "promote_at": s.promote_at,
                        }
                        for s in curriculum_stages or []
                    ],
                    "promotions": curriculum_cb.promotions,
                    "final_stage_index": curriculum_cb.current_stage_index,
                    "final_skill": curriculum_cb.current_skill,
                    "final_num_bots": curriculum_cb.current_num_bots,
                },
                indent=2,
            ),
        )

    per_episode_metrics = {
        f"episode_{key}": episodes.get(key, np.array([]))
        for key in EPISODE_METRIC_KEYS
    }
    savez_kwargs: dict[str, Any] = {
        "episode_rewards": episodes.get("rewards", np.array([])),
        "episode_lengths": episodes.get("lengths", np.array([])),
        # SB3 EvalCallback stores a 2-D (n_evals, n_eval_episodes) results
        # matrix; convert to the legacy 5-column (step, mean_r, std_r, mean_l,
        # std_l) layout that ``generate_summary.py`` consumes.
        "eval_rewards": _legacy_eval_log(evaluations),
        "wall_time_seconds": wall_time,
        "fps": fps,
        # Steps actually run, not requested: they differ whenever training
        # stops early, and reporting the request would overstate both this and
        # the FPS derived from it.
        "total_env_steps": steps_done,
        **per_episode_metrics,
    }
    np.savez(metrics_dir / "training.npz", **savez_kwargs)

    learning_curves_path = figures_dir / "learning_curves.png"
    _plot_learning_curves(
        algo, scenario, progress, episodes, evaluations, out_path=learning_curves_path,
    )

    video_paths: list[Path] = []
    if record_video:
        out_name = f"{algo.lower()}_{scenario}.mp4"
        # Record the gameplay clip from the best eval checkpoint when
        # available — that's the weights the notebooks showcase — and only
        # fall back to the in-memory (final-step) model if eval didn't run.
        best_ckpt = ckpt_dir / "best" / "best_model.zip"
        if best_ckpt.exists():
            video_cls = resolve_algo_class(algo)
            video_model: BaseAlgorithm = video_cls.load(str(best_ckpt), device=device)
        else:
            video_model = model
        # Match the video env to the eval env. When a curriculum ran, the
        # relevant difficulty is the highest rung the agent actually reached
        # (``current_skill``/``current_num_bots``) — that's what
        # ``EvalCallback`` was measuring against at the end of training, so
        # the recorded clip should use the same setting. Without a curriculum
        # we fall back to the fixed ``doom_skill``/``num_bots`` the caller
        # passed in.
        if curriculum_cb is not None:
            video_doom_skill = curriculum_cb.current_skill
            video_num_bots = curriculum_cb.current_num_bots or 0
        else:
            video_doom_skill = doom_skill
            video_num_bots = num_bots
        video_paths = _record_video(
            video_model,
            scenario,
            media_dir / out_name,
            fps=video_fps,
            n_episodes=video_episodes,
            resize_shape=resize_shape,
            frame_skip=frame_skip,
            num_stack=num_stack,
            use_compound_actions=use_compound_actions,
            doom_skill=video_doom_skill,
            num_bots=video_num_bots,
        )

    # Finalise config.json + latest pointer, then write the per-run summary.
    # config.json holds identity + hardware + hyperparameters only; all
    # training *results* (eval rewards, success rate, wall-time, FPS) live in
    # metrics/training.npz and are rendered into stage_summary.txt.
    final_eval_mean = float(_legacy_eval_log(evaluations)[-1][1]) \
        if evaluations.get("results") is not None and evaluations["results"].size \
        else None
    eval_log = _legacy_eval_log(evaluations)
    best_eval_mean = (
        float(eval_log[:, 1].max()) if eval_log.shape[0] else None
    )
    # Success rate = fraction of training episodes that ended with the agent
    # reaching the scenario's goal (e.g. the vest in deadly_corridor).
    total_eps = sum(termination_tracker.counts.values())
    goal_count = termination_tracker.counts.get("goal_reached", 0)
    success_rate = (goal_count / total_eps) if total_eps else None
    mark_run_status(
        run_dir,
        status="completed",
        stopped_early=bool(early_stop_cb and early_stop_cb.stopped_early),
        stopped_at_step=early_stop_cb.stopped_at_step if early_stop_cb else None,
        completed_timesteps=steps_done,
    )
    update_latest_symlink(scenario, algo.lower(), run_dir)

    # Write stage_summary.txt for this run, in-process (no subprocess).
    from rl_doom.summary import generate_run_summary
    (run_dir / "stage_summary.txt").write_text(generate_run_summary(run_dir))

    result = {
        "run_dir": run_dir,
        "wall_time_seconds": wall_time,
        "fps": fps,
        "mean_eval_reward": final_eval_mean,
        "best_eval_reward": best_eval_mean,
        "success_rate": success_rate,
        "video_paths": video_paths,
        "learning_curves_path": learning_curves_path,
        "curriculum_final_skill": (
            curriculum_cb.current_skill if curriculum_cb else None
        ),
        "curriculum_final_num_bots": (
            curriculum_cb.current_num_bots if curriculum_cb else None
        ),
        "curriculum_promotions": (
            curriculum_cb.promotions if curriculum_cb else None
        ),
        "stopped_early": bool(early_stop_cb and early_stop_cb.stopped_early),
        "stopped_at_step": early_stop_cb.stopped_at_step if early_stop_cb else None,
    }

    if on_complete is not None:
        try:
            on_complete(run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[train_sb3] on_complete hook raised: {exc}")

    return result


# ---------------------------------------------------------------------------
# Monitor helpers
# ---------------------------------------------------------------------------


def _collect_episode_stats(monitor_dir: Path) -> dict[str, np.ndarray]:
    """Read all ``monitor_*.monitor.csv`` files and concatenate episode stats.

    SB3's Monitor writes a single-line JSON header followed by a CSV
    (``r,l,t``), plus any extra columns configured via ``info_keywords``
    (we pass the full :data:`EPISODE_METRIC_KEYS` tuple so per-episode
    combat/exploration counts are captured alongside reward/length).
    We append across workers in wall-clock order. Older monitor CSVs
    that predate a metric column are handled by defaulting the missing
    cells to 0.
    """
    monitor_dir = Path(monitor_dir)
    empty: dict[str, np.ndarray] = {
        "rewards": np.array([]),
        "lengths": np.array([]),
    }
    for key in EPISODE_METRIC_KEYS:
        empty[key] = np.array([])
    files = sorted(monitor_dir.glob("monitor_*.monitor.csv"))
    if not files:
        return empty
    rewards: list[float] = []
    lengths: list[int] = []
    timestamps: list[float] = []
    metrics: dict[str, list[int]] = {key: [] for key in EPISODE_METRIC_KEYS}
    for path in files:
        with path.open() as f:
            # Skip the JSON header line.
            header = f.readline()
            if not header.startswith("#"):
                f.seek(0)
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rewards.append(float(row["r"]))
                    lengths.append(int(row["l"]))
                    timestamps.append(float(row["t"]))
                except (KeyError, TypeError, ValueError):
                    continue
                for key in EPISODE_METRIC_KEYS:
                    raw = row.get(key, "")
                    try:
                        metrics[key].append(
                            int(float(raw)) if raw not in ("", None) else 0,
                        )
                    except (TypeError, ValueError):
                        metrics[key].append(0)
    if not timestamps:
        return empty
    order: np.ndarray = np.argsort(timestamps)
    out: dict[str, np.ndarray] = {
        "rewards": np.asarray(rewards, dtype=np.float32)[order],
        "lengths": np.asarray(lengths, dtype=np.int64)[order],
    }
    for key, values in metrics.items():
        out[key] = np.asarray(values, dtype=np.int64)[order]
    return out


def _legacy_eval_log(evaluations: dict[str, np.ndarray]) -> np.ndarray:
    """Convert SB3 ``evaluations.npz`` to ``(step, mean_r, std_r, mean_l, std_l)``.

    ``generate_summary.py`` expects this 5-column layout in ``eval_rewards``;
    keeping the same shape avoids having to fork that script for SB3.
    """
    ts = evaluations.get("timesteps")
    results = evaluations.get("results")
    lengths = evaluations.get("ep_lengths")
    if ts is None or results is None or ts.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    mean_r = results.mean(axis=1)
    std_r = results.std(axis=1)
    if lengths is not None and lengths.shape == results.shape:
        mean_l = lengths.mean(axis=1)
        std_l = lengths.std(axis=1)
    else:
        mean_l = np.zeros_like(mean_r)
        std_l = np.zeros_like(std_r)
    return np.stack([ts, mean_r, std_r, mean_l, std_l], axis=1).astype(np.float32)
