"""DreamerV3 agent driver wrapping the NM512/dreamerv3-torch port.

The upstream port has no ``setup.py`` / ``pyproject.toml`` and so cannot be
installed via pip. The notebook ``git clone``s it and adds the checkout to
``sys.path``; this module then imports the four port files we actually need
(``models``, ``networks``, ``tools``, ``exploration``) lazily via
:func:`_import_port` and re-implements the thin ``Dreamer`` ``nn.Module``
driver inline so we avoid pulling in upstream's ``envs.wrappers`` (which
imports the legacy ``gym`` package) and ``ruamel.yaml`` deps.

See ``DREAMER_PLAN.md`` for the integration plan.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

# Single source of truth for the upstream commit pin. The notebook reads this
# constant when ``git clone``-ing so the cloned tree and the imported code can
# never disagree.
UPSTREAM_PIN: str = "6ef8646d807cd10ce0c88e10a7e943211e7fc44c"

# Names of the four upstream modules we actually consume. Kept as a tuple so
# :func:`_import_port` can validate the checkout before mutating ``sys.path``.
_REQUIRED_PORT_FILES: tuple[str, ...] = (
    "models.py",
    "networks.py",
    "tools.py",
    "exploration.py",
    "configs.yaml",
)


def _import_port(port_path: str | Path) -> dict[str, ModuleType]:
    """Import the four upstream modules we need, returning them in a dict.

    ``port_path`` is the directory containing the cloned ``dreamerv3-torch``
    repo (it should contain ``models.py``, ``tools.py``, etc.). The path is
    inserted at the front of ``sys.path`` so the upstream's intra-package
    imports (``import models`` etc.) resolve to the clone, not to anything
    else that happens to share those names.

    Raises ``RuntimeError`` with a friendly message — including the pinned
    commit SHA — when the checkout is missing or incomplete.
    """
    port_path = Path(port_path).expanduser().resolve()
    missing = [
        name for name in _REQUIRED_PORT_FILES if not (port_path / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"DreamerV3 port not found at {port_path!s} "
            f"(missing: {', '.join(missing)}). Clone the upstream repo first:\n"
            f"  git clone https://github.com/NM512/dreamerv3-torch {port_path}\n"
            f"  git -C {port_path} checkout {UPSTREAM_PIN}\n"
            f"and then pass its path as ``port_path``.",
        )
    if str(port_path) not in sys.path:
        sys.path.insert(0, str(port_path))
    import importlib

    return {name: importlib.import_module(name) for name in (
        "models", "networks", "tools", "exploration",
    )}


def _load_port_configs(port_path: str | Path) -> dict[str, Any]:
    """Read the upstream's ``configs.yaml`` via PyYAML (no ruamel dep).

    The file ships verbatim with the port and contains ``defaults`` plus the
    suite-specific presets (``atari100k``, ``crafter``, etc.). We load it with
    :func:`yaml.safe_load` rather than pulling in ``ruamel.yaml`` like the
    upstream CLI does — the file is plain YAML and round-trips fine.
    """
    import yaml

    text = (Path(port_path).expanduser() / "configs.yaml").read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected mapping at top-level of configs.yaml, got {type(data).__name__}",
        )
    return data


def _recursive_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    """In-place ``base <- update`` deep merge, matching upstream's helper."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value


def _to_namespace(d: dict[str, Any]) -> SimpleNamespace:
    """Convert the merged config dict to ``SimpleNamespace`` (attr access)."""
    return SimpleNamespace(**d)


# ---------------------------------------------------------------------------
# Embedded ``_Dreamer`` nn.Module (re-implemented from the port's dreamer.py)
# ---------------------------------------------------------------------------
#
# We deliberately do **not** import the upstream's ``dreamer`` module: it
# imports ``envs.wrappers`` (which pulls in legacy ``gym``) and ``ruamel.yaml``
# at module load time. Since the actual ``Dreamer`` class is just glue around
# ``models.WorldModel`` + ``models.ImagBehavior`` + the exploration heads — all
# of which live in modules with clean dependencies — we re-embed the class
# verbatim here. Locked to UPSTREAM_PIN; tracked manually if we ever bump.


def _build_dreamer_class(port: dict[str, ModuleType]) -> type:
    """Construct the embedded ``_Dreamer`` nn.Module class once the port is imported.

    The class needs ``models``, ``tools``, and ``exploration`` references at
    class-definition time, which we don't have until ``_import_port`` runs.
    Building it inside a factory keeps the import order honest (and means the
    module stays importable on machines without the port checked out).
    """
    import numpy as np
    import torch
    from torch import nn

    models_mod = port["models"]
    tools_mod = port["tools"]
    expl_mod = port["exploration"]

    def _to_np(x: Any) -> Any:
        return x.detach().cpu().numpy()

    class _Dreamer(nn.Module):  # noqa: D401 — verbatim from upstream
        """Embedded copy of ``dreamerv3-torch.dreamer.Dreamer``.

        Pinned to UPSTREAM_PIN. Only edits vs upstream:
        * imports come from the captured ``port`` dict (rather than module-level
          ``import models`` etc.) so this works without modifying ``sys.path``
          a second time;
        * ``_to_np`` is local rather than module-level.
        """

        def __init__(self, obs_space, act_space, config, logger, dataset):
            super().__init__()
            self._config = config
            self._logger = logger
            self._should_log = tools_mod.Every(config.log_every)
            batch_steps = config.batch_size * config.batch_length
            self._should_train = tools_mod.Every(batch_steps / config.train_ratio)
            self._should_pretrain = tools_mod.Once()
            self._should_reset = tools_mod.Every(config.reset_every)
            self._should_expl = tools_mod.Until(
                int(config.expl_until / config.action_repeat),
            )
            self._metrics: dict[str, Any] = {}
            self._step = logger.step // config.action_repeat
            self._update_count = 0
            self._dataset = dataset
            self._wm = models_mod.WorldModel(obs_space, act_space, self._step, config)
            self._task_behavior = models_mod.ImagBehavior(config, self._wm)
            import os as _os
            if config.compile and _os.name != "nt":
                self._wm = torch.compile(self._wm)
                self._task_behavior = torch.compile(self._task_behavior)
            reward = lambda f, s, a: self._wm.heads["reward"](f).mean()  # noqa: E731
            self._expl_behavior = dict(
                greedy=lambda: self._task_behavior,
                random=lambda: expl_mod.Random(config, act_space),
                plan2explore=lambda: expl_mod.Plan2Explore(config, self._wm, reward),
            )[config.expl_behavior]().to(self._config.device)

        def __call__(self, obs, reset, state=None, training=True):
            step = self._step
            if training:
                steps = (
                    self._config.pretrain
                    if self._should_pretrain()
                    else self._should_train(step)
                )
                for _ in range(steps):
                    self._train(next(self._dataset))
                    self._update_count += 1
                    self._metrics["update_count"] = self._update_count
                if self._should_log(step):
                    for name, values in self._metrics.items():
                        self._logger.scalar(name, float(np.mean(values)))
                        self._metrics[name] = []
                    if self._config.video_pred_log:
                        openl = self._wm.video_pred(next(self._dataset))
                        self._logger.video("train_openl", _to_np(openl))
                    self._logger.write(fps=True)

            policy_output, state = self._policy(obs, state, training)

            if training:
                self._step += len(reset)
                self._logger.step = self._config.action_repeat * self._step
            return policy_output, state

        def _policy(self, obs, state, training):
            if state is None:
                latent = action = None
            else:
                latent, action = state
            obs = self._wm.preprocess(obs)
            embed = self._wm.encoder(obs)
            latent, _ = self._wm.dynamics.obs_step(
                latent, action, embed, obs["is_first"],
            )
            if self._config.eval_state_mean:
                latent["stoch"] = latent["mean"]
            feat = self._wm.dynamics.get_feat(latent)
            if not training:
                actor = self._task_behavior.actor(feat)
                action = actor.mode()
            elif self._should_expl(self._step):
                actor = self._expl_behavior.actor(feat)
                action = actor.sample()
            else:
                actor = self._task_behavior.actor(feat)
                action = actor.sample()
            logprob = actor.log_prob(action)
            latent = {k: v.detach() for k, v in latent.items()}
            action = action.detach()
            if self._config.actor["dist"] == "onehot_gumble":
                # Upstream writes ``torch.one_hot(...)`` here, which is a
                # typo — the real API is ``torch.nn.functional.one_hot``,
                # so the upstream code would crash at runtime if this
                # branch ever fired. We never configure
                # ``actor.dist == "onehot_gumble"`` (atari100k preset uses
                # plain ``"onehot"``), so this is dead code in practice;
                # we use the correct API anyway in case a downstream user
                # opts into Gumbel-softmax via YAML.
                action = torch.nn.functional.one_hot(
                    torch.argmax(action, dim=-1), self._config.num_actions,
                )
            policy_output = {"action": action, "logprob": logprob}
            state = (latent, action)
            return policy_output, state

        def _train(self, data):
            metrics: dict[str, Any] = {}
            post, context, mets = self._wm._train(data)
            metrics.update(mets)
            start = post
            reward = lambda f, s, a: self._wm.heads["reward"](  # noqa: E731
                self._wm.dynamics.get_feat(s),
            ).mode()
            metrics.update(self._task_behavior._train(start, reward)[-1])
            if self._config.expl_behavior != "greedy":
                mets = self._expl_behavior.train(start, context, data)[-1]
                metrics.update({"expl_" + key: value for key, value in mets.items()})
            for name, value in metrics.items():
                if name not in self._metrics:
                    self._metrics[name] = [value]
                else:
                    self._metrics[name].append(value)

    return _Dreamer


class _Damy:
    """Re-implementation of upstream's ``parallel.Damy`` shim.

    ``tools.simulate`` calls ``env.step(a)`` / ``env.reset()`` and treats the
    return value as a *thunk* it then invokes (so a parallel-process backend
    can do the work async). For a single in-process env we just wrap the call
    so ``simulate`` can call the result with no args.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any) -> Any:
        return lambda: self._env.step(action)

    def reset(self) -> Any:
        return lambda: self._env.reset()


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


# Doom-specific config overrides split into two layers:
#
# * :data:`_DOOM_HARD_OVERRIDES` — locked by the integration architecture; user
#   YAML cannot reach these without breaking the adapter (e.g. ``envs > 1``
#   would race against our single in-process ViZDoom process).
# * :data:`_DOOM_DEFAULT_OVERRIDES` — Doom-friendly defaults that user YAML
#   *can* override. Applied *before* the user's ``hyperparams`` block so a YAML
#   knob like ``video_pred_log: true`` (to inspect imagined rollouts) or
#   ``expl_behavior: plan2explore`` (for sparse-reward scenarios like
#   ``deadly_corridor``) actually takes effect.
#
# Kept as module-level constants so tests can introspect them.
_DOOM_HARD_OVERRIDES: dict[str, Any] = {
    "task": "doom_custom",          # we don't use upstream's make_env, so the
                                    # task suite tag is informational only.
    "envs": 1,                      # single in-process env (see DREAMER_PLAN.md).
    "parallel": False,              # never spawn upstream's Parallel workers.
    "offline_traindir": "",         # online learning only.
    "offline_evaldir": "",
}

_DOOM_DEFAULT_OVERRIDES: dict[str, Any] = {
    "compile": False,               # ``torch.compile`` is brittle on Colab L4
                                    # + our env's small obs; default off so the
                                    # PoC trains predictably. Override in YAML
                                    # if the environment is known-good.
    "video_pred_log": False,        # off by default — saves VRAM and speeds up
                                    # eval on Colab. Flip on in YAML to inspect
                                    # the world model's imagined rollouts.
    "expl_behavior": "greedy",      # match atari100k preset. Override to
                                    # ``plan2explore`` for sparse-reward
                                    # scenarios (see DREAMER_PLAN.md §2).
}

# Back-compat alias: the union of both override layers, mirroring the
# pre-split behaviour for callers (mostly tests) that want to inspect "every
# Doom-side config knob the wrapper sets". Reads only — mutating it doesn't
# affect ``_build_config``.
_DOOM_FIXED_OVERRIDES: dict[str, Any] = {
    **_DOOM_DEFAULT_OVERRIDES,
    **_DOOM_HARD_OVERRIDES,
}


def _build_config(
    *,
    port_path: str | Path,
    scenario: str,
    env_cfg: dict[str, Any],
    hyperparams: dict[str, Any],
    eval_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    seed: int,
    device: str,
    logdir: str | Path,
) -> SimpleNamespace:
    """Assemble the SimpleNamespace config the embedded ``_Dreamer`` consumes.

    Merge order (later wins):
        upstream ``configs.yaml:defaults``
        upstream ``configs.yaml:<preset>``           (default ``atari100k``)
        Doom-specific *defaults*                       (:data:`_DOOM_DEFAULT_OVERRIDES`)
        our YAML's ``hyperparams`` block               (model / training knobs)
        Doom-specific *hard locks*                     (:data:`_DOOM_HARD_OVERRIDES`)
        per-call overrides                             (logdir, seed, device, total_timesteps, …)

    The returned object has attribute access (``cfg.batch_size``) the same way
    upstream's argparse-built config does, so the embedded ``_Dreamer`` and the
    ``models``/``tools`` modules consume it without modification.
    """
    cfg_yaml = _load_port_configs(port_path)
    merged: dict[str, Any] = {}
    _recursive_update(merged, cfg_yaml.get("defaults", {}))

    preset = hyperparams.get("preset", "atari100k")
    if preset not in cfg_yaml:
        available = sorted(k for k in cfg_yaml if k != "defaults")
        raise ValueError(
            f"Unknown dreamer preset {preset!r}; available presets in "
            f"upstream configs.yaml: {available}",
        )
    _recursive_update(merged, cfg_yaml[preset])

    # Doom defaults *before* user YAML so a knob like ``video_pred_log: true``
    # in the user's YAML can flip the default off->on.
    _recursive_update(merged, _DOOM_DEFAULT_OVERRIDES)

    # YAML hyperparams are the user's tuning surface. Strip ``preset`` since
    # it's a meta-key consumed above, not a Dreamer config field.
    user_hp = {k: v for k, v in hyperparams.items() if k != "preset"}
    _recursive_update(merged, user_hp)

    # Hard locks last so YAML can't accidentally re-enable ``parallel`` etc.
    _recursive_update(merged, _DOOM_HARD_OVERRIDES)

    # Image size: take from env_cfg.resize_shape, default 64x64 (matches the
    # atari100k preset). Upstream stores ``size`` as a 2-tuple ``(H, W)``.
    resize = env_cfg.get("resize_shape", [64, 64])
    merged["size"] = tuple(resize)
    merged["grayscale"] = bool(env_cfg.get("grayscale", False))
    merged["action_repeat"] = int(env_cfg.get("frame_skip", 4))

    # Per-call wiring: total_timesteps -> ``steps``, paths, seed, device.
    total_timesteps = int(training_cfg.get("total_timesteps", merged.get("steps", 1e6)))
    eval_every = int(eval_cfg.get("eval_freq", merged.get("eval_every", 10000)))
    merged["eval_every"] = eval_every
    # The training loop runs while ``agent._step < cfg.steps + cfg.eval_every``
    # so a final eval pass triggers — that overshoots ``total_timesteps`` by
    # one eval chunk (e.g. budget=50k, eval_every=5k -> ~55k env steps).
    # Subtract eval_every from ``steps`` here so total env interaction
    # (prefill + agent-driven training) lands at ``total_timesteps`` ± one
    # ``eval_every`` chunk — well below the unfixed 10% drift on small budgets.
    # Floor at ``eval_every`` so degenerate ``total_timesteps <= eval_every``
    # configs (smoke tests) still run a complete training cycle.
    merged["steps"] = max(eval_every, total_timesteps - eval_every)
    merged["eval_episode_num"] = int(
        eval_cfg.get("n_episodes", merged.get("eval_episode_num", 5)),
    )
    merged["seed"] = int(seed)
    merged["device"] = str(device)
    logdir = Path(logdir)
    merged["logdir"] = str(logdir)
    merged["traindir"] = str(logdir / "train_eps")
    merged["evaldir"] = str(logdir / "eval_eps")

    # ``time_limit`` is in env steps for the upstream codepath; ViZDoom
    # scenarios already have an in-engine episode timeout, so set a generous
    # ceiling unless the user overrode it. Mirrors atari100k preset (108k).
    merged.setdefault("time_limit", 108000)

    # ``num_actions`` is filled in by ``train_dreamer`` after the env is
    # built — included here as a sentinel so ``models.WorldModel`` doesn't
    # explode if anything tries to read it before that.
    merged.setdefault("num_actions", 0)

    # Tag the scenario for run-artefact provenance. The embedded Dreamer
    # never reads this field; it's purely informational.
    merged["scenario"] = scenario

    return _to_namespace(merged)


# ---------------------------------------------------------------------------
# Helpers — episode log scanning, video recording, plotting
# ---------------------------------------------------------------------------


def _scan_episode_npzs(directory: Path) -> tuple[list[float], list[int]]:
    """Walk ``directory/*.npz`` and return per-episode reward sums + lengths.

    ``tools.simulate`` writes one ``.npz`` per finished episode containing
    arrays for ``reward``, ``image``, ``action``, etc. Length is
    ``len(reward) - 1`` (the first row is the reset transition with reward 0).
    """
    import numpy as np

    rewards: list[float] = []
    lengths: list[int] = []
    if not directory.exists():
        return rewards, lengths
    for path in sorted(directory.glob("*.npz")):
        try:
            with np.load(path) as ep:
                r = ep["reward"]
                rewards.append(float(np.sum(r)))
                lengths.append(int(len(r) - 1))
        except (OSError, KeyError, ValueError):
            continue
    return rewards, lengths


def _read_metrics_jsonl(path: Path) -> dict[str, list[float]]:
    """Parse ``tools.Logger``'s ``metrics.jsonl`` into per-key time series.

    Each line is a JSON object ``{"step": <int>, "<key>": <float>, ...}``.
    Returns ``{"step": [...], "<key>": [...], ...}`` with per-key arrays
    aligned to whichever rows actually contained that key (so metrics with
    different log cadences don't co-mingle).
    """
    import json

    if not path.exists():
        return {}
    series: dict[str, list[tuple[float, float]]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = float(row.get("step", 0))
            for k, v in row.items():
                if k == "step":
                    continue
                try:
                    series.setdefault(k, []).append((step, float(v)))
                except (TypeError, ValueError):
                    continue
    out: dict[str, list[float]] = {}
    for k, pairs in series.items():
        out[f"{k}__step"] = [p[0] for p in pairs]
        out[k] = [p[1] for p in pairs]
    return out


def _synthesize_evaluations_npz(
    eval_history: list[tuple[int, list[float], list[int]]],
    out_path: Path,
) -> None:
    """Write an ``evaluations.npz`` matching SB3 ``EvalCallback``'s schema.

    Each row in ``eval_history`` is ``(timestep, episode_rewards, episode_lengths)``
    from one eval pass. We stack into the ``(timesteps, results, ep_lengths)``
    layout :func:`rl_doom.sb3_utils._plot_eval_performance` and
    :func:`rl_doom.summary.generate_run_summary` consume, so those functions
    work unchanged on dreamer runs.
    """
    import numpy as np

    if not eval_history:
        return
    timesteps = np.asarray([t for t, _r, _l in eval_history], dtype=np.int64)
    n_evals = len(eval_history)
    n_eps = max(len(r) for _t, r, _l in eval_history) or 1
    results = np.zeros((n_evals, n_eps), dtype=np.float32)
    ep_lengths = np.zeros((n_evals, n_eps), dtype=np.int64)
    for i, (_t, r, le) in enumerate(eval_history):
        for j, (rj, lj) in enumerate(zip(r, le)):
            results[i, j] = float(rj)
            ep_lengths[i, j] = int(lj)
    np.savez(
        out_path,
        timesteps=timesteps,
        results=results,
        ep_lengths=ep_lengths,
    )


def _plot_dreamer_curves(
    scenario: str,
    episode_rewards: list[float],
    episode_lengths: list[int],
    metrics_path: Path,
    out_path: Path,
) -> None:
    """Render a 2x3 learning-curves grid analogous to SB3's plotter.

    Reads ``tools.Logger``'s ``metrics.jsonl`` (NOT SB3's progress.csv) so the
    keys are dreamer-specific: ``kl_loss`` / ``model_loss`` / ``actor_ent`` /
    ``value_loss`` / ``return_pred`` / ``image_loss``. Falls back to "not
    logged" placeholders for keys missing from a particular run (e.g. early
    runs that hit a crash before the first log flush).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    series = _read_metrics_jsonl(metrics_path)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ep_r = np.asarray(episode_rewards, dtype=np.float32)
    ep_l = np.asarray(episode_lengths, dtype=np.int64)

    ax = axes[0, 0]
    if ep_r.size:
        ax.plot(ep_r, alpha=0.3, label="Raw")
        if ep_r.size >= 20:
            sm = np.convolve(ep_r, np.ones(20) / 20, mode="valid")
            ax.plot(range(19, 19 + len(sm)), sm, label="MA-20")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No episode data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"Episode Rewards - {scenario}")

    ax = axes[0, 1]
    if ep_l.size:
        ax.plot(ep_l, alpha=0.3, color="green", label="Raw")
        if ep_l.size >= 20:
            sm = np.convolve(ep_l, np.ones(20) / 20, mode="valid")
            ax.plot(range(19, 19 + len(sm)), sm, color="darkgreen", label="MA-20")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Lengths")

    panel_defs = [
        ((0, 2), "model_loss", "Model Loss", "World Model Loss"),
        ((1, 0), "kl_loss", "KL Loss", "Dynamics KL Loss"),
        ((1, 1), "actor_ent", "Actor Entropy", "Policy Entropy"),
        ((1, 2), "value_loss", "Value Loss", "Critic Loss"),
    ]
    for (r, c), key, ylabel, title in panel_defs:
        ax = axes[r, c]
        ys = series.get(key)
        xs = series.get(f"{key}__step")
        if ys and xs and len(ys) == len(xs):
            ax.plot(xs, ys)
        else:
            ax.text(
                0.5, 0.5, f"{key} not logged", ha="center", va="center",
                transform=ax.transAxes, fontsize=9,
            )
        ax.set_xlabel("Env Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _record_dreamer_video(
    agent: Any,
    scenario: str,
    out_path: Path,
    *,
    fps: int,
    n_episodes: int,
    env_cfg: dict[str, Any],
    max_steps: int = 5_000,
) -> list[Path]:
    """Drive a fresh ``DreamerDoomEnv`` greedily and write per-episode mp4s.

    Mirrors the layout :func:`rl_doom.sb3_utils._record_video` produces — one
    mp4 per rollout plus a single ``_episodes.json`` sidecar — so the
    notebook display logic and ``generate_run_summary`` work unchanged.
    """
    import json

    import imageio
    import numpy as np

    from rl_doom.dreamer_env import DreamerDoomEnv
    from rl_doom.scenario_limits import EPISODE_METRIC_KEYS

    env = DreamerDoomEnv(
        scenario=scenario,
        resize_shape=tuple(env_cfg.get("resize_shape", (64, 64))),
        frame_skip=int(env_cfg.get("frame_skip", 4)),
        grayscale=bool(env_cfg.get("grayscale", False)),
        doom_skill=env_cfg.get("doom_skill"),
        num_bots=int(env_cfg.get("num_bots", 0)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_clip(frames: list[np.ndarray], path: Path) -> Path:
        try:
            imageio.mimsave(path, frames, fps=fps)
            return path
        except Exception as exc:  # noqa: BLE001
            fallback = path.with_suffix(".gif")
            imageio.mimsave(fallback, frames, fps=fps, loop=0)
            print(
                f"[video] mp4 encode failed ({exc}); wrote GIF fallback: {fallback}"
            )
            return fallback

    episode_stats: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    for ep in range(max(n_episodes, 1)):
        obs = env.reset()
        # Drive the embedded ``_Dreamer`` in eval mode — single-env so we wrap
        # values in a length-1 batch dim, matching ``tools.simulate``'s
        # contract.
        state = None
        done = np.array([False])
        frames: list[np.ndarray] = [obs["image"].copy()]
        total_reward = 0.0
        last_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        step = 0
        while not done[0] and step < max_steps:
            batched = {k: np.stack([obs[k]]) for k in obs}
            action_dict, state = agent(batched, done, state, training=False)
            action = action_dict["action"][0].detach().cpu().numpy()
            obs, reward, is_last, last_info = env.step(action)
            total_reward += float(reward)
            frames.append(obs["image"].copy())
            done = np.array([is_last])
            terminated = bool(obs["is_terminal"])
            truncated = bool(is_last and not terminated)
            step += 1
        # Mirrors ``sb3_utils._record_video``: the env labels every episode it
        # ends (a scenario timeout arrives as a truncation, not a termination),
        # so ``"truncated"`` means only that the recorder hit its own cap.
        if terminated or truncated:
            termination = str(last_info.get("termination_reason", "unknown"))
        elif step >= max_steps:
            termination = "truncated"
        else:
            termination = "unknown"
        clip_path = out_path.with_name(f"{out_path.stem}_ep{ep + 1}{out_path.suffix}")
        written = _write_clip(frames, clip_path)
        video_paths.append(written)
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

    stats_path = out_path.with_name(out_path.stem + "_episodes.json")
    stats_path.write_text(
        json.dumps(
            {
                "fps": fps,
                "env_config": {
                    "scenario": scenario,
                    "doom_skill": env_cfg.get("doom_skill"),
                    "num_bots": int(env_cfg.get("num_bots", 0)),
                    "resize_shape": list(env_cfg.get("resize_shape", (64, 64))),
                    "frame_skip": int(env_cfg.get("frame_skip", 4)),
                    "grayscale": bool(env_cfg.get("grayscale", False)),
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


def _make_dreamer_env(scenario: str, env_cfg: dict[str, Any]) -> Any:
    """Build a single ``DreamerDoomEnv`` from the YAML env block."""
    from rl_doom.dreamer_env import DreamerDoomEnv

    return DreamerDoomEnv(
        scenario=scenario,
        resize_shape=tuple(env_cfg.get("resize_shape", (64, 64))),
        frame_skip=int(env_cfg.get("frame_skip", 4)),
        grayscale=bool(env_cfg.get("grayscale", False)),
        doom_skill=env_cfg.get("doom_skill"),
        num_bots=int(env_cfg.get("num_bots", 0)),
    )


def train_dreamer(
    *,
    scenario: str,
    run_dir: str | Path,
    total_timesteps: int,
    hyperparams: dict[str, Any],
    env_cfg: dict[str, Any],
    eval_cfg: dict[str, Any] | None = None,
    training_cfg: dict[str, Any] | None = None,
    port_path: str | Path,
    seed: int = 42,
    device: str | None = None,
    record_video: bool = True,
    video_episodes: int = 5,
    video_fps: int = 20,
    on_complete: Any = None,
    curriculum: Any = None,
) -> dict[str, Any]:
    """Train a DreamerV3 agent end-to-end and write *all* artefacts now.

    Mirrors :func:`rl_doom.sb3_utils.train_sb3`'s signature/return shape so
    notebook code stays uniform across PPO/DQN/Dreamer.

    On return ``run_dir`` contains:
    * ``checkpoints/{latest.pt, final.pt}`` — upstream's checkpoint format
    * ``metrics/training.npz`` — episode-aligned arrays + synthesized
      ``eval_rewards`` 5-column matrix (matches train_sb3's schema)
    * ``metrics/evaluations.npz`` — synthesized to match SB3's
      ``EvalCallback`` schema so the SB3 eval plotter / summary work
    * ``metrics/metrics.jsonl`` — written by upstream's ``tools.Logger``
    * ``figures/{learning_curves.png, eval_performance.png}``
    * ``media/dreamer_<scenario>_ep<N>.mp4`` + ``_episodes.json`` sidecar
    * ``tensorboard/`` event files
    * ``stage_summary.txt`` via ``rl_doom.summary.generate_run_summary``
    """
    import functools
    import json

    import numpy as np
    import torch

    from rl_doom.curriculum import (
        CurriculumController,
        CurriculumStage,
        apply_stage_to_doom_env,
        parse_curriculum_config,
    )
    from rl_doom.paths import mark_run_status, update_latest_symlink
    from rl_doom.sb3_utils import _plot_eval_performance
    from rl_doom.summary import generate_run_summary

    eval_cfg = dict(eval_cfg or {})
    training_cfg = dict(training_cfg or {})
    training_cfg.setdefault("total_timesteps", total_timesteps)

    run_dir = Path(run_dir)
    tb_dir = run_dir / "tensorboard"
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    media_dir = run_dir / "media"
    ckpt_dir = run_dir / "checkpoints"
    for d in (tb_dir, metrics_dir, figures_dir, media_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- curriculum parsing (before env construction so the first stage's
    #     difficulty seeds env_cfg) ---------------------------------------
    controller: CurriculumController | None = None
    sync_eval_env = True
    curriculum_stages: list[CurriculumStage] | None = None
    if curriculum is not None:
        if isinstance(curriculum, list):
            curriculum_stages = curriculum
            min_gap = 1
        else:
            curriculum_stages = parse_curriculum_config(curriculum)
            min_gap = int(curriculum.get("min_evals_between_promotions", 1))
            sync_eval_env = bool(curriculum.get("sync_eval_env", True))
        if curriculum_stages:
            controller = CurriculumController(
                curriculum_stages,
                min_evals_between_promotions=min_gap,
            )
            # Override env_cfg so the env starts on rung 0 (mirrors train_sb3).
            env_cfg = dict(env_cfg)
            first = curriculum_stages[0]
            if first.skill is not None:
                if (
                    env_cfg.get("doom_skill") is not None
                    and env_cfg["doom_skill"] != first.skill
                ):
                    print(
                        f"[train_dreamer] curriculum enabled: overriding env_cfg "
                        f"doom_skill={env_cfg['doom_skill']} with stage-0 "
                        f"skill={first.skill}",
                    )
                env_cfg["doom_skill"] = first.skill
            if first.num_bots is not None:
                if int(env_cfg.get("num_bots", 0)) != first.num_bots:
                    print(
                        f"[train_dreamer] curriculum enabled: overriding env_cfg "
                        f"num_bots={env_cfg.get('num_bots', 0)} with stage-0 "
                        f"num_bots={first.num_bots}",
                    )
                env_cfg["num_bots"] = first.num_bots

    port = _import_port(port_path)
    tools_mod = port["tools"]
    cfg = _build_config(
        port_path=port_path,
        scenario=scenario,
        env_cfg=env_cfg,
        hyperparams=hyperparams,
        eval_cfg=eval_cfg,
        training_cfg=training_cfg,
        seed=seed,
        device=device,
        logdir=tb_dir,
    )
    Path(cfg.traindir).mkdir(parents=True, exist_ok=True)
    Path(cfg.evaldir).mkdir(parents=True, exist_ok=True)

    tools_mod.set_seed_everywhere(cfg.seed)

    train_env_inner = _make_dreamer_env(scenario, env_cfg)
    eval_env_inner = _make_dreamer_env(scenario, env_cfg)
    cfg.num_actions = int(train_env_inner.action_space.n)

    train_envs = [_Damy(train_env_inner)]
    eval_envs = [_Damy(eval_env_inner)]

    # Apply the curriculum's initial stage now that envs exist. Always
    # propagates to the eval env regardless of ``sync_eval_env`` — that
    # flag only gates *future* promotions.
    if controller is not None:
        initial_stage = controller.record_initial()
        apply_stage_to_doom_env(train_env_inner._env, initial_stage)
        apply_stage_to_doom_env(eval_env_inner._env, initial_stage)
        print(
            f"[dreamer] curriculum start {controller.describe_stage()} "
            f"(stage 1 / {controller.num_stages})",
        )

    logger = tools_mod.Logger(tb_dir, 0)

    # --- prefill ---------------------------------------------------------
    train_eps: dict[str, Any] = {}
    eval_eps: dict[str, Any] = {}

    prefill_steps = max(0, int(getattr(cfg, "prefill", 0)))
    if prefill_steps:
        random_actor = tools_mod.OneHotDist(
            torch.zeros(cfg.num_actions).repeat(cfg.envs, 1),
        )

        def random_agent(o: Any, d: Any, s: Any) -> tuple[dict[str, Any], None]:
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        print(f"[dreamer] prefilling {prefill_steps} random env steps")
        tools_mod.simulate(
            random_agent,
            train_envs,
            train_eps,
            cfg.traindir,
            logger,
            limit=cfg.dataset_size,
            steps=prefill_steps,
        )
        logger.step += prefill_steps * cfg.action_repeat

    # --- agent -----------------------------------------------------------
    train_dataset = tools_mod.from_generator(
        tools_mod.sample_episodes(train_eps, cfg.batch_length), cfg.batch_size,
    )

    DreamerCls = _build_dreamer_class(port)
    agent = DreamerCls(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        cfg,
        logger,
        train_dataset,
    ).to(cfg.device)
    agent.requires_grad_(requires_grad=False)

    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=cfg.device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        tools_mod.recursively_load_optim_state_dict(
            agent, ckpt["optims_state_dict"],
        )
        agent._should_pretrain._once = False

    # --- train loop ------------------------------------------------------
    eval_history: list[tuple[int, list[float], list[int]]] = []
    state = None
    t_start_wall = __import__("time").time()
    while agent._step < cfg.steps + cfg.eval_every:
        logger.write()
        if cfg.eval_episode_num > 0:
            print(f"[dreamer] eval @ step {agent._step}")
            eval_policy = functools.partial(agent, training=False)
            tools_mod.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                cfg.evaldir,
                logger,
                is_eval=True,
                episodes=cfg.eval_episode_num,
            )
            eval_rewards, eval_lengths = _scan_episode_npzs(Path(cfg.evaldir))
            recent_r = eval_rewards[-cfg.eval_episode_num:]
            recent_l = eval_lengths[-cfg.eval_episode_num:]
            if recent_r:
                eval_history.append(
                    (int(agent._step * cfg.action_repeat), recent_r, recent_l),
                )

            # Curriculum: feed this eval to the controller and apply any
            # promotion to both envs (eval env only when sync_eval_env).
            # Always emit current-stage scalars on every eval so the TB
            # series isn't sparse for runs without promotions.
            if controller is not None and recent_r:
                cur_step = int(agent._step * cfg.action_repeat)
                mean_r = float(np.mean(recent_r))
                new_stage = controller.maybe_promote(
                    current_step=cur_step, eval_mean_reward=mean_r,
                )
                if new_stage is not None:
                    apply_stage_to_doom_env(train_env_inner._env, new_stage)
                    if sync_eval_env:
                        apply_stage_to_doom_env(eval_env_inner._env, new_stage)
                    print(
                        f"[dreamer] curriculum step={cur_step} "
                        f"eval_mean={mean_r:.2f} -> promote to "
                        f"{controller.describe_stage(new_stage)} "
                        f"(stage {controller.current_stage_index + 1} / "
                        f"{controller.num_stages})",
                    )
                    logger.scalar("curriculum/promotion_step", float(cur_step))
                if controller.current_skill is not None:
                    logger.scalar(
                        "curriculum/skill", float(controller.current_skill),
                    )
                if controller.current_num_bots is not None:
                    logger.scalar(
                        "curriculum/num_bots", float(controller.current_num_bots),
                    )
                logger.scalar(
                    "curriculum/stage_index",
                    float(controller.current_stage_index),
                )

        next_target = min(agent._step + cfg.eval_every, cfg.steps)
        print(f"[dreamer] train @ step {agent._step} -> {next_target}")
        state = tools_mod.simulate(
            agent,
            train_envs,
            train_eps,
            cfg.traindir,
            logger,
            limit=cfg.dataset_size,
            steps=cfg.eval_every,
            state=state,
        )
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools_mod.recursively_collect_optim_state_dict(
                    agent,
                ),
            },
            latest_ckpt,
        )

    wall_time = __import__("time").time() - t_start_wall

    # --- close envs (best-effort) ---------------------------------------
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    # --- save final checkpoint ------------------------------------------
    final_ckpt = ckpt_dir / "final.pt"
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools_mod.recursively_collect_optim_state_dict(agent),
        },
        final_ckpt,
    )

    # --- aggregate per-episode stats ------------------------------------
    train_rewards, train_lengths = _scan_episode_npzs(Path(cfg.traindir))

    # Synthesise the SB3-shaped evaluations.npz so the existing eval plotter
    # and summary code work without a fork.
    evaluations_path = metrics_dir / "evaluations.npz"
    _synthesize_evaluations_npz(eval_history, evaluations_path)

    # Build training.npz in train_sb3's schema so generate_run_summary works.
    eval_log: np.ndarray = np.empty((0, 5), dtype=np.float32)
    if eval_history:
        rows: list[list[float]] = []
        for ts, rs, ls in eval_history:
            r_arr = np.asarray(rs, dtype=np.float32)
            l_arr = np.asarray(ls, dtype=np.float32)
            rows.append([
                float(ts), float(r_arr.mean()), float(r_arr.std()),
                float(l_arr.mean()), float(l_arr.std()),
            ])
        eval_log = np.asarray(rows, dtype=np.float32)

    fps = float(cfg.steps / wall_time) if wall_time > 0 else 0.0
    np.savez(
        metrics_dir / "training.npz",
        episode_rewards=np.asarray(train_rewards, dtype=np.float32),
        episode_lengths=np.asarray(train_lengths, dtype=np.int64),
        eval_rewards=eval_log,
        wall_time_seconds=np.float64(wall_time),
        fps=np.float64(fps),
        total_env_steps=np.int64(cfg.steps),
    )

    # --- learning curves + eval figure ----------------------------------
    learning_curves_path = figures_dir / "learning_curves.png"
    _plot_dreamer_curves(
        scenario, train_rewards, train_lengths,
        tb_dir / "metrics.jsonl",
        learning_curves_path,
    )
    if eval_history:
        evaluations_dict: dict[str, np.ndarray] = {
            "timesteps": np.asarray(
                [t for t, _r, _l in eval_history], dtype=np.int64,
            ),
            "results": np.asarray(
                [r for _t, r, _l in eval_history], dtype=np.float32,
            ),
            "ep_lengths": np.asarray(
                [le for _t, _r, le in eval_history], dtype=np.int64,
            ),
        }
        _plot_eval_performance(
            "dreamer", scenario, evaluations_dict,
            figures_dir / "eval_performance.png",
        )

    # --- video -----------------------------------------------------------
    video_paths: list[Path] = []
    if record_video:
        video_paths = _record_dreamer_video(
            agent,
            scenario,
            media_dir / f"dreamer_{scenario}.mp4",
            fps=video_fps,
            n_episodes=video_episodes,
            env_cfg=env_cfg,
        )

    # --- curriculum.json (matches train_sb3's schema so summary.py +
    #     matrix CSV consumers work unchanged) ---------------------------
    if controller is not None:
        (metrics_dir / "curriculum.json").write_text(
            json.dumps(
                {
                    "stages": [
                        {
                            "skill": s.skill,
                            "num_bots": s.num_bots,
                            "promote_at": s.promote_at,
                        }
                        for s in (curriculum_stages or [])
                    ],
                    "promotions": controller.promotions,
                    "final_stage_index": controller.current_stage_index,
                    "final_skill": controller.current_skill,
                    "final_num_bots": controller.current_num_bots,
                },
                indent=2,
            ),
        )

    # --- finalise --------------------------------------------------------
    mark_run_status(run_dir, status="completed")
    update_latest_symlink(scenario, "dreamer", run_dir)
    (run_dir / "stage_summary.txt").write_text(generate_run_summary(run_dir))

    final_eval_mean = float(eval_log[-1, 1]) if eval_log.shape[0] else None
    best_eval_mean = float(eval_log[:, 1].max()) if eval_log.shape[0] else None
    result: dict[str, Any] = {
        "run_dir": run_dir,
        "wall_time_seconds": wall_time,
        "fps": fps,
        "mean_eval_reward": final_eval_mean,
        "best_eval_reward": best_eval_mean,
        "video_paths": video_paths,
        "learning_curves_path": learning_curves_path,
        "curriculum_final_skill": (
            controller.current_skill if controller is not None else None
        ),
        "curriculum_final_num_bots": (
            controller.current_num_bots if controller is not None else None
        ),
        "curriculum_final_stage_index": (
            controller.current_stage_index if controller is not None else None
        ),
        "curriculum_promotions": (
            controller.promotions if controller is not None else None
        ),
    }
    if on_complete is not None:
        try:
            on_complete(run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[train_dreamer] on_complete hook raised: {exc}")
    return result
