"""Run a matrix of training variants and track each under the standard layout.

Why this exists
---------------

The notebooks train one YAML config at a time. Comparing "PPO baseline vs
PPO curriculum vs DQN baseline" from a notebook means re-running three
cells and manually keeping the run directories organised. The matrix
runner collapses that into a single CLI call that:

* Loads a *matrix YAML* describing variants to run,
* Expands it over the provided seeds,
* Invokes :func:`rl_doom.sb3_utils.train_sb3` for each (variant, seed) cell,
* Writes every run under the same
  ``training_jobs/<scenario>/<algo>/runs/<timestamp>_seed<N>_<variant>/``
  tree, so TensorBoard's ``--logdir training_jobs/deadly_corridor/``
  shows every result in a single browser tab.
* Drops a ``matrix_summary.csv`` at the matrix root with one row per run
  for quick grep/pandas queries.

Matrix YAML schema
------------------

.. code-block:: yaml

    # configs/matrix/deadly_corridor_curriculum.yaml
    name: deadly_corridor_curriculum
    seeds: [42, 123, 777]
    variants:
      - name: ppo_baseline
        base_config: ppo_deadly_corridor
      - name: ppo_curriculum
        base_config: ppo_deadly_corridor_curriculum
      - name: dqn_baseline
        base_config: dqn_deadly_corridor
      - name: dqn_curriculum
        base_config: dqn_deadly_corridor_curriculum

Each variant is a thin pointer to an existing ``configs/<name>.yaml`` —
the matrix file does *not* duplicate hyperparameters. ``seeds`` at the
top level fan out every variant across those seeds.

Optional per-variant ``overrides:`` apply arbitrary patches on top of
the base config (merge is shallow-per-block; e.g. overriding
``hyperparams.lr`` leaves the rest of ``hyperparams`` untouched).

Usage
-----

.. code-block:: bash

    python -m scripts.run_experiment_matrix \\
        --matrix configs/matrix/deadly_corridor_curriculum.yaml

    # Dry run — print the (variant, seed) grid without training.
    python -m scripts.run_experiment_matrix --matrix <path> --dry-run

    # Cap each run to a shorter budget for a fast smoke test.
    python -m scripts.run_experiment_matrix --matrix <path> --total-timesteps 50000
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

# Support running as ``python scripts/run_experiment_matrix.py`` directly
# (outside an installed env) by injecting the src/ tree onto sys.path.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rl_doom.paths import (  # noqa: E402
    REPO_ROOT,
    TRAINING_JOBS,
    load_yaml_config,
    new_run_dir,
    write_config,
)

# ``sb3_utils`` pulls in torch + matplotlib + stable-baselines3. Importing
# it at module load time would make the pure-Python helpers in this file
# (``_deep_merge``, ``_expand_variants``) unusable in lean environments
# that just want to smoke-test the matrix expansion. Defer those imports
# to ``_run_one`` where they're actually needed.


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` without mutating inputs."""
    out = deepcopy(base)
    for key, value in overrides.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _expand_variants(matrix_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Cross variants with seeds and return one spec dict per run."""
    seeds = matrix_cfg.get("seeds", [matrix_cfg.get("seed", 42)])
    if not seeds:
        raise ValueError("matrix config must provide 'seeds' (or 'seed')")
    variants = matrix_cfg.get("variants")
    if not variants:
        raise ValueError("matrix config must provide a non-empty 'variants' list")

    runs: list[dict[str, Any]] = []
    for variant in variants:
        name = variant.get("name")
        if not name:
            raise ValueError("every variant must have a 'name'")
        base_path = variant.get("base_config")
        if not base_path:
            raise ValueError(f"variant {name!r} missing 'base_config'")
        base_cfg = load_yaml_config(base_path)
        overrides = variant.get("overrides") or {}
        merged = _deep_merge(base_cfg, overrides)
        for seed in seeds:
            spec = deepcopy(merged)
            spec["seed"] = int(seed)
            spec["_variant_name"] = name
            spec["_base_config"] = str(base_path)
            runs.append(spec)
    return runs


def _run_one(
    spec: dict[str, Any],
    *,
    matrix_name: str,
    total_timesteps_override: int | None,
    device: str | None,
) -> dict[str, Any]:
    # Heavy imports stay deferred so the pure-logic tests can import
    # this module without pulling in torch / SB3 / matplotlib.
    from rl_doom.sb3_utils import (
        gpu_info,
        policy_kwargs_from_config,
        train_sb3,
    )

    scenario = spec["scenario"]
    algo = spec["algorithm"]
    seed = int(spec["seed"])
    variant_name = spec["_variant_name"]

    hp = dict(spec["hyperparams"])
    env_cfg = dict(spec.get("env", {}))
    policy_cfg = dict(spec.get("policy", {}))
    training_cfg = dict(spec["training"])
    eval_cfg = dict(spec.get("eval", {}))
    curriculum_cfg = spec.get("curriculum")

    total_ts = int(
        total_timesteps_override
        if total_timesteps_override is not None
        else training_cfg["total_timesteps"],
    )
    n_envs = int(env_cfg.get("n_envs", 1))

    # The tag embeds both the matrix name and the variant so a TensorBoard
    # run list is immediately parseable at the parent scenario level.
    tag = f"{matrix_name}_{variant_name}"
    run_dir = new_run_dir(scenario, algo, seed=seed, tag=tag)

    write_config(
        run_dir,
        env=scenario,
        algo=algo,
        seed=seed,
        hyperparams={**hp, "total_timesteps": total_ts, "n_envs": n_envs},
        env_settings=env_cfg,
        policy_kwargs=policy_cfg,
        gpu_setup=gpu_info(),
        training_schedule={
            "checkpoint_freq": int(training_cfg.get("checkpoint_freq", 100_000)),
            "eval_freq": int(eval_cfg.get("eval_freq", 25_000)),
            "eval_episodes": int(eval_cfg.get("n_episodes", 10)),
        },
        curriculum=curriculum_cfg,
        matrix_name=matrix_name,
        variant=variant_name,
        base_config=spec.get("_base_config"),
    )

    t_start = time.time()
    print(
        f"\n[matrix:{matrix_name}] variant={variant_name} scenario={scenario} "
        f"algo={algo} seed={seed} run_dir={run_dir}",
    )
    result = train_sb3(
        algo=algo,
        scenario=scenario,
        run_dir=run_dir,
        hyperparams=hp,
        seed=seed,
        total_timesteps=total_ts,
        n_envs=n_envs,
        eval_freq=int(eval_cfg.get("eval_freq", 25_000)),
        eval_episodes=int(eval_cfg.get("n_episodes", 10)),
        checkpoint_freq=int(training_cfg.get("checkpoint_freq", 100_000)),
        record_video=True,
        device=device,
        resize_shape=tuple(env_cfg.get("resize_shape", (84, 84))),
        frame_skip=int(env_cfg.get("frame_skip", 4)),
        num_stack=int(env_cfg.get("num_stack", 4)),
        doom_skill=env_cfg.get("doom_skill"),
        num_bots=int(env_cfg.get("num_bots", 0)),
        policy_kwargs=policy_kwargs_from_config(policy_cfg),
        curriculum=curriculum_cfg,
    )
    wall = time.time() - t_start
    print(
        f"[matrix:{matrix_name}] done variant={variant_name} seed={seed} "
        f"wall={wall:.1f}s eval_mean={result.get('mean_eval_reward')}",
    )

    return {
        "matrix": matrix_name,
        "variant": variant_name,
        "scenario": scenario,
        "algo": algo,
        "seed": seed,
        "run_dir": str(result["run_dir"]),
        "mean_eval_reward": result.get("mean_eval_reward"),
        "best_eval_reward": result.get("best_eval_reward"),
        "success_rate": result.get("success_rate"),
        "wall_time_seconds": result.get("wall_time_seconds"),
        "curriculum_final_skill": result.get("curriculum_final_skill"),
    }


def _write_matrix_summary(
    matrix_name: str, rows: list[dict[str, Any]],
) -> Path:
    """Drop a single CSV at ``training_jobs/_matrix/<matrix_name>.csv``."""
    out_dir = TRAINING_JOBS / "_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{matrix_name}.csv"
    fieldnames = [
        "matrix",
        "variant",
        "scenario",
        "algo",
        "seed",
        "mean_eval_reward",
        "best_eval_reward",
        "success_rate",
        "curriculum_final_skill",
        "wall_time_seconds",
        "run_dir",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--matrix",
        required=True,
        help="Path to a matrix YAML (absolute, relative, or bare stem).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded (variant, seed) grid and exit.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help=(
            "Override each run's ``training.total_timesteps`` (useful for a "
            "smoke test). Defaults to each variant's YAML value."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (``cpu`` / ``cuda``). Default: auto-detect.",
    )
    args = parser.parse_args(argv)

    matrix_path = Path(args.matrix)
    if not matrix_path.suffix and not matrix_path.exists():
        matrix_path = REPO_ROOT / "configs" / "matrix" / f"{matrix_path.name}.yaml"
    matrix_cfg = load_yaml_config(matrix_path)
    matrix_name = matrix_cfg.get("name") or matrix_path.stem

    runs = _expand_variants(matrix_cfg)
    print(f"[matrix:{matrix_name}] expanded to {len(runs)} runs")
    for spec in runs:
        print(
            f"  - variant={spec['_variant_name']} scenario={spec['scenario']} "
            f"algo={spec['algorithm']} seed={spec['seed']}",
        )
    if args.dry_run:
        return 0

    rows: list[dict[str, Any]] = []
    for spec in runs:
        rows.append(
            _run_one(
                spec,
                matrix_name=matrix_name,
                total_timesteps_override=args.total_timesteps,
                device=args.device,
            ),
        )
    summary_path = _write_matrix_summary(matrix_name, rows)
    print(f"\n[matrix:{matrix_name}] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
