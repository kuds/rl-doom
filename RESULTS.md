# Results — Deadly Corridor, Baseline vs. Skill Curriculum

This page tracks the headline comparison for the curriculum work: does
ramping `doom_skill` 1 → 2 → 3 during training beat training directly on
skill 3 for Deadly Corridor? Populated as matrix runs land.

## Experimental setup

- **Scenario:** `deadly_corridor`
- **Algorithms:** PPO, DQN, Recurrent PPO
- **Seeds per cell:** 3 (`42`, `123`, `777`)
- **Total cells:** 6 variants × 3 seeds = 18 runs
- **Budget:** each variant uses its own YAML's `training.total_timesteps`
  (PPO/RPPO 2 M, DQN 1 M — see `configs/*deadly_corridor*.yaml`)
- **Curriculum stages (PPO / RPPO):**
  `skill 1 → 2 @ eval≥40`, `skill 2 → 3 @ eval≥60`, `skill 3 terminal`
- **Curriculum stages (DQN):** looser gates (`30` / `50`) because DQN
  eval variance is higher without VecNormalize reward scaling.

Kick the full matrix off with:

```bash
python -m scripts.run_experiment_matrix \
    --matrix configs/matrix/deadly_corridor_curriculum.yaml
```

All cells write to `training_jobs/deadly_corridor/<algo>/runs/` and a
roll-up CSV lands at `training_jobs/_matrix/deadly_corridor_curriculum.csv`.

## Headline numbers

Mean ± std of `mean_eval_reward` over the 3 seeds per cell. **To be
filled in once the matrix runs.** Pull from
`training_jobs/_matrix/deadly_corridor_curriculum.csv`:

| Algorithm | Baseline (skill 3 from start) | Curriculum (1 → 2 → 3) | Δ |
|---|---|---|---|
| PPO | _pending_ | _pending_ | — |
| DQN | _pending_ | _pending_ | — |
| Recurrent PPO | _pending_ | _pending_ | — |

### Promotion timeline (curriculum runs only)

Per-seed, the step at which the callback promoted from one skill to the
next, read from `<run_dir>/metrics/curriculum.json`.

| Algorithm | Seed | skill 1 → 2 step | skill 2 → 3 step | Final skill |
|---|---|---|---|---|
| PPO | 42 | _pending_ | _pending_ | _pending_ |
| PPO | 123 | _pending_ | _pending_ | _pending_ |
| PPO | 777 | _pending_ | _pending_ | _pending_ |
| DQN | 42 | _pending_ | _pending_ | _pending_ |
| … | | | | |

## Reading the artifacts

Every run directory has the same layout:

```
training_jobs/deadly_corridor/<algo>/runs/<ts>_seed<N>_<matrix>_<variant>/
├── config.json               # full merged config (hyperparams + curriculum)
├── checkpoints/              # SB3 .zip snapshots + final model
├── figures/
│   ├── learning_curves.png
│   └── eval_performance.png
├── media/<algo>_deadly_corridor.mp4   # greedy rollout on the final skill
├── metrics/
│   ├── training.npz          # eval_rewards[(step, mean, std), ...]
│   └── curriculum.json       # promotion timeline (curriculum runs only)
└── runs/                     # TensorBoard event files
```

### TensorBoard

```bash
tensorboard --logdir training_jobs/deadly_corridor/
```

Every variant × seed shows up as a separate run; the run-name prefix
encodes the matrix + variant so filtering is a substring match. The
curriculum callback additionally logs three custom scalars:

- `curriculum/skill` — current `doom_skill` at each eval
- `curriculum/stage_index` — which rung of the ladder we're on
- `curriculum/promotion_step` — single point at each promotion

### Matrix summary CSV

```
training_jobs/_matrix/deadly_corridor_curriculum.csv
```

One row per `(variant, seed)`:

```
matrix,variant,scenario,algo,seed,mean_eval_reward,best_eval_reward,success_rate,curriculum_final_skill,wall_time_seconds,run_dir
```

Load it in a notebook and `groupby("variant").agg(["mean", "std"])` for
the headline table.

## Qualitative observations

_Drop gameplay GIFs and a sentence or two of commentary here once the
matrix finishes. Suggested shots:_

- Skill-3 PPO baseline dying in the first room (shows why the curriculum
  is needed).
- Curriculum-PPO at final checkpoint clearing the corridor.
- `curriculum/skill` TensorBoard scalar stepping 1 → 2 → 3.

## Next steps

- **Reward shaping baseline.** Curriculum helps, but it doesn't address
  the underlying sparse-reward problem. The natural next comparison is
  curriculum vs. `GameVariable.POSITION_X`-based shaping vs. both.
- **HER for DQN.** Would require injecting agent `(x, y)` into a
  `Dict` observation and wiring `HerReplayBuffer`; non-trivial, not
  implemented yet.
- **Port the matrix to `defend_the_center` and `deathmatch`.** Same
  YAML template, different stage thresholds.
