# RL-Doom

Reinforcement learning agents that learn to play Doom using [ViZDoom](https://github.com/Farama-Foundation/ViZDoom). Implements **Double DQN** and **PPO** with CNN-based visual policies, progressing from simple shooting to complex combat scenarios.

## Scenarios

Every scenario below has a matching config for both algorithms in `configs/<algo>_<scenario>.yaml`.

| Scenario | Description | Step budget |
|----------|-------------|-------------|
| **Basic** | Single room — shoot a stationary monster | 100k |
| **Deadly Corridor** | Run a gauntlet of enemies to reach armor | 200k |
| **Defend the Center** | Stationary 360° defense against waves | 200k |
| **Deathmatch** | Full free-for-all map: navigate, aim, and fight | 1M–2M |

See [`BENCHMARKS.md`](BENCHMARKS.md) for per-scenario episode timeouts,
reward structures, and "good agent" reference scores pulled from the
ViZDoom literature.

## Project Structure

```
rl-doom/
├── notebooks/
│   ├── 01_environment_exploration.ipynb   # Env setup, wrappers, random baseline
│   ├── 02_dqn_training.ipynb              # Double DQN training + LR sweep
│   ├── 03_ppo_training.ipynb              # PPO training loop
│   ├── 04_recurrent_ppo_training.ipynb    # Recurrent PPO (CnnLstmPolicy) training loop
│   └── 05_analysis_and_results.ipynb      # Cross-scenario comparison, saliency, GIFs
├── src/rl_doom/                           # Core library
│   ├── env.py                             # Gymnasium wrappers (DoomEnv, FrameStack, etc.)
│   ├── models.py                          # CNN networks (DQN, Actor-Critic)
│   ├── agents/
│   │   ├── dqn.py                         # Double DQN agent
│   │   └── ppo.py                         # PPO agent (Recurrent PPO uses sb3-contrib directly)
│   ├── replay_buffer.py                   # Experience replay for DQN
│   ├── train.py                           # CLI training entry point
│   └── evaluate.py                        # Evaluation & episode recording
├── configs/                               # YAML hyperparameter configs per (algorithm, scenario) pair
├── pyproject.toml
└── PLAN.md
```

## Setup

```bash
# Clone
git clone https://github.com/kuds/rl-doom.git
cd rl-doom

# Install (editable)
pip install -e ".[notebooks,dev]"
```

### Google Colab

Each notebook includes a commented-out **Colab Setup** cell at the top. Uncomment it to automatically:
1. Clone the repo to `/content/rl-doom`
2. Install the package with all dependencies (`pip install -e .[notebooks]`)
3. Set the working directory to `notebooks/`

There is also a separate **Google Drive** cell you can uncomment to persist artifacts (checkpoints, logs, figures, media) across Colab sessions via Google Drive symlinks.

## Quick Start

### Notebooks

Run the notebooks in order:

1. **01 — Environment Exploration**: Verify ViZDoom installation, visualize observations, run random baseline
2. **02 — DQN Training**: Train Double DQN, plot learning curves, run a hyperparameter sweep
3. **03 — PPO Training**: Train PPO, log policy/value losses and entropy
4. **04 — Recurrent PPO Training**: Train RecurrentPPO (`sb3_contrib.RecurrentPPO`, CnnLstmPolicy) for memory-dependent scenarios
5. **05 — Analysis**: Cross-algorithm × cross-scenario comparison table, multi-seed evaluation, saliency maps, GIF generation

### CLI

Pick any `(algorithm, scenario)` pair:

```bash
python -m rl_doom.train --config configs/dqn_basic.yaml
python -m rl_doom.train --config configs/ppo_deadly_corridor.yaml
python -m rl_doom.train --config configs/recurrent_ppo_deadly_corridor.yaml
```

## Algorithms

### Double DQN
- CNN feature extractor (Nature-DQN architecture)
- Epsilon-greedy exploration with linear decay
- Experience replay buffer
- Target network with periodic hard updates

### PPO
- Shared CNN backbone with actor-critic heads
- Clipped surrogate objective
- GAE (Generalized Advantage Estimation)
- Entropy bonus for exploration
- Mini-batch updates over collected rollouts

### Recurrent PPO (sb3-contrib)
- `CnnLstmPolicy`: NatureCNN feature extractor → LSTM (256-d hidden by default) → actor + critic heads
- Same clipped surrogate + GAE as PPO; LSTM hidden state is reset at episode boundaries via `episode_starts`
- Designed for partially-observable scenarios where a 4-frame stack isn't enough memory (deadly_corridor enemies leaving FOV, deathmatch map exploration)
- Wall-clock slower than PPO at the same step count because the LSTM forward/backward serialises within each rollout segment

## Curriculum Learning

Every scenario config accepts an optional `curriculum:` block that ramps
one of two difficulty knobs during training once the agent clears a
per-stage eval threshold:

- **`skill`** — ViZDoom's `doom_skill` (1..5). Used by Deadly Corridor,
  where the death penalty at skill 3 kills the agent before it can
  discover the "push forward + shoot" gradient. Starting on skill 1
  lets the policy learn distance shaping first, then ramping up to
  skill 3 tunes combat without losing the navigation prior.
- **`num_bots`** — count of ZDoom AI bots spawned on the deathmatch
  map (capped at 8, matching the stock `deathmatch.cfg` roster). With
  0 bots the scenario has no enemies and zero reward signal; with 8
  bots a fresh policy dies before landing kills. A 2 → 4 → 8 ramp
  gives frequent combat encounters that scale with the policy's
  capability.

```yaml
curriculum:
  enabled: true
  min_evals_between_promotions: 2
  sync_eval_env: true
  stages:
    - {skill: 1, promote_at: 1500.0}   # start easy
    - {skill: 2, promote_at: 1500.0}
    - {skill: 3, promote_at: null}     # terminal
```

Thresholds are on the raw eval-reward scale (`EvalCallback` does not
apply VecNormalize rescaling). See [`BENCHMARKS.md`](BENCHMARKS.md) for
per-skill reference scores used to calibrate them.

Paired "baseline vs. curriculum" YAMLs live under `configs/`:

| Scenario | Baseline | Curriculum (knob) |
|---|---|---|
| Deadly Corridor | `ppo_deadly_corridor.yaml` | `ppo_deadly_corridor_curriculum.yaml` (skill) |
| Deadly Corridor | `dqn_deadly_corridor.yaml` | `dqn_deadly_corridor_curriculum.yaml` (skill) |
| Deadly Corridor | `recurrent_ppo_deadly_corridor.yaml` | `recurrent_ppo_deadly_corridor_curriculum.yaml` (skill) |
| Deathmatch | `ppo_deathmatch.yaml` | `ppo_deathmatch_curriculum.yaml` (num_bots) |
| Deathmatch | `dqn_deathmatch.yaml` | `dqn_deathmatch_curriculum.yaml` (num_bots) |
| Deathmatch | `recurrent_ppo_deathmatch.yaml` | `recurrent_ppo_deathmatch_curriculum.yaml` (num_bots) |

Under the hood, `SkillCurriculumCallback`
(`src/rl_doom/curriculum.py`) watches the SB3 `EvalCallback` and, when
the eval mean clears the current stage's `promote_at`, applies the new
stage's knobs: `DoomGame.set_doom_skill` for `skill` changes and a
direct write to `DoomEnv._num_bots` for `num_bots` changes (the latter
takes effect on the next `reset()` because `addbot` commands are
re-issued per episode). The promotion timeline is saved to
`metrics/curriculum.json` and logged to TensorBoard as
`curriculum/skill`, `curriculum/num_bots`, `curriculum/stage_index`.

## Experiment Matrix Runner

`scripts/run_experiment_matrix.py` runs a grid of (variant × seed) combos
from a single YAML so a baseline-vs-curriculum comparison across
algorithms can be kicked off in one command:

```bash
python -m scripts.run_experiment_matrix \
    --matrix configs/matrix/deadly_corridor_curriculum.yaml
```

Each cell writes to the standard
`training_jobs/<scenario>/<algo>/runs/<timestamp>_seed<N>_<matrix>_<variant>/`
layout, so TensorBoard's `--logdir training_jobs/deadly_corridor/` shows
every variant side-by-side. A roll-up CSV lands at
`training_jobs/_matrix/<matrix_name>.csv` with one row per run:

```
matrix,variant,scenario,algo,seed,mean_eval_reward,best_eval_reward,success_rate,curriculum_final_skill,wall_time_seconds,run_dir
```

Pass `--dry-run` to expand the grid without training, or
`--total-timesteps 50000` for a smoke test.

## Visual Pipeline

```
ViZDoom RGB → Grayscale → Resize 84x84 → Stack 4 frames → CNN
```

## Training Artifacts

All notebooks save artifacts to disk for reproducibility:

| Directory | Contents |
|-----------|----------|
| `figures/` | Learning curves, evaluation plots, saliency maps, action distributions (PNG) |
| `logs/` | Training metrics (`.npz`), cross-scenario comparisons (`.csv`), saliency arrays, multi-seed evaluations |
| `checkpoints/` | Model weights (`.pt`) and full checkpoints with optimizer state & config (`*_full.pt`) |
| `media/` | Gameplay GIFs of trained agents |
| `runs/` | TensorBoard event files for real-time monitoring |

### Metrics logged per algorithm

| Metric | DQN | PPO | Recurrent PPO |
|--------|-----|-----|---------------|
| Episode rewards | Yes | Yes | Yes |
| Episode lengths | Yes | Yes | Yes |
| Training loss | Yes (Huber) | Yes (policy + value) | Yes (policy + value) |
| Epsilon schedule | Yes | — | — |
| Mean Q-values | Yes | — | — |
| Action distribution | Yes | — | — |
| Policy entropy | — | Yes | Yes |
| Clip fraction | — | Yes | Yes |
| Eval rewards + lengths | Yes | Yes | Yes |
| Wall-clock time & FPS | Yes | Yes | Yes |
| Reproducibility metadata | Yes | Yes | Yes |

### TensorBoard

Launch TensorBoard to monitor training in real time:

```bash
tensorboard --logdir runs/
```

## Shared Utilities

`src/rl_doom/utils.py` provides reusable helpers for Google Drive persistence, reproducibility metadata collection, smoothed plotting, training timers, and checkpoint saving with metadata.

## Blog Posts
- [Advanced Architectures and Methodologies in Visual Reinforcement Learning: A Technical Analysis of the ViZDoom Platform](https://www.findingtheta.com/blog/advanced-architectures-and-methodologies-in-visual-reinforcement-learning-a-technical-analysis-of-the-vizdoom-platform)

## License

MIT
