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
