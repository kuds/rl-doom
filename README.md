# RL-Doom

Reinforcement learning agents that learn to play Doom using [ViZDoom](https://github.com/Farama-Foundation/ViZDoom). Implements **Double DQN** and **PPO** with CNN-based visual policies, progressing from simple shooting to complex combat scenarios.

## Scenarios

| Scenario | Description | Algorithm |
|----------|-------------|-----------|
| **Basic** | Single room — shoot a monster | DQN |
| **Deadly Corridor** | Navigate a corridor with enemies | PPO |
| **Defend the Center** | Survive waves of enemies | PPO |
| **Deathmatch** | Full combat (future) | — |

## Project Structure

```
rl-doom/
├── notebooks/
│   ├── 01_environment_exploration.ipynb   # Env setup, wrappers, random baseline
│   ├── 02_dqn_training.ipynb              # Double DQN on Basic scenario
│   ├── 03_ppo_training.ipynb              # PPO on Deadly Corridor & Defend the Center
│   └── 04_analysis_and_results.ipynb      # Cross-scenario comparison, saliency, GIFs
├── src/rl_doom/                           # Core library
│   ├── env.py                             # Gymnasium wrappers (DoomEnv, FrameStack, etc.)
│   ├── models.py                          # CNN networks (DQN, Actor-Critic)
│   ├── agents/
│   │   ├── dqn.py                         # Double DQN agent
│   │   └── ppo.py                         # PPO agent
│   ├── replay_buffer.py                   # Experience replay for DQN
│   ├── train.py                           # CLI training entry point
│   └── evaluate.py                        # Evaluation & episode recording
├── configs/                               # YAML hyperparameter configs per scenario
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
2. **02 — DQN Training**: Train Double DQN on Basic (100k steps), plot learning curves, hyperparameter sweep
3. **03 — PPO Training**: Train PPO on Deadly Corridor and Defend the Center (200k steps each)
4. **04 — Analysis**: Cross-scenario comparison table, multi-seed evaluation, saliency maps, GIF generation

### CLI

```bash
python -m rl_doom.train --config configs/basic.yaml
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

| Metric | DQN | PPO |
|--------|-----|-----|
| Episode rewards | Yes | Yes |
| Episode lengths | Yes | Yes |
| Training loss | Yes (Huber) | Yes (policy + value) |
| Epsilon schedule | Yes | — |
| Mean Q-values | Yes | — |
| Action distribution | Yes | — |
| Policy entropy | — | Yes |
| Clip fraction | — | Yes |
| Eval rewards + lengths | Yes | Yes |
| Wall-clock time & FPS | Yes | Yes |
| Reproducibility metadata | Yes | Yes |

### TensorBoard

Launch TensorBoard to monitor training in real time:

```bash
tensorboard --logdir runs/
```

### Shared Utilities

`src/rl_doom/utils.py` provides reusable helpers for Google Drive persistence,
reproducibility metadata collection, smoothed plotting, training timers, and
checkpoint saving with metadata.

## Blog Posts
- [Advanced Architectures and Methodologies in Visual Reinforcement Learning: A Technical Analysis of the ViZDoom Platform](https://www.findingtheta.com/blog/advanced-architectures-and-methodologies-in-visual-reinforcement-learning-a-technical-analysis-of-the-vizdoom-platform)

## License

MIT
