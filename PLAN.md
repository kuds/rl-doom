# RL-Doom: Implementation Plan

## Overview

Build a reinforcement learning agent that learns to play Doom using the
[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) environment. The project
will progress from simple scenarios to complex gameplay, using modern RL
algorithms (PPO, DQN) with CNN-based visual policies.

---

## Phase 1: Project Scaffolding & Environment Setup  ✓

### 1.1 Project Structure
```
rl-doom/
├── LICENSE
├── README.md
├── pyproject.toml              # Dependencies & project metadata
├── .gitignore
├── configs/
│   ├── dqn_basic.yaml               # <algo>_<scenario>.yaml per pair
│   ├── ppo_basic.yaml
│   ├── dqn_deadly_corridor.yaml
│   ├── ppo_deadly_corridor.yaml
│   ├── dqn_defend_the_center.yaml
│   └── ppo_defend_the_center.yaml
├── src/
│   └── rl_doom/
│       ├── __init__.py
│       ├── env.py              # ViZDoom env wrappers (Gymnasium API)
│       ├── models.py           # CNN policy & value networks
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── dqn.py          # DQN agent
│       │   └── ppo.py          # PPO agent
│       ├── replay_buffer.py    # Experience replay (for DQN)
│       ├── train.py            # Training loop entry point
│       ├── evaluate.py         # Evaluation / recording
│       └── utils.py            # Logging, seeding, video helpers
├── notebooks/
│   ├── 01_environment_exploration.ipynb
│   ├── 02_dqn_training.ipynb
│   ├── 03_ppo_training.ipynb
│   ├── 04_recurrent_ppo_training.ipynb
│   └── 05_analysis_and_results.ipynb
├── scripts/
│   ├── train.sh                # CLI training launcher
│   └── record_gameplay.sh
└── tests/
    ├── test_env.py
    ├── test_models.py
    └── test_agents.py
```

### 1.2 Dependencies (`pyproject.toml`)  ✓
- **Core:** `vizdoom>=1.2`, `gymnasium`, `torch>=2.0`, `numpy`
- **Utilities:** `opencv-python`, `pyyaml`, `tensorboard`, `matplotlib`
- **Notebooks:** `jupyter`, `ipywidgets`
- **Dev:** `pytest`, `ruff`, `pre-commit`

### 1.3 Initial Files  ✓
- `README.md` — project overview, setup instructions, quickstart
- `.gitignore` — Python, Jupyter, model checkpoints, logs
- `pyproject.toml` — PEP 621 project definition

---

## Phase 2: Environment Wrappers (`src/rl_doom/env.py`)  ✓

Wrap ViZDoom to comply with the Gymnasium API so standard RL libraries work seamlessly.

### Key wrappers (composable):
| Wrapper | Purpose |
|---------|---------|
| `DoomEnv` | Base Gymnasium wrapper around ViZDoom |
| `FrameStack` | Stack N grayscale frames (temporal info) |
| `ResizeObservation` | Resize frames to 84×84 or 64×64 |
| `SkipFrame` | Action repeat / frame skipping |

### Scenarios to support (built into ViZDoom):
1. **Basic** — single room, shoot a monster
2. **Deadly Corridor** — navigate corridor with enemies
3. **Defend the Center** — survive waves of enemies
4. **Deathmatch** — full combat scenario

---

## Phase 3: Models (`src/rl_doom/models.py`)  ✓

### 3.1 CNN Feature Extractor
```
Conv2d(in, 32, 8, stride=4) → ReLU
Conv2d(32, 64, 4, stride=2) → ReLU
Conv2d(64, 64, 3, stride=1) → ReLU
Flatten → Linear(3136, 512) → ReLU
```
Standard Nature-DQN architecture, proven for Doom-scale visual inputs.

### 3.2 DQN Head
- `Linear(512, n_actions)` — Q-value for each action

### 3.3 PPO Actor-Critic Head
- **Actor:** `Linear(512, n_actions)` → action distribution
- **Critic:** `Linear(512, 1)` → state value estimate

---

## Phase 4: Agents  ✓

### 4.1 DQN Agent (`src/rl_doom/agents/dqn.py`)
- Epsilon-greedy exploration (decay schedule)
- Experience replay buffer (uniform)
- Target network with periodic hard updates
- Double DQN variant to reduce overestimation
- Huber loss

### 4.2 PPO Agent (`src/rl_doom/agents/ppo.py`)
- Clipped surrogate objective (clip_eps=0.2)
- GAE (Generalized Advantage Estimation, lambda=0.95)
- Mini-batch updates over collected rollouts
- Entropy bonus for exploration
- Value function loss
- Gradient norm clipping

---

## Phase 5: Training Infrastructure

> **Status:** Notebooks contain inline training loops (see Phase 6).
> A standalone `src/rl_doom/train.py` CLI entry point is not yet implemented.

- YAML-driven config (scenario, hyperparams, agent type)
- TensorBoard logging (rewards, loss, epsilon, FPS)
- Periodic checkpoint saving (model + optimizer state)
- Resume from checkpoint support
- Seed control for reproducibility
- CLI via `python -m rl_doom.train --config configs/dqn_basic.yaml`

---

## Phase 6: Notebooks  ✓

### `01_environment_exploration.ipynb`
- Install/verify ViZDoom
- Visualize observations, rewards, and available actions
- Test wrappers (frame stacking, resizing)
- Random agent baseline performance

### `02_dqn_training.ipynb`
- Train DQN on "Basic" scenario
- Plot learning curves (reward, loss, epsilon)
- Hyperparameter sensitivity exploration
- Render trained agent gameplay (embedded video)

### `03_ppo_training.ipynb`
- Train PPO on "Deadly Corridor" and "Defend the Center"
- Compare sample efficiency vs DQN
- Visualize policy entropy and value loss over time

### `04_recurrent_ppo_training.ipynb`
- Train Recurrent PPO (`sb3_contrib.RecurrentPPO`, CnnLstmPolicy) on the same four scenarios
- LSTM hidden state on top of NatureCNN features for partial observability
- Paired with PPO configs so the LSTM-vs-stacked-frames comparison is apples-to-apples

### `05_analysis_and_results.ipynb`
- Cross-scenario comparison table covering DQN, PPO, and Recurrent PPO
- Three-way head-to-head training curves
- Statistical analysis (mean ± std over N seeds)
- Qualitative analysis: what the agent learned (saliency maps)
- GIF/video generation of best runs

---

## Phase 7: Evaluation & Recording (`src/rl_doom/evaluate.py`)  ✓

- Load trained checkpoint and run episodes
- Record gameplay as frames (via ViZDoom screen buffer)
- Compute aggregate stats (mean reward per episode)
- GIF generation for notebooks and README

---

## Phase 8: Testing

> **Status:** Not yet implemented.

- `test_env.py` — env creation, reset, step, observation shapes
- `test_models.py` — forward pass shapes, parameter counts
- `test_agents.py` — action selection, single training step

---

## Recommended Implementation Order

| Step | What | Priority | Status |
|------|------|----------|--------|
| 1 | Project scaffolding (pyproject.toml, .gitignore, README) | **P0** | ✓ Done |
| 2 | Environment wrappers + Notebook 01 | **P0** | ✓ Done |
| 3 | CNN model + DQN agent | **P0** | ✓ Done |
| 4 | Training loop + configs | **P0** | Partial (notebooks only) |
| 5 | Notebook 02 (DQN training) | **P1** | ✓ Done |
| 6 | PPO agent | **P1** | ✓ Done |
| 7 | Notebook 03 (PPO training) | **P1** | ✓ Done |
| 8 | Evaluation/recording tools | **P1** | ✓ Done |
| 9 | Recurrent PPO (sb3-contrib) + Notebook 04 | **P1** | ✓ Done |
| 10 | Notebook 05 (analysis) | **P2** | ✓ Done |
| 11 | Tests | **P2** | Not started |
| 11 | Advanced: Prioritized replay, Dueling DQN, curiosity | **P3** | Not started |

---

## Still TODO

- `configs/*.yaml` — YAML config files for CLI training
- `src/rl_doom/train.py` — standalone CLI training entry point
- `scripts/train.sh`, `scripts/record_gameplay.sh` — shell launchers
- `tests/` — unit tests for env, models, agents
- `RewardShaping` wrapper (listed in original plan, not yet needed by notebooks)

---

## Future Improvements (Phase 9+)

- **Curiosity-driven exploration** (ICM/RND) for sparse-reward scenarios
- **Dueling DQN** architecture
- **Prioritized Experience Replay**
- **Multi-agent / self-play** in deathmatch
- **Hyperparameter search** via Optuna
- **Stable-Baselines3 integration** as a comparison baseline
- **Weights & Biases** logging as an alternative to TensorBoard
- **Docker container** for reproducible training environment
- **GitHub Actions CI** — linting + tests on push
