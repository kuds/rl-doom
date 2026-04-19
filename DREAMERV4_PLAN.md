# DreamerV4 on ViZDoom — Integration Plan

Branch: `claude/dreamerv4-doom-exploration-P2avL`

Exploration-only document. No code changes yet. Goal: decide whether adding a
DreamerV4 agent to `rl-doom` is worth the complexity, and if so, pin down the
smallest concrete slice that proves it works on one scenario before committing
to full integration.

---

## 1. Why DreamerV4 fits ViZDoom

- **Pixel obs, discrete actions** — the regime DreamerV3/V4 are tuned for
  (Atari-100k, DMLab, Crafter, Minecraft).
- **Partial observability** — the RSSM's recurrent latent subsumes what
  `FrameStack(num_stack=4)` and `RecurrentPPO`'s LSTM try to do, and handles
  longer-horizon dependencies (enemies leaving FOV in `deadly_corridor`,
  map memory in `deathmatch`).
- **Sample efficiency** — the harder scenarios in this repo already run to
  1–2M env steps on PPO. Dreamer's reported gains on DMLab/Crafter suggest
  getting comparable or better returns at a fraction of the interaction
  budget, which matters because each ViZDoom step drives a native process.
- **Reward sparsity** — `deadly_corridor` is death-penalty dominated;
  Dreamer's symlog/two-hot reward head is well-behaved on heavy-tailed,
  sparse reward distributions where PPO needs entropy-coef tuning to avoid
  collapse.

## 2. Tradeoffs / open risks

- **Complexity**: RSSM world model + imagination rollouts + symlog targets is
  a much bigger surface than the current SB3-wrapped DQN/PPO agents. A new
  `agents/dreamer.py` will be the largest file in `src/rl_doom/`.
- **Implementation source**: no SB3-native Dreamer. The pragmatic options are
  wrapping a reference impl (danijar/dreamerv3 in JAX, or a PyTorch port such
  as NM512/dreamerv3-torch) rather than a clean-room port. That adds a new
  framework to the repo unless we pick the PyTorch port.
- **Compute**: needs a GPU with ≥8 GB VRAM for default sizes. Colab T4 is
  OK at `small` config; full `xlarge` is out of scope.
- **Throughput**: Dreamer trains on sequences from a replay buffer, with
  ~1 gradient step per env step at default ratios. Wall-clock will likely be
  slower per env step than PPO's 8-env rollouts even though total steps are
  fewer. This should be measured, not assumed.
- **Action space mismatch**: the curated compound actions in
  `SCENARIO_ACTION_SETS` (see `src/rl_doom/env.py:41`) are already discrete
  integers — Dreamer's categorical actor consumes this directly, so no env
  changes needed for the action side.

## 3. Scope decision — start small

Pick **one** scenario for the proof-of-concept:

- **Recommended: `defend_the_center`** (200k step budget, small action set of
  5, stationary agent → the world model only needs to learn enemy dynamics,
  not navigation). Fastest to validate that the pipeline works end-to-end.
- **Stretch: `deadly_corridor`** once PoC is green — this is where the
  memory/partial-observability advantage should actually show vs PPO.

Do **not** start with `deathmatch`. Long episodes, multiple bots, and large
maps will confound "did the integration work?" with "is Dreamer tuned?".

## 4. Implementation slice

Minimum viable integration:

1. **Dependency.** Add `dreamerv3-torch` (PyTorch port) as an optional extra
   in `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   dreamer = ["dreamerv3-torch @ git+https://github.com/NM512/dreamerv3-torch@<pinned-sha>"]
   ```
   Rationale: stays in the existing PyTorch stack, avoids pulling JAX/XLA
   into the repo. Pin to a commit SHA — this project is research code and
   master moves.

2. **Env adapter.** `src/rl_doom/env.py` already produces `(num_stack, H, W)`
   uint8 frames. Dreamer wants `(H, W, C)` uint8 in `[0, 255]` and a discrete
   action int. Add a thin wrapper (either in `env.py` or a new
   `src/rl_doom/dreamer_env.py`) that:
   - Disables `FrameStack` (RSSM handles memory).
   - Keeps `ResizeObservation` at 64×64 (Dreamer default) instead of 84×84.
   - Returns `(64, 64, 3)` RGB — skip the grayscale step in
     `ResizeObservation` by adding a `grayscale: bool = True` kwarg, default
     preserves current behavior.
   - Leaves `SkipFrame` in place (frame_skip=4 matches Dreamer Atari
     defaults).

3. **Agent wrapper.** `src/rl_doom/agents/dreamer.py`:
   - Thin class that owns a `dreamerv3_torch.Dreamer` instance.
   - Exposes `train(env, total_timesteps, ...)` matching the shape of
     `PPOAgent.train` / `DQNAgent.train` so `src/rl_doom/train.py` can
     dispatch on `algorithm: dreamer`.
   - Forwards TensorBoard logging to the same `runs/` dir the other agents
     use, so the analysis notebook (`05_analysis_and_results.ipynb`) can pick
     it up without changes.

4. **Config.** `configs/dreamer_defend_the_center.yaml`, cloning the shape of
   `configs/ppo_defend_the_center.yaml`:
   ```yaml
   scenario: defend_the_center
   algorithm: dreamer
   seed: 42

   env:
     resize_shape: [64, 64]
     frame_skip: 4
     num_stack: 1               # RSSM replaces frame stacking
     grayscale: false           # Dreamer consumes RGB
     n_envs: 1                  # Dreamer trains single-env by default

   hyperparams:
     preset: small              # dreamerv3-torch config preset
     batch_size: 16
     batch_length: 64
     train_ratio: 512           # replay:env step ratio
     imag_horizon: 15

   training:
     total_timesteps: 200000
     checkpoint_freq: 50000

   eval:
     eval_freq: 10000
     n_episodes: 5
   ```

5. **Train dispatch.** In `src/rl_doom/train.py`, add an `elif algorithm ==
   "dreamer"` branch. Keep DQN/PPO/RecurrentPPO paths untouched.

6. **Smoke test.** `tests/test_dreamer_smoke.py`: build the wrapped env, run
   ~1000 env steps with the agent, assert no exception + checkpoint writes.
   Do **not** assert reward — that's what the notebook is for.

## 5. What stays out of scope for the PoC

- No notebook (`06_dreamer_training.ipynb`) until the CLI path trains a
  non-trivial policy.
- No multi-seed sweep, no saliency, no cross-algo comparison table updates.
- No `deathmatch` / self-play / bots integration. Those already push the
  existing agents hard; they'd muddy the Dreamer signal.
- No custom RSSM / symlog surgery. Use the port's defaults.

## 6. Success criteria for the PoC

In order:

1. **Correctness**: agent runs for 200k env steps on `defend_the_center`
   without shape/dtype errors, checkpoints load.
2. **Signal**: eval return rises above the random baseline captured in
   `notebooks/01_environment_exploration.ipynb`.
3. **Parity**: eval return reaches within ~1 std of PPO's final score on the
   same scenario, measured over 3 seeds. (DQN/PPO on `defend_the_center`
   already converge fast, so this is a reasonable bar.)
4. **Efficiency claim check**: record env-steps-to-threshold vs PPO. If
   Dreamer isn't more sample-efficient on *this* scenario, it's unlikely to
   pay off on the harder ones either — reassess before investing in
   `deadly_corridor` / `deathmatch`.

Only after (3) is hit do we plan the follow-up work (harder scenarios,
notebook, analysis integration).

## 7. Open questions to resolve before writing code

- Which PyTorch port to pin? `NM512/dreamerv3-torch` is the most starred,
  but last-commit date / test coverage need checking. If it's unmaintained,
  vendoring a minimal RSSM implementation might end up smaller than the
  adapter code needed to bend someone else's training loop.
- Does the port's `Dreamer.train` own its own env loop, or can we feed it
  transitions from our existing env? If it owns the loop we give up the
  `DummyVecEnv` / `Monitor` plumbing and need a parallel logging path.
- GPU availability assumption: is the target hardware Colab T4 or local
  workstation? Affects which size preset (`small` vs `medium`) to default
  to in the config.
