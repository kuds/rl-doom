# DreamerV3 on ViZDoom — Integration Plan

Branch: `claude/dreamerv4-doom-exploration-P2avL` (branch name predates the
V3 vs V4 decision — kept to avoid churn)

## TL;DR

Start with **DreamerV3** via the `NM512/dreamerv3-torch` PyTorch port, on the
`defend_the_center` scenario, as a proof-of-concept. Only escalate to V4 or
harder scenarios if the V3 PoC hits its success criteria.

## 1. V3 vs V4 — why V3 first

- **Reference code maturity**: V4's official release is JAX
  (`danijar/dreamerv4`). The mature, maintained PyTorch ports are all V3
  (`NM512/dreamerv3-torch`, `jurgisp/dreamerv3-pytorch`, etc.).
  Using V4 here means adding JAX to the dep stack or porting ourselves —
  a much bigger lift than wrapping an existing V3 port.
- **Fit to this problem**: V4's headline contributions (transformer world
  model, shortcut forcing, strong *offline* learning on Minecraft) matter
  most for very-long-horizon or offline regimes. The Doom scenarios here
  are online, ≤2M steps, moderate horizons — exactly the regime V3 is
  tuned for and benchmarked on (Atari, DMLab, Crafter, Minecraft online).
- **Risk**: V4 is newer, fewer independent reproductions, more surface
  area to debug against novel envs.
- **Escalation trigger**: reach for V4 only if the V3 PoC clearly works
  and we hit a ceiling that V4's architecture specifically addresses.

## 2. Why Dreamer fits ViZDoom at all

- **Pixel obs, discrete actions** — the regime DreamerV3 is tuned for.
- **Partial observability** — the RSSM's recurrent latent subsumes what
  `FrameStack(num_stack=4)` and `RecurrentPPO`'s LSTM try to do, and
  handles longer-horizon dependencies (enemies leaving FOV in
  `deadly_corridor`, map memory in `deathmatch`).
- **Sample efficiency** — the harder scenarios here run to 1–2M env steps
  on PPO. Dreamer's reported gains on DMLab/Crafter suggest comparable
  or better returns at a fraction of the interaction budget, which
  matters because each ViZDoom step drives a native process.
- **Reward sparsity** — `deadly_corridor` is death-penalty dominated;
  Dreamer's symlog/two-hot reward head is well-behaved on heavy-tailed,
  sparse reward distributions where PPO needs entropy-coef tuning.
- **Action space**: the curated compound actions in
  `SCENARIO_ACTION_SETS` (`src/rl_doom/env.py:41`) are already discrete
  integers — Dreamer's categorical actor consumes this directly.

## 3. What we learned about the V3 port (NM512/dreamerv3-torch)

Checked upstream before planning the adapter. Key facts that shape the
integration:

- **Entry point**: `dreamer.py` with `main(config)`. CLI form is
  `python3 dreamer.py --configs <preset> --task <suite_task>`.
- **Agent class**: `Dreamer(obs_space, act_space, config, logger, dataset)`,
  an `nn.Module`.
- **Env construction**: `make_env(config, mode, id)` builds env per suite
  (DMC, Atari, DMLab, MemoryMaze, Crafter, Minecraft). **Doom is not a
  supported suite** — we provide our own env directly, bypassing
  `make_env`.
- **Training loop**: Dreamer does **not** call `env.step()` itself. The
  top level calls `tools.simulate(agent, train_envs, train_eps, ...)`
  which owns env interaction. We can reuse `tools.simulate()` with our
  env.
- **Expected observation format** (from `envs/atari.py`):
  - Dict space: `{"image": Box(0, 255, (H, W, C), uint8)}` plus
    `is_first`, `is_terminal` keys returned from `reset`/`step`.
  - `step(action)` returns `(obs_dict, reward, is_last, info)` — the
    older 4-tuple gym API, not Gymnasium's 5-tuple.
  - `reset()` returns a single obs dict (not `(obs, info)`).
  - Dreamer's internal `OneHotAction` wrapper converts discrete actions
    to one-hot; we can keep discrete and let it wrap.
- **Config presets available**: `dmc_proprio`, `dmc_vision`, `crafter`,
  `atari100k`, `minecraft`, `memorymaze`, `debug`. `atari100k` is the
  closest starting point architecturally (pixel-based, discrete actions).
- **Key config knobs**: `size`, `action_repeat`, `steps`, `grayscale`,
  `batch_size`, `batch_length`, `train_ratio`, `imag_horizon`.

## 4. Scope — one scenario, one algo

- **PoC scenario**: `defend_the_center` (200k step budget, 5-action
  space, stationary agent → world model only needs to learn enemy
  dynamics, not navigation). Fastest path to validate the pipeline
  end-to-end.
- **Stretch once green**: `deadly_corridor` — where the
  memory/partial-observability advantage should actually show vs PPO.
- **Out of scope**: `deathmatch`, self-play/bots, multi-seed sweeps,
  saliency, notebook. Those muddy the "did the integration work?"
  signal.

## 5. Implementation slice

1. **Dependency** (optional extra in `pyproject.toml`):
   ```toml
   [project.optional-dependencies]
   dreamer = [
     "dreamerv3-torch @ git+https://github.com/NM512/dreamerv3-torch@<pinned-sha>",
   ]
   ```
   Pin to a commit SHA — the upstream is research code and `main` moves.
   (Pin TBD on first install.)

2. **Env adapter** — `src/rl_doom/dreamer_env.py`:
   - Wraps our existing `DoomEnv` (keeps `SCENARIO_ACTION_SETS` discrete
     action space).
   - Converts to the V3 port's expected format:
     - Dict observation space with `image` key, shape `(64, 64, 3)` RGB
       (or `(64, 64, 1)` if grayscale).
     - `reset()` returns a single obs dict (not Gymnasium's
       `(obs, info)`).
     - `step()` returns `(obs, reward, is_last, info)` 4-tuple.
     - `is_first`/`is_terminal` flags in the obs dict.
   - Skips our `FrameStack` (RSSM replaces frame stacking) and bypasses
     the grayscale step in `ResizeObservation` (Dreamer consumes RGB by
     default; we expose a `grayscale` kwarg to match the port's config).
   - Keeps `SkipFrame` (frame_skip=4 matches the port's atari defaults).

3. **Agent wrapper** — `src/rl_doom/agents/dreamer.py`:
   - Thin driver around `dreamerv3_torch.Dreamer` + `tools.simulate()`.
   - Builds the port's config object from our YAML.
   - Owns logging into the same `runs/` dir the other agents use.
   - Exposes a `train(total_timesteps, ...)` method matching the shape
     of `PPOAgent`/`DQNAgent`, so the train dispatcher pattern stays
     uniform.

4. **Config** — `configs/dreamer_defend_the_center.yaml`:
   ```yaml
   scenario: defend_the_center
   algorithm: dreamer
   seed: 42

   env:
     resize_shape: [64, 64]
     frame_skip: 4
     grayscale: false           # Dreamer consumes RGB by default
     n_envs: 1
     doom_skill: 3
     num_bots: 0

   hyperparams:
     preset: atari100k          # dreamerv3-torch preset to inherit from
     batch_size: 16
     batch_length: 64
     train_ratio: 512
     imag_horizon: 15

   training:
     total_timesteps: 200000
     checkpoint_freq: 50000

   eval:
     eval_freq: 10000
     n_episodes: 5
   ```

5. **Train dispatch**: the repo currently drives training from notebooks
   rather than a shared `train.py` CLI (despite the README's example).
   For the V3 PoC we'll invoke the agent from a small CLI script
   `scripts/train_dreamer.py` that loads the YAML and calls
   `DreamerAgent.train`. Adding a unified dispatcher is a separate
   refactor and should not be coupled to this work.

6. **Smoke test** — `tests/test_dreamer_env.py`:
   - Build the wrapped env, check obs-dict shape/dtype/keys.
   - Run ~10 env steps, assert no exception.
   - Does **not** import `dreamerv3_torch` — just validates the adapter.

## 6. What stays out of scope for the PoC

- No `notebooks/06_dreamer_training.ipynb` until CLI trains a
  non-trivial policy.
- No multi-seed, no saliency, no cross-algo comparison table updates.
- No `deathmatch` / self-play / bots integration.
- No custom RSSM / symlog surgery. Port defaults only.

## 7. Success criteria

In order:

1. **Correctness**: agent runs for 200k env steps on `defend_the_center`
   without shape/dtype errors; checkpoints save and load.
2. **Signal**: eval return rises above the random baseline captured in
   `notebooks/01_environment_exploration.ipynb`.
3. **Parity**: eval return reaches within ~1 std of PPO's final score
   on the same scenario, measured over 3 seeds.
4. **Efficiency check**: record env-steps-to-threshold vs PPO. If
   Dreamer isn't more sample-efficient on *this* scenario, it's
   unlikely to pay off on the harder ones either — reassess before
   `deadly_corridor`.

Only after (3) is hit do we plan the follow-up work (harder scenarios,
notebook, analysis integration, optional V4 migration).

## 8. Open questions still to resolve during implementation

- Upstream pin: pick the latest stable commit on first `pip install`
  and record it in `pyproject.toml`.
- GPU target: Colab T4 (≈16 GB VRAM) is the default assumption — set
  `atari100k` preset sizes accordingly. Revisit if targeting weaker hw.
- `tools.simulate` vs writing our own loop: using the port's function
  gives us their replay-dataset plumbing for free; the cost is
  conforming to its env interface exactly (what the adapter in §5.2 is
  for).
