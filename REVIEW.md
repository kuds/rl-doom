# RL Pipeline Review

A structural, correctness, and cleanup review of the `rl-doom` reinforcement-learning
pipeline. Every claim below was verified against the source (and, where noted, by
executing the code). Line references are to the state of `main` at the time of review.

---

## Status: Wave 1 is implemented

All ten Wave 1 items have landed. Findings 1.1, 1.3, 1.4, 1.5, 1.6, 1.10, 2.4, 4.1, 4.3,
5.1, 5.2 and 5.3 are **fixed**; the rest stand as written. The suite went from 126 tests
(114 in CI) to **261**, and coverage from 48% (44% in CI) to **67%**.

Three of this review's own recommendations were wrong, and implementation caught them:

- **1.5 said to pass `optimize_memory_usage=True`.** SB3 2.9 raises if that is combined
  with `handle_timeout_termination=True` — which is exactly what makes the 1.1 truncation
  fix work for DQN. The two Wave 1 items were in direct conflict. Fixed instead by sizing
  the buffers to the target hardware and adding a pre-allocation guard.
- **5.1 said to build a `fake_vizdoom` stub for `DoomEnv`.** Unnecessary: ViZDoom now ships
  manylinux wheels for cp39–cp314, so CI just installs it and `test_env.py` runs against
  the real binary. Less code, higher fidelity. The premise that CI "cannot install vizdoom"
  was simply out of date.
- **5.2 undercounted.** The missing `sb3-contrib` did not only skip the Recurrent PPO test —
  the `importorskip` was module-level and mid-file, so it dropped nine unrelated DQN/PPO
  tests with it.

Implementation also turned up things the review missed: the live 1v1 multiplayer test
segfaults rather than failing (pre-existing, now opt-in via `RL_DOOM_MULTIPLAYER_TESTS=1`);
`test_reset_clears_bots_before_new_episode` had never run and was broken against the real
pybind11 `DoomGame`; and the Dreamer budget mismatch in 1.10 was deliberate rather than
accidental, so the baselines were raised to match rather than the curriculum arms cut.

Each fixed finding is marked **[FIXED]** below with the commit that closed it.

---

## Executive summary

This is a well-above-average research repo. The artifact discipline is genuinely good —
`paths.py` owns a real run layout, every run drops `config.json` + `metrics/` +
`figures/` + `stage_summary.txt`, `summary.py` discovers runs via structured metadata
rather than filename parsing, and the comment quality throughout is unusually high
(comments explain *why*, not *what*). The curriculum subsystem is properly factored into
a policy-free controller plus an SB3 callback, and the DreamerV3 integration is a careful
piece of work.

The four most important problems are:

1. **A value-function correctness bug affecting DQN, PPO, and Recurrent PPO** (1.1).
   ViZDoom's episode timeout is reported as `terminated=True, truncated=False`, so every
   timeout is bootstrapped as an absorbing state. The Dreamer path already *fixes* this and
   documents exactly why (`dreamer_env.py:147-157`); the SB3 path — 3 of 4 algorithms and
   the entire flagship curriculum experiment — never got the fix.
2. **The DQN configs can't run on the hardware the repo targets** (1.5). SB3's replay
   buffer stores `obs` and `next_obs` separately unless `optimize_memory_usage=True`, which
   is never set. `dqn_deathmatch.yaml` therefore asks for **16.9 GB** of replay RAM and
   `dqn_deadly_corridor.yaml` for **11.3 GB**, against Colab's ~12.7 GB.
3. **No crash resilience for multi-hour runs** (4.1–4.3). `model.learn()` is unguarded, so
   a failure at step 2.4M of 2.5M loses the model, metrics, figures, and video.
   `vec_normalize.pkl` is saved "so a future resumed run can pick up" but nothing loads it,
   and the periodic checkpoint `.zip`s are never read by anything. The 24-run matrix runner
   contains zero `try:` statements and writes its summary CSV only after the last run.
4. **CI is quieter than it looks** (5.1–5.2). Reproducing CI's exact install, the suite
   runs **114 passed, 4 skipped** against 126 locally — three whole test modules are
   silently disabled, `env.py` (571 lines, every wrapper in the pipeline) sits at **0%
   coverage**, and total coverage is **44%**. One cause is a module-level
   `importorskip("sb3_contrib")` placed mid-file, which drops nine unrelated DQN/PPO tests
   along with it. The stub pattern that would fix the env half already exists in
   `tests/conftest.py`.

The single highest-leverage change is **item 1** — a ~10-line fix in `DoomEnv.step`, but it
silently biases every reward number the repo reports.

There is also a plain usability break worth fixing first because it is trivial: in five of
the six notebooks the "Colab Setup" cell says *"Uncomment the block below when running on
Google Colab"* directly above code that is **already uncommented** (2.4). Run locally, they
`os.chdir("/content/rl-doom/notebooks")` and `from google.colab import drive`. Together with
the missing `python -m rl_doom.train` CLI (2.3), the repo currently has no working local path
to train anything.

Three findings are worth separating out because they invalidate *results* rather than code:
the Dreamer curriculum configs give the curriculum arm 1.5–2× its baseline's budget (1.10);
the deathmatch curricula silently discard their configured `doom_skill` (1.6); and the
analysis notebook evaluates every agent at the wrong difficulty with no seeding, and cannot
load its Recurrent PPO runs at all (1.8, 1.4). The headline "does the curriculum help?"
comparison is confounded on two of its four algorithm rows.

Beyond correctness, the repo carries meaningful weight it isn't using: `utils.py` is 199
lines with zero references anywhere; `self_play.py` + `tournament.py` are 1,030 lines with
735 lines of tests and no production caller; and `sb3_utils.py` has grown into a
1,059-line module with six unrelated responsibilities behind one 361-line function.

A pattern worth naming, because it recurs: **several comments promise behaviour the code
does not implement.** `vec_normalize.pkl` is saved "so a future resumed run can pick up"
(no resume exists, 4.2); `_pick_free_port`'s docstring says a collision costs "one retry at
env construction" (no retry exists, 4.5); `setup_google_drive` claims to be "a no-op" off
Colab (it raises, 2.7); the curriculum configs claim a guard against a single lucky eval
(there is none, 1.9). Each reads as done during review. Treating a comment that describes
a capability as a TODO until a test covers it would catch this whole class.

---

## What's working well

- **Run artifacts are first-class.** `paths.py` centralises `training_jobs/<env>/<algo>/runs/<id>/`,
  handles the Colab/Drive no-symlink case with a text-pointer fallback (`paths.py:106-135`),
  and `summary.py:397` discovers runs by looking for `config.json` — structured metadata,
  not brittle name parsing.
- **`train_sb3` is self-contained per run** — scenario N is fully shippable before N+1 starts.
- **Curriculum design is clean.** The stage controller (`curriculum.py:180-260`) is pure
  logic with no SB3 dependency; the callback is a thin adapter. That separation is correct.
- **The Dreamer integration is thoughtful** — an upstream commit pin (`UPSTREAM_PIN`)
  shared between notebook and wrapper, deliberate avoidance of the upstream's `gym`/`ruamel`
  dependencies, and a clear rationale in `pyproject.toml`.
- **Comments carry real reasoning** — e.g. why `DummyVecEnv` over `SubprocVecEnv`
  (`env.py:519-523`), why compound actions exist (`env.py:55-65`), why `removebots` is
  needed before `new_episode` (`env.py:291-295`). This is rare and valuable.
- **Ruff and mypy both pass**, and the suite runs in 8 seconds.

---

## Findings

### 1. Correctness & RL semantics

#### 1.1 Episode timeouts are bootstrapped as absorbing states for DQN/PPO/RecurrentPPO  `critical` · `small`  **[FIXED]**

> **Fixed** in `f1cd729` — Report scenario timeouts as truncation, not termination.

**Where:** `src/rl_doom/env.py:328`, `env.py:330-343`; contrast `src/rl_doom/dreamer_env.py:147-157`

**What:** `DoomEnv.step` returns `return obs, reward, terminated, False, info` — `truncated`
is hard-coded to `False`. `_classify_termination` *computes* that the episode ended by
timeout and records it in `info["termination_reason"]`, then that distinction is thrown
away at the Gymnasium boundary. `SkipFrame` and `FrameStack` pass `truncated` through
faithfully, so it is `False` all the way up to SB3.

The Dreamer path already recognised and fixed this, with a comment that states the problem
exactly:

```python
# DoomEnv folds ViZDoom's scenario timeout into ``terminated=True`` and never
# sets ``truncated``, so we have to consult ``info["termination_reason"]`` to
# recover the distinction — otherwise the world model learns that running out
# the clock is an absorbing state.
if terminated and info.get("termination_reason") == "timeout":
    is_terminal = False
```

**Why it matters:** SB3 bootstraps `V(s')` only when `truncated` is true. With
`truncated=False`, DQN's Bellman target becomes `r` instead of `r + γ·V(s')`, and PPO/GAE's
`next_non_terminal` is zeroed. The agent is taught that surviving to the clock limit is
worth zero future return. This is not a corner case: per `BENCHMARKS.md`, deadly_corridor
and defend_the_center time out at 525 env-steps and deathmatch at 1050, and a
deadly_corridor policy that survives but doesn't reach the vest times out *by definition*.
Every eval number for DQN, PPO, and Recurrent PPO — including the entire
`configs/matrix/deadly_corridor_curriculum.yaml` experiment — is produced under this bias.

**Fix:** In `DoomEnv.step`, classify on termination and split the flag:

```python
terminated = self.game.is_episode_finished()
truncated = False
if terminated:
    info["termination_reason"] = self._classify_termination()
    if info["termination_reason"] == "timeout":
        terminated, truncated = False, True
```

Two things must land with it, or the fix makes matters worse:
- `_get_obs` returns a **black frame** when `game.get_state()` is `None` (`env.py:279-280`).
  Once `truncated=True`, SB3 will bootstrap `V(s')` from that black frame. Cache the last
  real frame and return it on the terminal step instead.
- `SkipFrame.step` (`env.py:399-408`) must keep propagating `truncated` — it already does.

Add a unit test asserting `truncated is True` when `termination_reason == "timeout"`. It
should use the `fake_vizdoom` stub (see 5.1) so it runs in CI.

---

#### 1.2 The uint8 normalization guard never fires — hand-rolled agents train on 0–255 inputs  `high` · `small`

**Where:** `src/rl_doom/models.py:36-38`; callers at `agents/dqn.py:66,80,83`, `agents/ppo.py:83,148`

**What:** `_CNNBase.forward` normalizes conditionally:

```python
if x.dtype == torch.uint8:
    x = x.float() / 255.0
```

Every call site converts first: `torch.FloatTensor(np.array(obs))`. By the time the tensor
reaches the network its dtype is already `float32`, so the branch is dead and the CNN
receives raw 0–255 values.

**Verified by execution:**

```
uint8 tensor  -> feature abs-mean: 0.0109
float tensor  -> feature abs-mean: 2.0362      (~186x)

DQNAgent.select_action -> net input dtype: torch.float32  max: 255.0
DQNAgent.train_step    -> net input dtype: torch.float32  max: 255.0
PPOAgent.select_action -> net input dtype: torch.float32  max: 255.0
```

**Why it matters:** A Nature-DQN conv stack fed unnormalized pixels produces ~186× larger
activations than intended. The code *looks* correct, which is what makes it dangerous —
anyone reusing `models.py` inherits the bug silently.

**Scope:** this affects only the legacy hand-rolled stack (see 2.5), not the SB3 training
path — SB3's `NatureCNN` normalizes internally. But `evaluate.py` still loads legacy
checkpoints through these classes, and `models.py` is the module a reader would copy.

**Fix:** Normalize unconditionally in `_CNNBase.forward` (`x = x.float() / 255.0`), or
better, drop the conditional and have callers pass `torch.as_tensor(obs)` preserving
`uint8`. Add an assertion test that the network's first-layer input is in `[0, 1]`.

---

#### 1.3 The deathmatch action table is maintained twice, with different length and ordering  `high` · `small`  **[FIXED]**

> **Fixed** in `d7fd2ad` — Source both deathmatch action tables from one definition.

**Where:** `src/rl_doom/env.py:104-130` (16 actions) vs `src/rl_doom/multiplayer_env.py:40` (14 actions)

**What:** Verified by diffing the two tables:

| idx | `env.py` | `multiplayer_env.py` | |
|----:|---|---|---|
| 0–7 | *(identical)* | *(identical)* | OK |
| 8 | `MOVE_LEFT+ATTACK` | `MOVE_FORWARD+TURN_LEFT` | **differ** |
| 9 | `MOVE_RIGHT+ATTACK` | `MOVE_FORWARD+TURN_RIGHT` | **differ** |
| 10–11 | *(identical)* | *(identical)* | OK |
| 12 | `MOVE_FORWARD+TURN_LEFT` | `MOVE_LEFT+ATTACK` | **differ** |
| 13 | `MOVE_FORWARD+TURN_RIGHT` | `MOVE_RIGHT+ATTACK` | **differ** |
| 14 | `SPEED+MOVE_FORWARD` | — | **missing** |
| 15 | `SPEED+MOVE_FORWARD+ATTACK` | — | **missing** |

**Why it matters:** the single-player deathmatch env exposes `Discrete(16)`;
`multiplayer_env` builds `Discrete(14)` from its own table (`multiplayer_env.py:168,197`).
An SB3 `.zip` trained on `configs/ppo_deathmatch.yaml` therefore cannot be loaded as a
self-play opponent snapshot — the action-space check fails. And if it were forced through,
indices 8/9/12/13 mean *different buttons*, so the policy would strafe when it meant to
turn. This breaks the single-player → self-play transfer that `self_play.py` and
`tournament.py` exist to support.

**Fix:** Delete `DEATHMATCH_ACTIONS` and have `multiplayer_env` import
`SCENARIO_ACTION_SETS["deathmatch"]` from `env.py`. Add a test asserting the multiplayer
action space matches `make_wrapped_env("deathmatch").action_space`.

---

#### 1.4 Algorithm → SB3 class dispatch exists in three places; the analysis notebook cannot load its own Recurrent PPO runs  `high` · `small`  **[FIXED]**

> **Fixed** in `df3db6b` — Dispatch algorithm classes through one registry.

**Where:** `src/rl_doom/sb3_utils.py:114-167` (authoritative), `sb3_utils.py:885-893` (video loader), `src/rl_doom/evaluate.py:216` (`load_run`)

**What:** `_build_model` dispatches correctly and **raises** on an unknown algo
(`sb3_utils.py:166`). Two other sites re-implement it and both have a silent catch-all:

```python
# sb3_utils.py:885-893 — video loader
if algo_norm == "recurrent_ppo":  cls = RecurrentPPO
elif algo_norm == "ppo":          cls = PPO
else:                             cls = DQN      # <-- silent catch-all

# evaluate.py:216 — load_run
algo = json.loads((run_dir / "config.json").read_text()).get("algo", "ppo")
cls = PPO if algo.lower() == "ppo" else DQN      # <-- RecurrentPPO -> DQN
```

**Why it matters:** this is not latent. Notebook 05 — README's step 5, the
cross-algorithm comparison — sets `ALGOS = ["dqn", "ppo", "recurrent_ppo"]` and loads every
run through `load_run`. A `recurrent_ppo` run's `config.json` reports
`algo="recurrent_ppo"`, so `load_run` calls `DQN.load()` on a `CnnLstmPolicy` zip and
fails. **One third of the analysis notebook's runs cannot be loaded at all.** The
`sb3_utils.py:885` copy fails differently and worse — it silently loads the wrong class at
the *end* of a multi-hour training run, after training succeeded.

**Fix:** One `resolve_algo_class(algo) -> type[BaseAlgorithm]` registry with lazy
`sb3_contrib` import, raising on unknown names; call it from all three sites.

---

#### 1.5 DQN replay buffers are sized at 5.6–16.9 GB and `optimize_memory_usage` is never set  `critical` · `small`  **[FIXED]**

> **Fixed** in `db43608` — Size DQN replay buffers to fit the target hardware.

**Where:** `src/rl_doom/sb3_utils.py:143-166`; all six `configs/dqn_*.yaml`

**What:** SB3's `ReplayBuffer` allocates **separate** `observations` and `next_observations`
arrays unless `optimize_memory_usage=True`. That flag is never set anywhere in the repo or
the configs, and `_build_model`'s `DQN(...)` call does not pass it. Computed from the
configs' own `buffer_size` and `resize_shape`/`num_stack`:

| config | `buffer_size` | obs bytes | replay RAM |
|---|---:|---:|---:|
| `dqn_basic.yaml` | 100,000 | 28,224 | **5.6 GB** |
| `dqn_deadly_corridor.yaml` | 200,000 | 28,224 | **11.3 GB** |
| `dqn_deadly_corridor_curriculum.yaml` | 200,000 | 28,224 | **11.3 GB** |
| `dqn_defend_the_center.yaml` | 200,000 | 28,224 | **11.3 GB** |
| `dqn_deathmatch.yaml` | 300,000 | 28,224 | **16.9 GB** |
| `dqn_deathmatch_curriculum.yaml` | 300,000 | 28,224 | **16.9 GB** |

**Why it matters:** Colab's standard runtime provides ~12.7 GB of RAM, and the repo
documents Colab as its target (`README.md` "Google Colab" section, plus the Drive-persistence
cells in all six notebooks). Both deathmatch DQN configs are unrunnable there, and the three
11.3 GB configs will OOM once the ViZDoom processes (`n_envs`), the model, and the eval env
are accounted for. Six of the 24 runs in
`configs/matrix/deadly_corridor_curriculum.yaml` are `dqn_deadly_corridor{,_curriculum}` at
11.3 GB each — the flagship matrix cannot complete on a Colab-class machine.

**Fix:** Pass `optimize_memory_usage=hyperparams.get("optimize_memory_usage", True)` in
`_build_model`'s `DQN(...)` call — SB3's frame-sharing layout halves this to 2.8–8.5 GB.
Then re-check the deathmatch buffer sizes against the target machine, and add a startup
assertion that estimates replay RAM against `psutil.virtual_memory().available` and fails
fast with the number rather than 40 minutes later with a `MemoryError`.

---

#### 1.6 The deathmatch curricula silently discard the configured `doom_skill`  `medium` · `small`  **[FIXED]**

> **Fixed** in `4c74aa2` — Make the curriculum-vs-baseline comparison controlled.

**Where:** `src/rl_doom/sb3_utils.py:700-708`; `configs/*_deathmatch_curriculum.yaml`; `src/rl_doom/curriculum.py:307`

**What:** `train_sb3` overrides the caller's skill with stage 0's, unconditionally:

```python
doom_skill = curriculum_stages[0].skill     # sb3_utils.py:708 — no `is not None` guard
```

The deathmatch curricula ramp **bots**, not skill — their stages carry no `skill` key:

```yaml
stages:
  - {num_bots: 2, promote_at: 3.0}
  - {num_bots: 4, promote_at: 5.0}
  - {num_bots: 8, promote_at: null}
```

So `curriculum_stages[0].skill` is `None`, and the configured `env.doom_skill: 3` is
replaced by `None`. Both envs are then built with `doom_skill=None`, which falls through
`SCENARIO_DEFAULT_SKILL` (an empty dict, `env.py:139`) to the scenario `.cfg`'s baked-in
default. The curriculum callback does not recover it either — `apply_stage_to_doom_env`
guards with `if stage.skill is not None:` (`curriculum.py:307`). The code even *announces*
the override at `sb3_utils.py:704-707`.

**Why it matters:** `configs/ppo_deathmatch.yaml` sets `doom_skill: 3` and its curriculum
sibling sets `doom_skill: 3` too — but only the baseline actually gets it. That is the same
class of confound as 1.10, across all four `*_deathmatch_curriculum.yaml` configs.

*(The `num_bots` half is fine: the callback applies stage 0's bot count at
`_on_training_start`, before any env steps, to both train and eval envs.)*

**Fix:** Guard the override —
`if curriculum_stages[0].skill is not None: doom_skill = curriculum_stages[0].skill` — and
likewise seed `num_bots` from stage 0 at env-construction time so the pre-callback and
post-callback state agree.

---

#### 1.7 Single-player deathmatch never respawns the player  `medium` · `small`

**Where:** `src/rl_doom/env.py:303-343`; contrast `src/rl_doom/multiplayer_env.py:256-257`

**What:** `multiplayer_env` handles death explicitly:

```python
if self._respawn_on_death and game.is_player_dead():
    game.respawn_player()
```

`DoomEnv` never calls `respawn_player()`. Its only use of `is_player_dead()` is in
`_classify_termination` (`env.py:338`), i.e. to *label* the episode as ended by death.

**Why it matters:** the 12 `*_deathmatch*.yaml` configs train single-player deathmatch with
bots, and the episode therefore ends at the agent's first death rather than running the
scenario's 4200-tic match. `BENCHMARKS.md` sets the deathmatch target at "**≥8 frags vs 8
bots**" and "mean time between deaths" — both assume a match the agent survives across
multiple lives. As configured, the agent must land 8 frags before dying once.

**Fix:** Add `respawn_on_death: bool = False` to `DoomEnv` (mirroring `multiplayer_env`'s
parameter name), respawn in `step` when the player is dead and the episode has not finished,
and set it for deathmatch. Confirm against a short run which of the two behaviours ViZDoom's
`deathmatch.cfg` actually produces before changing the default — this one needs the binary
to settle, which the current CI setup cannot do (see 5.1).

---

#### 1.8 Notebook 05 evaluates every agent under the wrong conditions  `high` · `small`

**Where:** `notebooks/05_analysis_and_results.ipynb` cell 4; `src/rl_doom/env.py:459-496`

**What:** The analysis notebook's only env factory is:

```python
def make_env(scenario, seed=None):
    return make_wrapped_env(scenario)
```

`make_wrapped_env` takes `doom_skill`, `num_bots`, `resize_shape`, `frame_skip`,
`num_stack`, and `use_compound_actions` — none are forwarded, and the `seed` parameter is
accepted and then ignored.

**Why it matters:** three separate problems in one line.
- **Difficulty mismatch:** deathmatch agents trained against 4 or 8 bots are evaluated
  against **0**; curriculum agents trained up to skill 3 are evaluated at the cfg default.
  The comparison table README advertises is measuring a different task than the one trained.
- **No reproducibility:** `DoomEnv.reset` only calls `game.set_seed` when a seed is passed
  (`env.py:289-290`), so the ViZDoom RNG is never reset. README's "multi-seed evaluation"
  is not seeded.
- Combined with 1.4, the notebook also cannot load its Recurrent PPO runs.

**Fix:** Read `config.json` from each run dir (it already records `env_settings`) and build
the eval env from it, forwarding `doom_skill`, `num_bots`, and the preprocessing knobs.
Thread `seed` through to `env.reset(seed=seed)`. This is the same "reconstruct the env from
the run's own config" helper that `_record_video` needs.

---

#### 1.9 `min_evals_between_promotions` does not do what the configs say it does  `medium` · `small`

**Where:** `src/rl_doom/curriculum.py:233-247`; claim in `configs/ppo_deadly_corridor_curriculum.yaml` (and the 7 sibling curriculum configs)

**What:** The config comment states:

> `min_evals_between_promotions=2` prevents a single lucky eval from jumping a stage; the
> curriculum waits for at least two eval windows above threshold before promoting.

The implementation increments `_evals_since_promotion` on **every** eval, whether or not it
cleared the threshold, then promotes on `eval_mean_reward >= threshold and
_evals_since_promotion >= min_gap`. So it enforces a *minimum spacing between promotions*
— which is what the parameter name says — but it provides **no** protection against a
single lucky eval. Once two evals have elapsed since the last promotion, one above-threshold
result promotes immediately.

**Why it matters:** ViZDoom eval variance over 10 episodes is large. The stated guardrail
against premature promotion doesn't exist, and a spuriously good eval can advance
deadly_corridor from skill 1 to skill 2 before the policy is ready — exactly the failure the
comment claims to prevent.

**Fix:** Either correct the config comments to describe spacing, or implement the
documented behaviour by tracking consecutive above-threshold evals:

```python
self._consecutive_above = self._consecutive_above + 1 if eval_mean_reward >= threshold else 0
if self._consecutive_above >= self._min_gap: ...
```

The second is what the configs' tuning rationale assumes, so it is probably the intent.

---

#### 1.10 Dreamer curriculum configs give the curriculum arm 1.5–2× the baseline budget  `high` · `small`  **[FIXED]**

> **Fixed** in `4c74aa2` — Make the curriculum-vs-baseline comparison controlled.

**Where:** `configs/dreamer_deadly_corridor.yaml` vs `configs/dreamer_deadly_corridor_curriculum.yaml`; same for `dreamer_deathmatch*`

**What:** Verified across every baseline/curriculum pair:

| algorithm | scenario | baseline | curriculum | |
|---|---|---:|---:|---|
| ppo | deadly_corridor | 2,500,000 | 2,500,000 | OK |
| ppo | deathmatch | 2,500,000 | 2,500,000 | OK |
| dqn | deadly_corridor | 2,500,000 | 2,500,000 | OK |
| dqn | deathmatch | 2,500,000 | 2,500,000 | OK |
| recurrent_ppo | deadly_corridor | 2,500,000 | 2,500,000 | OK |
| recurrent_ppo | deathmatch | 2,500,000 | 2,500,000 | OK |
| **dreamer** | **deadly_corridor** | **500,000** | **1,000,000** | **2.0×** |
| **dreamer** | **deathmatch** | **1,000,000** | **1,500,000** | **1.5×** |

**Why it matters:** `configs/matrix/deadly_corridor_curriculum.yaml` runs
`dreamer_baseline` against `dreamer_curriculum` to answer "does the skill curriculum
help?". With unequal budgets the Dreamer comparison measures compute, not curriculum. This
also violates the principle the PPO curriculum config states in its own header: *"Identical
hyperparameters keep the comparison apples-to-apples; the only difference is the
`curriculum:` block."*

**Fix:** Set both Dreamer curriculum configs to their baselines' `total_timesteps` (and
matching `checkpoint_freq`). This is the strongest argument for the `extends:` mechanism in
6.1 — with config inheritance this class of drift becomes impossible.

---

### 2. Structure & restructuring

#### 2.1 `sb3_utils.py` is a god-module with six unrelated responsibilities  `high` · `large`

**Where:** `src/rl_doom/sb3_utils.py` (1,059 lines); `train_sb3` alone is **361 lines**
(`sb3_utils.py:613-973`), `_record_video` is 155

**What:** One module holds GPU introspection (`:64`), model construction (`:97`), policy-kwargs
parsing (`:171`), an SB3 callback (`:214`), termination reporting (`:256`), CSV/NPZ loading
(`:291-322`), matplotlib plotting (`:323-450`), video recording (`:451-607`), the training
driver (`:613`), and Monitor aggregation (`:981`). The name "sb3_utils" describes a
dependency, not a responsibility — which is why everything lands there.

**Why it matters:** `train_sb3` is the highest-value function in the repo and the hardest
to test — you cannot exercise its artifact-writing leg without running training. It's also
why the artifact leg is duplicated in `train_dreamer` (2.2).

**Fix:** Split by responsibility, moving code verbatim first (no behaviour change):
`rl_doom/training/model_factory.py` (`_build_model`, `policy_kwargs_from_config`,
`resolve_algo_class`), `rl_doom/training/callbacks.py` (`TerminationTracker`),
`rl_doom/reporting/plots.py`, `rl_doom/reporting/video.py`, `rl_doom/reporting/metrics.py`
(`_load_progress_csv`, `_load_evaluations`, `_collect_episode_stats`, `_legacy_eval_log`),
leaving `rl_doom/training/sb3.py` holding `train_sb3` reduced to orchestration.

---

#### 2.2 `agents/dreamer.py` is a 1,165-line training driver living in the agents package  `high` · `large`

**Where:** `src/rl_doom/agents/dreamer.py`; `train_dreamer` is **425 lines** (`:741`)

**What:** The module mixes a `sys.path` shim for the upstream port (`:37-62`), a
re-implementation of the upstream `Dreamer` nn.Module (`:112+`), and a full training driver
with its own plotting, video recording, `training.npz` writer, and `curriculum.json` writer.
Its siblings (`agents/dqn.py`, `agents/ppo.py`) are 120- and 220-line nn.Module wrappers —
the package boundary means two different things.

**Why it matters:** the artifact/reporting leg — learning-curve figures, `_legacy_eval_log`'s
5-column matrix, `training.npz`, `curriculum.json`, video + `_episodes.json` sidecar — is
written twice, once per driver, and the two copies can drift. `summary.py` consumes both.

**Fix:** Move the driver to `rl_doom/training/dreamer.py`, leaving only the port shim and
the nn.Module in `agents/`. Then give both drivers a single
`finalize_run(run_dir, ...)` in `rl_doom/reporting/`.

---

#### 2.3 The documented CLI entry point does not exist  `medium` · `medium`

**Where:** `README.md:37`, `README.md:81-83`, `PLAN.md:38`, `PLAN.md:131`, `PLAN.md:216`

**What:** README's project-structure block lists `src/rl_doom/train.py  # CLI training entry
point` and Quick Start tells users to run `python -m rl_doom.train --config
configs/dqn_basic.yaml`. There is no `train.py`. `PLAN.md:131` acknowledges this
(*"A standalone `src/rl_doom/train.py` CLI entry point is not yet implemented"*) while
`PLAN.md:138` and `PLAN.md:216` still list it as available.

**Why it matters:** beyond the broken quick-start, the missing entry point is why the
YAML → `train_sb3(**kwargs)` adapter is copy-pasted into four notebooks and the matrix
runner (3.2). One CLI would collapse five copies into one.

**Fix:** Add `src/rl_doom/train.py` with `python -m rl_doom.train --config <path>
[--seed N] [--total-timesteps N] [--device ...]`, implemented over a single
`run_from_config(cfg, ...)` helper that the notebooks and matrix runner also call. Then
README's examples become true.

---

#### 2.4 Five of six notebooks run Colab-only setup unconditionally, so they cannot execute locally  `high` · `small`  **[FIXED]**

> **Fixed** in `74a54fe` — Make the notebooks runnable outside Colab.

**Where:** cells 3 and 4 of `notebooks/01`, `02`, `03`, `04`, `06`; `README.md` "Google Colab" section

**What:** README states:

> Each notebook includes a **commented-out** Colab Setup cell at the top. Uncomment it to
> automatically: 1. Clone the repo to `/content/rl-doom` …

The cell is not commented out. Verbatim from `notebooks/02_dqn_training.ipynb` cell 3:

```python
# --- Colab Setup ---
# Uncomment the block below when running on Google Colab
import subprocess, os
if not os.path.exists("/content/rl-doom"):
    subprocess.run(["git", "clone", "https://github.com/kuds/rl-doom.git", "/content/rl-doom"], check=True)
os.chdir("/content/rl-doom/notebooks")
subprocess.run(["pip", "install", "-q", "-e", "/content/rl-doom[notebooks]"], check=True)
```

The instruction to uncomment sits directly above live code. Cell 4 is the same:

```python
# Uncomment the block below when running on Google Colab
# ---
import shutil
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = "/content/drive/MyDrive/Finding Theta/rl-doom"
```

Only `notebooks/05_analysis_and_results.ipynb` cell 3 is genuinely commented out — which is
what the intended state looks like.

**Why it matters:** run locally, cell 3 clones the public repo into `/content/rl-doom` and
then `os.chdir` into it (or raises `FileNotFoundError`), and cell 4 raises
`ModuleNotFoundError: google.colab`. The Colab block is *fused into the same cell* as the
imports, seeding, and `GPU_INFO` setup, so a local user cannot skip it — they must edit the
notebook. Combined with 2.3 (the documented `python -m rl_doom.train` CLI does not exist),
**the repo currently has no working local path to train anything**, while README's Quick
Start says "Run the notebooks in order". `DRIVE_ROOT` also hardcodes a personal Drive path.

The same 24-line Drive block is byte-identical across notebooks 02/03/04/06 — and duplicates
`utils.setup_google_drive`, which is dead (2.7).

**Fix:** Guard rather than comment, so one cell works in both environments:

```python
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if IN_COLAB:
    ...clone / chdir / pip install / drive.mount...
```

Move the shared body into `rl_doom.notebook_setup.setup_colab(drive_root=...)` so the six copies
collapse to one call, and make `DRIVE_ROOT` a module-level constant the user overrides.
Split the Colab block out of the imports/seeding cell either way.

---

#### 2.5 The hand-rolled agent stack is legacy, but README presents it as the product  `medium` · `medium`

**Where:** `src/rl_doom/agents/dqn.py` (120), `agents/ppo.py` (220), `models.py` (87), `replay_buffer.py` (55); `README.md:110-125`

**What:** Notebooks 02/03/04 train via `train_sb3` — the hand-rolled classes are never used
for training. `ReplayBuffer`, `DQNAgent`, and `PPOAgent` are referenced only by
`tests/` and by `evaluate.py`'s legacy-checkpoint loader (`evaluate.py:247-253`), i.e. they
exist to load a checkpoint format the pipeline no longer produces. Meanwhile README's
"Algorithms" section describes *this* implementation — "epsilon-greedy exploration with
linear decay", "experience replay buffer", "target network with periodic hard updates",
"mini-batch updates over collected rollouts" — as if it were what runs.

**Why it matters:** ~480 lines carried at maintenance cost, with the repo's user-facing
documentation describing the dead path rather than the live one. Finding 1.2 is a direct
consequence — nobody exercises this code, so a silent normalization bug survived.

**Fix:** Decide explicitly. Either (a) delete the four modules, their tests, and the legacy
branch in `evaluate.py`, and rewrite README's Algorithms section to describe the SB3
configuration actually used; or (b) keep them as a teaching reference, move them to
`rl_doom/reference/`, fix 1.2, and label them as such in README. Option (a) is
recommended — the SB3 path is the product.

---

#### 2.6 `self_play.py` + `tournament.py` are 1,030 lines with no production caller  `medium` · `medium`

**Where:** `src/rl_doom/self_play.py` (500), `src/rl_doom/tournament.py` (530); tests in `tests/test_self_play.py` (301) + `tests/test_tournament.py` (434)

**What:** A repo-wide grep for every public symbol in these modules finds references only
from `tests/`. `train_self_play` and `build_self_play_env` — the two entry points — have
**zero** references anywhere, including tests. No notebook, script, or config reaches this
subsystem, and finding 1.3 means a single-player deathmatch checkpoint could not be used
as an opponent snapshot even if one did.

**Why it matters:** 1,030 lines of source plus 735 lines of tests that verify internal
consistency without verifying the feature works end to end. This is the largest single
block of speculative code in the repo.

**Fix:** Give it an entry point or park it. Minimum viable: fix 1.3, add
`configs/selfplay_deathmatch.yaml` plus a `scripts/run_self_play.py`, and one integration
test that trains ~1k steps against a random opponent and runs a 2-agent tournament. If
that isn't on the roadmap, move the subsystem to a branch and note it in PLAN.md.

---

#### 2.7 `utils.py` is entirely dead code, and it encodes a directory layout that contradicts `paths.py`  `low` · `small`

**Where:** `src/rl_doom/utils.py` (199 lines, 0% coverage)

**What:** All six public symbols — `setup_google_drive`, `plot_with_smoothing`,
`plot_eval_curve`, `TrainingTimer`, `save_full_checkpoint`, `ensure_artifact_dirs` — have
**zero** references anywhere in `src/`, `scripts/`, `tests/`, or `notebooks/`. Verified by
AST symbol extraction plus repo-wide grep.

Worse, `setup_google_drive` and `ensure_artifact_dirs` create a flat
`checkpoints/ logs/ figures/ media/ runs/` layout at the repo root, which contradicts the
`training_jobs/<env>/<algo>/runs/<id>/` layout `paths.py` actually implements. Two
mutually inconsistent conventions ship in one package. And the docstring at `utils.py:32`
claims *"On non-Colab runtimes this is a no-op (the import will fail silently)"* — it does
not: `from google.colab import drive` raises `ModuleNotFoundError`, uncaught.

Also dead, in `paths.py`: `new_sweep_dir`, `new_sweep_variant_dir`, `analysis_root` — the
entire `sweeps/` layout documented in the module docstring is unused.

**Fix:** Delete `utils.py` and the three dead `paths.py` helpers (or the `sweeps/` docstring
section too). If the Drive helper is wanted, rewrite it against `paths.py`'s real layout and
call it from the notebooks, which currently copy-paste that block six times (3.4).

---

### 3. Duplication & consolidation

#### 3.1 24 configs with no inheritance; the 8 curriculum files are whole-file clones  `high` · `medium`

**Where:** `configs/*.yaml` — 24 files, 45 distinct keys, 6 per algorithm

**What:** Each curriculum config is byte-equivalent to its baseline apart from removing
`env.doom_skill` and appending a `curriculum:` block. The PPO file says so itself. There is
no `extends:`/`base:` mechanism, so every shared value is repeated 6–24 times. Ironically,
`scripts/run_experiment_matrix.py` *already implements* a shallow-merge `overrides:`
mechanism for matrix variants — the capability exists but isn't available to the configs.

**Why it matters:** findings 1.6 and 1.10 are exactly the drift this causes — a budget change landed in
two of eight curriculum files and confounded an experiment. Tuning one hyperparameter today
means editing up to six files consistently.

**Fix:** Add `configs/_base/{common,ppo,dqn,recurrent_ppo,dreamer}.yaml` and an `extends:`
key resolved in `load_yaml_config` (`paths.py:222`) with the same shallow-per-block merge the
matrix runner already uses. Reduce each curriculum config to `extends: ppo_deadly_corridor`
plus its `curriculum:` block. Write the resolved config into `config.json` so runs stay
fully reproducible from a single artifact.

---

#### 3.2 The YAML → `train_sb3` adapter is copy-pasted five times with divergent defaults  `high` · `medium`

**Where:** `scripts/run_experiment_matrix.py:184-290`, and notebooks 02/03/04/06

**What:** Each site independently unpacks `cfg["env"]`, `cfg["hyperparams"]`,
`cfg["training"]`, `cfg["eval"]`, `cfg["policy"]`, calls `write_config(...)`, then calls
`train_sb3(**kwargs)`. `run_experiment_matrix.py` alone has 31 `.get(key, default)` calls
supplying its own fallbacks; the notebook copies use different ones.

**Why it matters:** the same YAML can produce different runs depending on which entry point
loaded it, and none of the five paths has a test.

**Fix:** One `run_from_config(cfg, *, run_dir=None, tag=None, total_timesteps_override=None,
device=None, on_complete=None)` that owns the adapter, dispatches SB3 vs Dreamer, and is
called by the new CLI (2.3), the matrix runner, and every notebook. This is the same
function the CLI needs, so 2.3 and 3.2 are one piece of work.

---

#### 3.3 ViZDoom setup and preprocessing are implemented twice; env-unwrapping seven times  `medium` · `medium`

**Where:** `env.py:190-265` vs `multiplayer_env.py`; unwrap loops at `sb3_utils.py:494`, `evaluate.py:87`, `curriculum.py:305`, `curriculum.py:515`, and elsewhere

**What:** Game construction, screen-format setup, available-game-variable patching, action-table
construction from button names, and grayscale/resize/stack preprocessing each exist once in
`env.py` and again inline in `multiplayer_env.py`. Separately, the "walk `.env` down to the
base `DoomEnv`" loop is written seven times with three different stop conditions.

**Why it matters:** finding 1.3 is the concrete cost of the first duplication. The unwrap
loops are each individually trivial but collectively a maintenance hazard — a new wrapper in
the stack means auditing seven call sites.

**Fix:** Extract `build_action_table(game, combos)`, `configure_game(scenario, ...)`, and
`extract_obs(game, shape)` as module-level helpers in `env.py` and import them in
`multiplayer_env.py`. Add one `unwrap_to(env, *, attr=None)` helper in `env.py` and replace
all seven loops.

---

#### 3.4 Video recording ×3, learning-curve plotting ×5, Drive setup ×6  `medium` · `medium`

**Where:** video: `sb3_utils.py:451`, `sb3_utils.py:507`, `agents/dreamer.py:607`;
plots: `sb3_utils.py:333`, `agents/dreamer.py:550`, notebook 05, and the dead `utils.py:72`;
Drive setup: all six notebooks

**What:** Greedy-rollout video with per-episode MP4 + `_episodes.json` sidecar and MP4→GIF
fallback exists in three near-identical implementations. The 2×3 learning-curves figure with
MA-20 smoothing exists in five, while the parameterised `utils.plot_with_smoothing` that
would serve all of them is dead code (2.7).

**Fix:** `rl_doom/reporting/video.py::write_clip(frames, path, fps)` and
`write_episode_sidecar(...)`; `rl_doom/reporting/plots.py` owning the figure, with
`plot_with_smoothing` revived as its smoothing primitive. Fold the notebook Drive cell into a
single `setup_google_drive()` call.

---

### 4. Robustness & long-run reliability

#### 4.1 `model.learn()` is unguarded — a crash near the end loses the entire run  `critical` · `medium`  **[FIXED]**

> **Fixed** in `dadb40a` — Salvage a train_sb3 run when training fails.

**Where:** `src/rl_doom/sb3_utils.py:806-827`

**What:** `model.learn(...)` is called with no `try`/`finally`. Everything that makes a run
useful — `model.save()`, `vec_normalize.pkl`, `training.npz`, figures, video, `config.json`
finalisation, `stage_summary.txt` — happens *after* it returns (`:817-945`). An exception
anywhere in `learn()` skips all of it. The `vec_env.close()` / `eval_env.close()` calls are
also only on the success path, and are wrapped in `except Exception: pass` (`:823-827`), so
ViZDoom processes leak on failure.

**Why it matters:** configs specify 2.5M timesteps. On Colab (the documented target
runtime, which preempts) a run that dies at 2.4M leaves nothing but orphaned
`CheckpointCallback` zips that nothing in the repo can consume.

**Fix:** Wrap `learn()` in `try/except/finally`. On exception: still save the model, still
write whatever metrics exist, call `mark_run_status(run_dir, status="failed")`, close both
envs in the `finally`, then re-raise. Move env teardown into `finally` unconditionally.

---

#### 4.2 There is no resume path, though the code claims to prepare for one  `high` · `medium`

**Where:** `sb3_utils.py:818-822`, `sb3_utils.py:789-794`, `sb3_utils.py:808`

**What:** `vec_normalize.pkl` is saved with the comment *"so a future resumed run can pick
up with the same reward-scale estimate"* — nothing anywhere loads it.
`CheckpointCallback` writes `step_*.zip` every `checkpoint_freq` steps — nothing reads them.
`model.learn()` is always called fresh with default `reset_num_timesteps=True`. Nothing
persists optimizer state beyond the SB3 zip, the curriculum stage index, or RNG state.

**Why it matters:** the repo is designed for multi-hour Colab runs and has no answer to
preemption. A resumed run would also restart the curriculum at stage 0, silently changing
the experiment.

**Fix:** Add `train_sb3(..., resume_from: Path | None = None)`: load the newest
`checkpoints/step_*.zip` via `resolve_algo_class(algo).load()`, restore `vec_normalize.pkl`
with `VecNormalize.load(path, vec_env)`, restore `SkillCurriculumCallback` state from
`metrics/curriculum.json`, and call `learn(..., reset_num_timesteps=False)`. Persist the
curriculum stage on every promotion, not only at the end.

---

#### 4.3 The 24-run matrix runner has zero exception handling and writes its CSV only at the end  `high` · `small`  **[FIXED]**

> **Fixed** in `78f5b02` — Isolate matrix cells so one failure doesn't lose the run.

**Where:** `scripts/run_experiment_matrix.py:362-373`

**What:** The module contains **no `try:` statement at all**. `main()` is:

```python
rows = []
for spec in runs:
    rows.append(_run_one(spec, ...))
summary_path = _write_matrix_summary(matrix_name, rows)
```

One exception in run 3 of 24 aborts the loop, skips runs 4–24, and skips
`_write_matrix_summary` — so the summary rows for the two runs that *did* succeed are
discarded too. There is no `--resume` and no `--skip-existing`.

**Why it matters:** `configs/matrix/deadly_corridor_curriculum.yaml` is 24 runs at up to
2.5M steps each — GPU-days of work behind a single unguarded loop.

**Fix:** Wrap `_run_one` in `try/except`, record `status: failed` + the exception in the
row, and continue. Append each row to the CSV as it completes rather than batching. Add
`--resume` that skips (variant, seed) cells whose run dir already has
`config.json` with `status=completed`, and `--continue-on-error/--fail-fast`.

---

#### 4.4 Eleven `except Exception` blocks, several swallowing silently  `medium` · `small`

**Where:** `env.py:317,326`, `sb3_utils.py:826`, `agents/dreamer.py:645,1028,1163`, and five others

**What:** `env.py:317` and `:326` catch bare `Exception` around game-variable reads and
substitute `None`/`0`. `sb3_utils.py:826` is `except Exception: pass` around env teardown.
`agents/dreamer.py:1028` likewise.

**Why it matters:** a misconfigured scenario that never exposes `KILLCOUNT` reports zero
kills for the whole run rather than failing loudly, and the resulting `stage_summary.txt`
reads as a legitimately bad result.

**Fix:** Narrow to the exception ViZDoom actually raises, and log once per run (not per
step) when a metric is unavailable, so the summary can mark it "unavailable" rather than 0.

---

#### 4.5 Multiplayer env init blocks forever on a failed peer, and the promised port retry doesn't exist  `medium` · `small`

**Where:** `src/rl_doom/multiplayer_env.py:355-362`, `multiplayer_env.py:58-67`

**What:** `_start_games` fans out per-seat `init()` calls and then waits with no timeout:

```python
for fut in futures:
    fut.result()          # multiplayer_env.py:361 — unbounded
```

ViZDoom's host/client handshake blocks in native code until every declared player joins. If
one seat's `_init_one` fails — port collision, a crashed peer, a bad `-join` argument — the
surviving seats block on the handshake and `fut.result()` blocks on them. Nothing times out.

Relatedly, `_pick_free_port` (`:58`) binds an ephemeral port, reads its number, and closes
the socket, then `init()` binds it again later. The docstring acknowledges the TOCTOU window
and says *"the cost of a collision is one retry at env construction"* — **there is no retry
anywhere in the module.** A collision surfaces as the unbounded hang above.

**Why it matters:** `tournament.py` runs many matches in sequence. One hung match hangs the
whole tournament with no diagnostic, and leaves ViZDoom processes behind.

**Fix:** Pass a timeout to `fut.result(timeout=...)` (60s is generous for a local handshake),
and on timeout or exception close every game that did start before re-raising. Either
implement the documented retry — loop `_pick_free_port` + `init` a few times on bind failure
— or delete the claim from the docstring.

---

#### 4.6 A resumed Dreamer run restarts its step counter at zero  `medium` · `small`

**Where:** `src/rl_doom/agents/dreamer.py:886`, `:930-937`, `:157`, `:943`

**What:** Unlike the SB3 path, `train_dreamer` *does* have a resume branch — it loads
`checkpoints/latest.pt` and restores weights and optimizer state (`:930-937`). But the step
counter is not part of it. The logger is always constructed at zero:

```python
logger = tools_mod.Logger(tb_dir, 0)          # :886
```

and `_Dreamer.__init__` derives its counter from the logger (`:157: self._step =
logger.step // config.action_repeat`), while the training loop runs
`while agent._step < cfg.steps + cfg.eval_every` (`:943`).

**Why it matters:** a resumed run re-executes the *entire* configured budget with restored
weights, so total training is double what the config says, and every step number written to
TensorBoard and `training.npz` restarts from zero. The resulting curves cannot be compared
against a non-resumed run — which is exactly what
`configs/matrix/deadly_corridor_curriculum.yaml` does.

Separately, `:932` calls `torch.load(latest_ckpt, map_location=cfg.device)` **without**
`weights_only=`, while `agents/dqn.py:118` and `agents/ppo.py:219` both pass
`weights_only=True`. `pyproject.toml` declares an unpinned `torch>=2.0`; PyTorch flipped
this default in 2.6, so this call's behaviour depends on which torch resolves at install
time.

**Fix:** Persist `step` in the checkpoint and seed `Logger(tb_dir, ckpt["step"])` on resume.
Pass `weights_only=False` explicitly at `:932` (the checkpoint legitimately contains
optimizer state) so the behaviour is pinned rather than version-dependent.

---

#### 4.7 `_record_video` buffers whole episodes of raw frames in memory  `low` · `small`

**Where:** `src/rl_doom/sb3_utils.py:530`

**What:** `frames: list[np.ndarray] = []` accumulates every rendered frame of an episode
before encoding. At 320×240 RGB (`env.py:194`) that is ~230 KB per frame; a deathmatch
episode at the 1050-step limit buffers ~240 MB. The buffer is per-episode, so it does not
grow across the five recorded episodes — but it lands at the very end of a training run,
alongside a loaded model and two live vec-envs. The video env's `close()` is also outside a
`finally`.

**Fix:** Stream frames to the encoder via `imageio.get_writer(...)` as they are produced
rather than collecting them, and move the env teardown into a `finally`.

---

### 5. Testing & CI

#### 5.1 `env.py` — the most important module in the repo — has 0% coverage in CI  `high` · `medium`  **[FIXED]**

> **Fixed** in `b824917` — Run the real test suite in CI.

**Where:** `tests/test_env.py:9`, `tests/test_dreamer_env.py:11`; contrast `tests/conftest.py:1-60`

**What:** Measured coverage in an environment matching `.github/workflows/ci.yml` exactly
(no `vizdoom`, no `sb3-contrib`):

```
src/rl_doom/env.py               177 stmts   177 miss    0%
src/rl_doom/dreamer_env.py        53 stmts    53 miss    0%
src/rl_doom/evaluate.py          112 stmts   112 miss    0%
src/rl_doom/utils.py              89 stmts    89 miss    0%
src/rl_doom/agents/ppo.py         83 stmts    67 miss   19%
src/rl_doom/sb3_utils.py         403 stmts   319 miss   21%
src/rl_doom/agents/dreamer.py    485 stmts   359 miss   26%
src/rl_doom/agents/dqn.py         51 stmts    37 miss   27%
------------------------------------------------------------
TOTAL                           2632 stmts  1483 miss   44%
```

(The `agents/*` and `sb3_utils` numbers are depressed by a second, independent problem —
see 5.2.)

`tests/test_env.py` and `tests/test_dreamer_env.py` call
`pytest.importorskip("vizdoom")` at module scope, so they collect **zero** tests in CI:

```
tests/test_env.py        -> no tests collected
tests/test_dreamer_env.py -> no tests collected
```

The suite reports "126 passed, 3 skipped" — which reads as healthy while the wrapper stack
is completely unexercised.

**Why it matters:** every finding in section 1 lives in code CI cannot see. The
`SkipFrame`/`FrameStack`/`ResizeObservation` contract, the truncation semantics, the action
table, and the observation shape are all untested where it counts.

**Fix:** The repo already solved this — `tests/conftest.py` provides a `fake_vizdoom`
fixture with a `_FakeDoomGame` stub, used by `test_multiplayer_env.py`, `test_self_play.py`,
and `test_tournament.py` (which is why `multiplayer_env.py` reaches 94%). Extend that stub
to cover `DoomGame` methods `env.py` uses (`load_config`, `set_*`, `init`,
`get_available_buttons`, `make_action`, `is_episode_finished`, `is_player_dead`,
`get_episode_time`, `get_game_variable`) and port `test_env.py` onto it. Keep the current
binary-dependent tests as a separate `test_env_integration.py`.

Highest-value test cases, named:
1. `test_timeout_sets_truncated_not_terminated` — the 1.1 regression guard.
2. `test_skipframe_accumulates_reward_and_stops_on_done`.
3. `test_framestack_resets_stack_on_episode_boundary`.
4. `test_deathmatch_action_space_matches_multiplayer` — the 1.3 guard.
5. `test_cnn_input_is_normalized` — the 1.2 guard.
6. `test_dqn_target_uses_online_argmax_and_target_value` — numeric assertion on a
   hand-constructed batch, not a shape check.
7. `test_gae_bootstraps_on_truncation` — numeric, against a closed-form expectation.
8. `test_config_rejects_unknown_key` — the 6.1 guard.

---

#### 5.2 A misplaced `importorskip` silently disables all 12 agent tests in CI  `high` · `small`  **[FIXED]**

> **Fixed** in `b824917` — Run the real test suite in CI.

**Where:** `tests/test_agents.py:149`; `.github/workflows/ci.yml:31-35`

**What:** CI installs dependencies by hand and omits `sb3-contrib`:

```yaml
pip install gymnasium numpy opencv-python pyyaml tensorboard matplotlib
pip install pettingzoo stable-baselines3      # <-- no sb3-contrib
pip install pytest ruff ruff mypy
pip install -e . --no-deps
```

`tests/test_agents.py:149` is a **module-level** `pytest.importorskip("sb3_contrib")`, placed
partway down the file after nine DQN/PPO tests. A module-level skip aborts collection of the
*whole module*, not just the tests below it — so the nine tests defined above line 149 are
skipped too.

**Verified by uninstalling `sb3-contrib` and re-running** (i.e. reproducing CI exactly):

```
$ python -m pytest tests/test_agents.py -q -rs
SKIPPED [1] tests/test_agents.py:149: could not import 'sb3_contrib'
1 skipped in 2.24s                       # 12 tests locally -> 0 in CI

$ python -m pytest -q
114 passed, 4 skipped                    # vs 126 passed, 3 skipped locally
```

CI therefore runs **12 fewer tests than a developer's local run and reports green**. Coverage
under true CI conditions drops accordingly:

| module | local | **CI** |
|---|---:|---:|
| `agents/dqn.py` | 100% | **27%** |
| `agents/ppo.py` | 100% | **19%** |
| `sb3_utils.py` | 26% | **21%** |
| **TOTAL** | 48% | **44%** |

So `DQNAgent.train_step`, `PPOAgent._compute_gae`, and `_build_model`'s entire dispatch block
(`sb3_utils.py:114-166`) are unexecuted in CI — which is why 1.2 and 1.4 survived.

More broadly, the hand-written install list can drift from `pyproject.toml` with nothing to
detect it, and `--no-deps` means a missing declaration never surfaces.

**Fix:** Two independent changes, both small.
- Move `importorskip("sb3_contrib")` to the top of `test_agents.py` and split the
  RecurrentPPO tests into `tests/test_recurrent_ppo.py`, so an optional dependency can only
  disable its own tests. Better still, add a CI check that fails when the collected test
  count drops below a floor.
- Install from the declared extras (`pip install -e ".[dev]"` plus the CPU-torch index) so CI
  validates `pyproject.toml` rather than a parallel list, and add `sb3-contrib` so Recurrent
  PPO — one of the four documented algorithms, with six configs — is exercised at all.

---

#### 5.3 `mypy` fails on a clean checkout with current dependency versions  `medium` · `small`  **[FIXED]**

> **Fixed** in `b824917` — Run the real test suite in CI.

**What:** On a fresh install of the declared dependencies (torch 2.13, numpy 2.4, Python 3.11):

```
src/rl_doom/self_play.py:303: error: Unused "type: ignore" comment  [unused-ignore]
src/rl_doom/env.py:345: error: Unused "type: ignore" comment  [unused-ignore]
Found 2 errors in 2 files
```

`warn_unused_ignores = true` (`pyproject.toml`) plus fully unpinned dependencies means a
`type: ignore` that is necessary against one stub version becomes an error against the next.
CI is green only because it happens to resolve older versions today.

**Fix:** Either pin dev-tool and stub versions (a `requirements-dev.lock` or
`constraints.txt` CI installs with `-c`), or relax `warn_unused_ignores`. Pinning is
preferable — it also fixes the reproducibility gap in 7.2.

---

#### 5.4 No numeric regression guard on the RL math; test weight is misallocated  `medium` · `medium`

**What:** `tests/test_curriculum.py` is 655 lines for a 562-line module, while the
replay buffer gets 50 lines and PPO's GAE gets two shape/sanity tests
(`test_agents.py:82-108`). `tests/test_tournament.py` (434) + `tests/test_self_play.py`
(301) — 735 lines — test a subsystem with no production caller (2.6). Several curriculum
tests reach into private state (`_evals_since_promotion`, `_stages`, `_idx`), which is why
`curriculum.py:395-419` carries forwarding properties documented as "test-only accessor".

Two concrete examples of tests that assert less than their names claim:

- `tests/test_tournament.py:55-78`,
  `test_elo_expected_score_matches_formula_for_200_point_gap`: both players start at
  `initial=1200.0`, so the assertion computes
  `expected = 1/(1 + 10^((1200-1200)/400)) = 0.5` — a **zero**-point gap. The test's own
  inline comment admits the setup didn't work (*"Seed ratings by prepending no-op draws
  won't work"*). The rating-difference term of the Elo formula is never exercised; the test
  would pass if `compute_elo` ignored ratings entirely.
- `tests/test_summary.py:115`, `assert "3" in out` — satisfied by almost any generated
  summary.

**Why it matters:** the tests are load-bearing where the code isn't, and absent where it is.
Nothing prevents a regression in the Bellman target or GAE from reaching `main`. Note also
that no test parses any of the 24 `configs/*.yaml` files, so a malformed config is caught
only by a real training run.

**Fix:** Add the numeric tests in 5.1 (items 6–7). Fix the Elo test to construct a genuine
rating gap. Add a parametrised `test_all_shipped_configs_load` that runs every
`configs/*.yaml` through the loader and asserts the required keys — cheap, and it would have
caught 1.10 and 1.6. Rebalance by testing the curriculum controller through its public API and
deleting the "legacy"/"test-only" forwarding properties in `curriculum.py:395-419` — they
exist only to support tests that shouldn't be reaching in.

---

### 6. Config & API

#### 6.1 There is no config schema; typos are silently ignored  `high` · `medium`

**Where:** `src/rl_doom/paths.py:222-243`; `scripts/run_experiment_matrix.py` (31 `.get()` fallbacks)

**What:** `load_yaml_config` does `yaml.safe_load` and validates only that the top level is
a mapping. There is no schema, no dataclass, and no unknown-key rejection. Downstream,
`_build_model` is reasonably strict for hyperparameters (direct `hyperparams["lr"]` raises
`KeyError`), but the `env`/`training`/`eval` blocks are read entirely through
`.get(key, default)`. A typo like `total_timestpes:` or `n_env:` therefore trains silently
with the default value.

**Why it matters:** these are runs measured in GPU-days. A silently-defaulted
`total_timesteps` or `frame_skip` produces a plausible-looking result that answers a
different question than intended.

**Fix:** Add `rl_doom/config.py` with a frozen `RunSpec` dataclass (`scenario`, `algo`,
`seed`, `env`, `hyperparams`, `training`, `eval`, `policy`, `curriculum`) and a
`RunSpec.from_yaml(path)` that **rejects unknown keys at every level** and applies defaults
in exactly one place. Fold `extends:` (3.1) into the same loader. Snapshot the resolved
`RunSpec` into `config.json` so a run is reproducible from its own artifacts.

---

#### 6.2 Adding a fifth algorithm requires touching at least five files  `medium` · `medium`

**What:** Algorithm knowledge is spread across `_build_model` (`sb3_utils.py:114`), the
video loader (`sb3_utils.py:885`), `evaluate.py:216`, the `_PPO_FAMILY` VecNormalize gate
(`sb3_utils.py:751`), the `algo == "dreamer"` branch in `run_experiment_matrix.py:229`, and
`ALGO_LABELS` in `summary.py:133`. Scenario knowledge is similarly spread across
`SCENARIO_MAP`, `SCENARIO_ACTION_SETS`, `SCENARIO_DEFAULT_SKILL`, `scenario_limits.py`, and
`BENCHMARKS.md`.

**Fix:** A single `rl_doom/registry.py` mapping algo name → `{class_factory, family,
driver, label}`, consulted by all six sites. Same for scenarios: one record per scenario
holding cfg name, action set, default skill, timeout, and benchmark targets — which also
lets `BENCHMARKS.md` be generated rather than hand-maintained.

---

#### 6.3 `env.grayscale` is honoured only on the Dreamer path  `low` · `small`

**Where:** `configs/dreamer_*.yaml` (6 files set `grayscale: false`); `env.py:459-496`

**What:** `ResizeObservation` supports `grayscale=False`, but `make_wrapped_env` never
forwards the flag, so the SB3 path cannot produce RGB observations. The key is read only by
`dreamer_env.py`. Setting `grayscale: false` in a PPO config would be silently ignored —
another instance of 6.1.

**Fix:** Forward `grayscale` through `make_wrapped_env` and `make_sb3_env`. Note that
`FrameStack` + RGB yields a 4-D `(stack, H, W, 3)` observation SB3's `CnnPolicy` cannot
consume, so the combination should raise a clear error rather than fail deep inside SB3.

---

### 7. Cleanup & docs

#### 7.1 README and PLAN describe a codebase that doesn't exist  `medium` · `small`

- `README.md:37,81-83` — the `train.py` CLI (2.3).
- `README.md:24-42` — project structure lists five notebooks; six exist (`06_dreamer_training.ipynb` missing).
- `README.md:110-125` — "Algorithms" describes the dead hand-rolled implementations, not the SB3 configuration that actually runs (2.5).
- `PLAN.md:131` contradicts `PLAN.md:138` and `:216` on whether the CLI exists.
- `utils.py:32` — claims `setup_google_drive` is a no-op off Colab; it raises `ModuleNotFoundError` (2.7).
- README's "Google Colab" section says each notebook's setup cell is commented out; in five of six it is live code (2.4).
- `sb3_utils.py:644` — `train_sb3`'s docstring lists `metrics/progress.csv`, but the file is written to `tensorboard/progress.csv` (`:774`, read back at `:829`).
- `PLAN.md:186,208` — Phase 8 (Testing) is headed "**Status:** Not yet implemented" and the roadmap table lists Tests as "Not started"; the repo has 15 test modules, 126 tests, and a 3-version CI matrix.

**Fix:** Bring README in line with reality in one pass; fold `PLAN.md`/`DREAMER_PLAN.md`
status sections into a single `ROADMAP.md` (or move them to `docs/`) so there's one place
where "what's built" is asserted.

---

#### 7.2 `RESULTS.md` contradicts the matrix it tells you to run  `medium` · `small`

**Where:** `RESULTS.md:10-13` vs `configs/matrix/deadly_corridor_curriculum.yaml`

| `RESULTS.md` says | The matrix actually defines |
|---|---|
| Algorithms: PPO, DQN, Recurrent PPO | + DreamerV3 |
| 6 variants × 3 seeds = 18 runs | 8 variants × 3 seeds = **24 runs** |
| "2.5M for every variant" | Dreamer variants are 500K–1.5M (see 1.10) |

The headline results table is entirely `_pending_`, so the repo's flagship claim is
currently unsupported by data.

**Fix:** Regenerate the setup section from the matrix YAML rather than restating it by
hand — `scripts/run_experiment_matrix.py --dry-run` already prints the expanded grid.

---

#### 7.3 Smaller cleanups  `low` · `small`

- **`SCENARIO_DEFAULT_SKILL` is an empty dict** (`env.py:139`) carrying six lines of
  documentation and a branch in `DoomEnv.__init__`. Either populate it from
  `scenario_limits.py` or delete it.
- **A module-level `assert`** (`env.py:38`) enforces the `EPISODE_METRIC_GAME_VARS` /
  `EPISODE_METRIC_KEYS` invariant — stripped under `python -O`. Raise `RuntimeError` instead.
- **Nine forwarding properties** in `curriculum.py:381-419`, several documented as "legacy
  accessor" / "test-only accessor", for a package with no external consumers. `_stages`
  reaches into `self._state._stages` across an object boundary. Delete with 5.4.
- **`__version__` fallback `"0.1.2"`** is hard-coded in `__init__.py:19`, duplicating
  `pyproject.toml`. It will silently go stale on the next version bump.
- **`.gitignore` misses `stage_summary.txt`**, which `scripts/generate_summary.py:48` writes
  to the repo root — running the summary script dirties the working tree.
- **17 `print()` calls in library code** (`src/rl_doom/`) rather than `logging`, so a
  notebook cannot control verbosity independently of SB3's `verbose`.
- **Ruff selects only `E,F,W,I`.** Adding `B` (bugbear), `SIM`, `ARG`, `RUF`, and `UP`
  would flag several items in this review automatically — a broad-ruleset run surfaces 6
  `C901` complex-structure and 6 `PLR0915` too-many-statements hits concentrated in exactly
  the functions named in 2.1 and 2.2.
- **No dependency pinning.** No lockfile or constraints file; combined with 5.3 this means
  neither CI nor a reported result is reproducible from the repo alone.
- **`scripts/` is neither linted nor type-checked.** CI runs `ruff check src tests`
  (`ci.yml:38`) and `pyproject.toml:71` scopes mypy to `files = ["src", "tests"]`. Running
  `ruff check scripts` locally finds a real violation at
  `scripts/run_experiment_matrix.py:320` that CI has never reported — in the 379-line module
  that orchestrates the flagship 24-run experiment. Add `scripts` to both.
- **`pre-commit` is a declared dev dependency with no `.pre-commit-config.yaml`.** Installing
  `.[dev]` gives you a tool that does nothing.
- **`training.checkpoint_freq` is dead in all six `configs/dreamer_*.yaml`.** It is consumed
  only on the SB3 path (`sb3_utils.py:790`); `agents/dreamer.py` never reads it. A textbook
  instance of 6.1 — with no schema, a key that no longer means anything looks identical to
  one that does.
- **mypy is configured loosely.** `check_untyped_defs` is on but none of the `disallow_*`
  flags are, so an unannotated parameter silently becomes `Any` and its call sites go
  unchecked. Consider `disallow_untyped_defs` on `src/rl_doom` at minimum.

---

### 8. Performance

#### 8.1 Resize + grayscale runs four times per agent step; three of four results are discarded  `medium` · `small`

**Where:** `src/rl_doom/env.py:487-496`

**What:** `make_wrapped_env` composes `DoomEnv → ResizeObservation → SkipFrame → FrameStack`.
Because `ResizeObservation` sits **inside** `SkipFrame`, `cv2.cvtColor` + `cv2.resize` run on
every skipped frame, and `SkipFrame` returns only the last one.

**Verified by execution** with a stub env (`frame_skip=4`, 25 agent steps):

```
repo order (Resize INSIDE Skip): 25 agent steps -> 100 env steps, 100 resize calls
skip-then-resize               : 25 agent steps -> 100 env steps,  25 resize calls
```

Separately, `DoomEnv.__init__` accepts `frame_skip` and passes it to
`game.make_action(action, frame_skip)` — ViZDoom's *native* frame skip — but
`make_wrapped_env` never forwards it, so `DoomEnv` runs at `frame_skip=1` and the Python-level
`SkipFrame` loop does four full round trips instead of one native call.

**Why it matters:** at 320×240 RGB, that is 75% of all image-processing work discarded, on
the hot path of every rollout step, multiplied by `n_envs=8` and 2.5M steps.

**Fix:** Reorder to `DoomEnv → SkipFrame → ResizeObservation → FrameStack`, or better, pass
`frame_skip` into `DoomEnv` and drop the `SkipFrame` wrapper entirely so ViZDoom does the
skipping natively. Either change alters the observation stream only in that fewer frames are
resized — but it *does* change reward accumulation semantics if `SkipFrame` is removed
(`make_action` already sums reward over tics), so verify against a short baseline run before
adopting.

---

#### 8.2 `ReplayBuffer` stores each frame twice and samples a deque by index  `low` · `small`

**Where:** `src/rl_doom/replay_buffer.py:33,44`

**What:** Each transition stores both `obs` and `next_obs` in full. At the default 50,000
capacity with `(4, 84, 84)` `uint8` frames that is 2 × 28,224 B × 50,000 ≈ **2.8 GB**, where a
frame-sharing circular buffer would need ~1.4 GB. `random.sample` is called on a `deque`,
whose `__getitem__` is O(n) for interior indices, so sampling is O(batch × capacity).

**Scope:** this is part of the legacy stack (2.5) — the live DQN path uses SB3's own buffer.
Fix it only if the hand-rolled stack is kept rather than deleted.

---

## Recommended sequencing

Scoped for a solo maintainer. Wave 1 is a weekend; waves 2–3 are incremental.

### Wave 1 — unblock, then guard  ✅ done

All landed. Kept here as the record of what was done and in what order; items 1–3 are the
ones that stopped a run from being wrong or from running at all.

1. **1.5** — pass `optimize_memory_usage=True` for DQN and add a startup RAM check. *(one line + an assertion; without it, 6 of the 24 matrix runs cannot execute)*
2. **1.10 + 1.6** — match the two Dreamer curriculum budgets to their baselines; guard the `doom_skill` override so the deathmatch curricula stop discarding it. *(under an hour; unblocks two confounded experiment rows)*
3. **1.4** — one `resolve_algo_class` registry replacing all three dispatch sites. *(currently notebook 05 cannot load any Recurrent PPO run)*
4. **2.4** — replace the "uncomment this" Colab cells with an `IN_COLAB` guard in all five notebooks. *(20 minutes; restores the documented local workflow)*
5. **5.1 (partial)** — extend the existing `fake_vizdoom` stub to cover `DoomEnv` and port `test_env.py` onto it. *(half a day; every fix below then has a CI guard)*
6. **1.1** — split `terminated`/`truncated` on timeout, plus the last-real-frame fix. *(~10 lines + the regression test)*
7. **1.3** — delete `DEATHMATCH_ACTIONS`, import from `env.py`, add the action-space test.
8. **4.1** — `try/except/finally` around `model.learn()`; env teardown in `finally`.
9. **4.3** — `try/except` per matrix cell, incremental CSV writes, `--resume`.
10. **5.2 + 5.3** — move the `importorskip` to the top of `test_agents.py` (or split the RecurrentPPO tests out), install CI deps from `pyproject` extras so `sb3-contrib` is present, and add a constraints file. *(recovers ~30 silently-disabled tests)*

### Wave 2 — consolidation (~3–5 days)

10. **1.8** — rebuild notebook 05's `make_env` from each run's `config.json`; thread `seed` through. *(makes the comparison table mean what it claims)*
11. **3.2 + 2.3** — one `run_from_config()`, then `rl_doom/train.py` on top of it. Repoint all four notebooks and the matrix runner at it. Fixes README's quick-start as a side effect.
12. **6.1 + 3.1** — `RunSpec` dataclass with unknown-key rejection and `extends:`. Collapse the 8 curriculum configs. *(makes 1.6/1.10 structurally impossible)*
13. **2.7 + 7.1 + 7.2 + 7.3** — delete `utils.py` and the dead `paths.py` sweep helpers; reconcile README / PLAN / RESULTS in one pass. Audit every comment that promises a capability (see the executive summary's last paragraph) and either implement or delete the claim.
14. **1.2 + 2.5** — decide the fate of the hand-rolled stack. If deleting, 1.2 and 8.2 go with it; if keeping, fix 1.2 and move it to `rl_doom/reference/`.
15. **8.1** — reorder the wrapper stack, validated against a short baseline run.
16. **1.7** — settle the deathmatch respawn question with a real run, then add `respawn_on_death` to `DoomEnv` if confirmed. *(needs the ViZDoom binary; schedule alongside another manual run)*

### Wave 3 — structure and reliability (~1–2 weeks, incremental)

17. **2.1 + 2.2** — split `sb3_utils.py` and `agents/dreamer.py`. Move code verbatim first, refactor after, so the diff stays reviewable.
18. **3.3 + 3.4** — shared env helpers, one `unwrap_to`, one video writer, one plot module.
19. **4.2 + 4.6** — real resume for both drivers: checkpoint + `VecNormalize` + curriculum stage + step counter + `reset_num_timesteps=False`.
20. **6.2** — one algorithm/scenario registry.
21. **2.6 + 4.5** — give self-play/tournament an entry point and an integration test (with the init timeout and port retry), or park the subsystem.
22. **5.4** — numeric RL-math tests; drop the "test-only" forwarding properties.
23. **1.9** — decide whether `min_evals_between_promotions` means spacing or consecutive evals, then make code and configs agree.
24. **4.4 + 4.7** — narrow the `except Exception` blocks; stream video frames instead of buffering.
