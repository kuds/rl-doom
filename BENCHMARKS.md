# Scenario Benchmarks

External reference numbers for the four ViZDoom scenarios this repo
trains (`basic`, `deadly_corridor`, `defend_the_center`, `deathmatch`),
plus what "good" looks like on each metric so the numbers in
`stage_summary.txt` land in a frame of reference rather than as bare
scalars.

All time conversions assume the repo default of `frame_skip=4` and the
stock Doom tic rate of 35 tics/s, so **1 env-step = 4 tics ≈ 0.114 s**.

## Quick reference

| Scenario            | Episode timeout (tics / env-steps / sec) | "Good" reward target | Primary metric |
|---------------------|---:|---:|---|
| basic               | 300 / 75 / 8.6 s    | 85–95             | reward                                           |
| deadly_corridor     | 2100 / 525 / 60 s   | ≥1500 (skill 3), ~2200 = corridor cleared | reward + success rate (goal_reached fraction) |
| defend_the_center   | 2100 / 525 / 60 s   | ≥20 (theoretical max 25) | reward (= kills − 1)                  |
| deathmatch          | 4200 / 1050 / 120 s | ≥8 frags vs 8 bots | frags; mean time between deaths                 |

## basic

- **Timeout:** 300 tics (75 env-steps, ≈ 8.6 s of Doom time).
- **Reward structure:** +101 for killing the single enemy, −1/tic alive,
  −5 per miss.
- **Sanity-check scenario.** If the pipeline is wired correctly, any
  of PPO / DQN / Recurrent PPO should converge in well under 100K
  training steps.
- **Good reward:** mean eval in the **85–95** band. Anything below ~50
  after 50K steps means the training loop is broken (obs misalignment,
  frozen gradients, etc.), not a hyperparameter issue.

## deadly_corridor

- **Timeout:** 2100 tics (525 env-steps, ≈ 60 s).
- **Reward structure:** +dX per tic of forward progress (distance
  shaping), −100 on death, ≈ +500 on reaching the green vest at the
  end of the corridor. A completed run lands around **2200–2300**
  total reward.
- **Per-skill record scores** from Khan et al. 2025 (PPO + curriculum +
  reward shaping):

  | Skill | Record reward |
  |---:|---:|
  | 1 | 734  |
  | 2 | 1576 |
  | 3 | 1920 |
  | 4 | 2280 |
  | 5 | 1605 |

  Skill 3 is the budgeted default in the baseline YAMLs. A curriculum
  that ramps 1 → 2 → 3 uses these as the upper envelope on each rung;
  the 1500 gates in the curriculum YAMLs are calibrated for the skill
  2 band before stepping to skill 3.

- **Reward-shaping baseline** (CEUR-WS paper 3094) hits the max
  reward in roughly 1M frames when shaping is applied. Without
  shaping or curriculum, training at skill 3 is dominated by the
  death penalty — the agent rarely learns to push forward.
- **Good reward:** on skill 3, **≥ 1500** consistently indicates the
  agent reaches the vest on most episodes; **2200+** means it's
  near-record. Complement reward with the `Success rate` line in
  `stage_summary.txt` (goal_reached / total episodes) — reward
  plateaus can hide kill-and-die local optima.

## defend_the_center

- **Timeout:** 2100 tics (525 env-steps, ≈ 60 s).
- **Reward structure:** +1 per kill, −1 on death. Ammo is capped at
  26 rounds, so the agent always runs out and dies.
- **Theoretical max:** 25 (one kill per bullet, minus the terminal
  death penalty).
- **Good reward:** **≥ 20 mean eval** is the "competent agent" bar.
  Expert policies land at **22–24**. Above 20 means the agent is
  essentially one-shotting each melee imp.
- **Episode length** is bounded by the speed at which enemies close
  the gap — typical competent runs last 30–60 seconds before the
  ammo pool empties.

## deathmatch

- **Timeout:** 4200 tics (1050 env-steps, ≈ 2 min of Doom time).
- **Reward structure:** frag count (kills − suicides). No built-in
  dense shaping — this is the sparsest scenario in the set.
- **Frag bands** (reference: Kieliger's PPO + Arnold / ViZDoom
  competition agents):

  | Frags per episode | Label |
  |---:|---|
  | 0–2   | "Agent is learning something"  (Kieliger PPO at 2M steps) |
  | 5–8   | "Decent"                        |
  | 10+   | "Strong"                        |
  | 15+   | Competition winners (Arnold, IntelAct) vs built-in bots |

- **Survival time is *not* a useful per-episode benchmark here** — the
  episode runs to the 2-minute timeout regardless of how many times
  the agent dies (it respawns). The informative survival metric is
  **mean time between deaths**: `episode_tics / (deaths + 1)`. A good
  agent should hold 20+ seconds per life.
- **Training budget:** Kieliger's baseline PPO reached ~2 frags after
  2M env-steps against 8 bots. Strong published agents train for
  10–50M frames with reward shaping (kill bonus, health/ammo pickups
  as extrinsic signals) to escape the sparse-reward regime.

## Sources

- [ViZDoom default scenarios — Farama docs](https://vizdoom.farama.org/environments/default/) — per-scenario episode timeouts and reward structure
- [Khan et al. 2025 — Optimizing RL Agents in Games Using Curriculum Learning and Reward Shaping (Wiley)](https://onlinelibrary.wiley.com/doi/abs/10.1002/cav.70008) — per-skill deadly_corridor record scores (734 / 1576 / 1920 / 2280 / 1605)
- [Kieliger — Deep RL in practice by playing Doom, Part 2](https://lkieliger.medium.com/deep-reinforcement-learning-in-practice-by-playing-doom-part-2-increasing-complexity-6510e7e5c3af) — defend_the_center "≥ 20" target; deathmatch ~2 frags at 2M steps
- [Reward Shaping for Deep RL in VizDoom (CEUR-WS 3094)](https://ceur-ws.org/Vol-3094/paper_17.pdf) — reward shaping converges to max on deadly_corridor in ~1M frames
- [ViZDoom Competitions: Playing Doom from Pixels (arXiv 1809.03470)](https://ar5iv.labs.arxiv.org/html/1809.03470) — deathmatch competition context, Arnold / IntelAct frag scores
- [Benchmarking RL Algorithms on ViZDoom scenarios (ResearchGate)](https://www.researchgate.net/publication/377671572_Using_VizDoom_Research_Platform_Scenarios_for_Benchmarking_Reinforcement_Learning_Algorithms_in_First-Person_Shooter_Games)
- [Farama ViZDoom defend_the_center.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/master/scenarios/defend_the_center.cfg) — confirms +1 per kill, −1 on death, 26-round ammo cap
