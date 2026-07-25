"""Scenario-wide constants shared between the env and curriculum layers.

These live in their own tiny module so that :mod:`rl_doom.curriculum` can
perform parse-time validation without transitively importing the ViZDoom
binary — :mod:`rl_doom.env` pulls ``vizdoom`` in at module load time, and
lean CI runners (plus the dedicated curriculum tests) deliberately run
without it.
"""

from __future__ import annotations

# Upper bound on ZDoom AI bots spawned per episode via ``addbot``. The stock
# deathmatch.cfg ships an 8-entry bot roster, so requesting more bots either
# recycles names or silently drops them; capping here makes the failure mode
# explicit instead of scenario-dependent.
MAX_NUM_BOTS: int = 8


# Per-episode combat/exploration metrics surfaced in ``DoomEnv``'s info dict
# on the terminal step and logged through Monitor -> training.npz. The keys
# here are the ``info`` dict field names and the column names used across
# monitor CSVs, ``termination_episodes.csv``, ``training.npz``
# (``episode_<key>``), the video playback JSON, and ``stage_summary.txt``.
#
# The mapping from these keys to ``vizdoom.GameVariable`` members lives in
# :mod:`rl_doom.env`, which is what requires the vizdoom import. We keep the
# key list here so downstream consumers (``summary.py``,
# ``sb3_utils._collect_episode_stats``) can iterate without having to import
# vizdoom on lean CI runners or offline analysis boxes.
EPISODE_METRIC_KEYS: tuple[str, ...] = (
    "kills",
    "damage_dealt",
    "damage_taken",
    "hits_dealt",
    "hits_taken",
    "items",
    "secrets",
)

# Curated action sets per scenario. Each inner list is the set of
# simultaneously pressed buttons for one discrete action, and the list order
# *is* the action space — index N means the same buttons everywhere.
#
# The original ``np.eye`` one-hot layout (one button pressed per action) cannot
# express compound actions that are required to play these scenarios well —
# e.g. "move forward while shooting" in Deadly Corridor or "turn while firing"
# in Defend the Center. Without compound actions the PPO policy on Deadly
# Corridor collapses to a deterministic "die faster" strategy because no
# single-button policy can survive the corridor.
#
# Button names map 1:1 to ``vizdoom.Button`` enum member names, but are plain
# strings here so this table stays importable without the ViZDoom binary. Any
# name not in a scenario's ``available_buttons`` raises at env init.
#
# This lives in scenario_limits rather than env.py because both
# :mod:`rl_doom.env` and :mod:`rl_doom.multiplayer_env` build action tables
# from it, and they must agree exactly: the multiplayer docstring promises
# that single-player checkpoints load directly for self-play, which requires
# an identical index -> button mapping. They previously kept separate copies
# that had drifted in both length and ordering.
SCENARIO_ACTION_SETS: dict[str, list[list[str]]] = {
    "basic": [
        ["MOVE_LEFT"],
        ["MOVE_RIGHT"],
        ["ATTACK"],
        # Strafe while firing — lets the agent track a moving target.
        ["MOVE_LEFT", "ATTACK"],
        ["MOVE_RIGHT", "ATTACK"],
    ],
    "deadly_corridor": [
        # Pure movement / turning (still useful for navigation + aiming).
        ["MOVE_FORWARD"],
        ["MOVE_BACKWARD"],
        ["TURN_LEFT"],
        ["TURN_RIGHT"],
        ["MOVE_LEFT"],
        ["MOVE_RIGHT"],
        # Stationary attack.
        ["ATTACK"],
        # The compound actions that actually let the agent survive:
        # push forward while firing / strafing / aiming.
        ["MOVE_FORWARD", "ATTACK"],
        ["MOVE_FORWARD", "MOVE_LEFT"],
        ["MOVE_FORWARD", "MOVE_RIGHT"],
        ["MOVE_FORWARD", "TURN_LEFT"],
        ["MOVE_FORWARD", "TURN_RIGHT"],
        # Fire while turning to track imps on either side.
        ["TURN_LEFT", "ATTACK"],
        ["TURN_RIGHT", "ATTACK"],
    ],
    "defend_the_center": [
        ["TURN_LEFT"],
        ["TURN_RIGHT"],
        ["ATTACK"],
        # Killer compound: rotate while firing to sweep the arena.
        ["TURN_LEFT", "ATTACK"],
        ["TURN_RIGHT", "ATTACK"],
    ],
    "deathmatch": [
        # Deathmatch needs both navigation and combat. We restrict to the
        # binary movement/attack buttons and skip the delta + weapon-select
        # buttons in deathmatch.cfg — delta buttons expect continuous values
        # rather than a 0/1 press, and weapon switching adds a large branching
        # factor that's better left for a policy with a more expressive
        # action space. SPEED is included so the agent can run while chasing.
        ["MOVE_FORWARD"],
        ["MOVE_BACKWARD"],
        ["TURN_LEFT"],
        ["TURN_RIGHT"],
        ["MOVE_LEFT"],
        ["MOVE_RIGHT"],
        ["ATTACK"],
        # Fire while moving/turning to track and engage opponents.
        ["MOVE_FORWARD", "ATTACK"],
        ["MOVE_LEFT", "ATTACK"],
        ["MOVE_RIGHT", "ATTACK"],
        ["TURN_LEFT", "ATTACK"],
        ["TURN_RIGHT", "ATTACK"],
        # Navigate while aiming.
        ["MOVE_FORWARD", "TURN_LEFT"],
        ["MOVE_FORWARD", "TURN_RIGHT"],
        # Sprint for closing distance / escaping.
        ["SPEED", "MOVE_FORWARD"],
        ["SPEED", "MOVE_FORWARD", "ATTACK"],
    ],
}


# Human-readable labels rendered in ``stage_summary.txt``. Kept alongside
# the keys so the summary lines stay in lock-step with the metric list.
EPISODE_METRIC_LABELS: dict[str, str] = {
    "kills": "Kills",
    "damage_dealt": "Damage dealt",
    "damage_taken": "Damage taken",
    "hits_dealt": "Hits landed",
    "hits_taken": "Hits taken",
    "items": "Items picked up",
    "secrets": "Secrets found",
}
