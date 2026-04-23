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
