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
