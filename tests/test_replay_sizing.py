"""DQN replay-buffer memory estimation and the startup guard.

The shipped DQN configs asked for 5.3-15.8 GiB of replay against Colab's
~12.7 GiB, and nothing caught it: SB3 stores ``observations`` and
``next_observations`` as separate arrays, and its own memory check is skipped
entirely when ``psutil`` is not installed (which it is not by default). The
observed failure was a bare numpy MemoryError minutes into a run.

``optimize_memory_usage=True`` would halve the cost but SB3 rejects it
alongside ``handle_timeout_termination=True``, which the pipeline needs so a
scenario timeout bootstraps ``V(s')`` rather than being treated as absorbing.
So the fix is to size the buffers honestly and check before allocating.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml

from rl_doom.sb3_utils import check_replay_buffer_fits, estimate_replay_bytes

STACKED_84 = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)

# Ceiling on what a shipped config may *ask* for, not a claim about any
# particular machine.
#
# The per-machine question is answered at runtime by check_replay_buffer_fits,
# which compares the estimate against real available RAM and fails fast with
# the size that would fit. Hardcoding a specific tier's RAM here was a mistake:
# these experiments run on an NVIDIA L4 Colab runtime (~53 GiB system RAM) and
# completed at buffer_size=300000, so a 5 GiB test budget would have rejected
# working configs.
#
# 24 GiB leaves the L4 runtime ample headroom for the model, the eval env and
# n_envs ViZDoom processes, while still catching an accidental extra zero.
REPLAY_BUDGET_BYTES = 24 * 2**30

# Resolved from this file rather than the working directory: a cwd-relative
# glob returns nothing when pytest is invoked from elsewhere, and an empty
# parametrize list collects zero tests and reports green.
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DQN_CONFIGS = sorted(CONFIG_DIR.glob("dqn_*.yaml"))
assert DQN_CONFIGS, f"no DQN configs found under {CONFIG_DIR}"


def test_estimate_accounts_for_the_next_obs_copy() -> None:
    """SB3 keeps obs and next_obs separately, so cost is 2x the frame stack."""
    one_obs = 4 * 84 * 84
    assert estimate_replay_bytes(1000, STACKED_84) == 1000 * one_obs * 2


def test_estimate_matches_sb3_actual_allocation() -> None:
    """Pin the estimate to what SB3 really allocates, not to our arithmetic."""
    from stable_baselines3.common.buffers import ReplayBuffer

    buf = ReplayBuffer(
        1000, STACKED_84, gym.spaces.Discrete(3), device="cpu", n_envs=1,
    )
    actual = buf.observations.nbytes + buf.next_observations.nbytes
    assert estimate_replay_bytes(1000, STACKED_84) == actual


def test_guard_raises_before_allocating_when_too_large() -> None:
    """A buffer larger than memory must fail fast, with the numbers in it."""
    huge = 100_000_000  # ~5.1 PiB — larger than any machine
    with pytest.raises(MemoryError) as excinfo:
        check_replay_buffer_fits(huge, STACKED_84)
    msg = str(excinfo.value)
    assert "buffer_size=100,000,000" in msg
    # The message must say what to do, not just that it failed.
    assert "Lower `hyperparams.buffer_size`" in msg


def test_guard_passes_for_a_reasonable_buffer() -> None:
    check_replay_buffer_fits(1_000, STACKED_84)


@pytest.mark.parametrize("config_path", DQN_CONFIGS, ids=lambda p: p.name)
def test_shipped_dqn_configs_stay_under_the_budget(config_path: Path) -> None:
    """Catch an implausible buffer_size — a stray zero, not a tight fit.

    Deliberately generous. Whether a config fits *this* machine is decided at
    runtime by check_replay_buffer_fits against real available memory; a test
    cannot know what hardware the run will land on.
    """
    cfg = yaml.safe_load(config_path.read_text())
    env_cfg = cfg["env"]
    height, width = env_cfg["resize_shape"]
    num_stack = env_cfg.get("num_stack", 4)
    obs_space = gym.spaces.Box(0, 255, (num_stack, height, width), dtype=np.uint8)

    needed = estimate_replay_bytes(cfg["hyperparams"]["buffer_size"], obs_space)
    assert needed <= REPLAY_BUDGET_BYTES, (
        f"{config_path.name} asks for {needed / 2**30:.1f} GiB of replay "
        f"(buffer_size={cfg['hyperparams']['buffer_size']:,}), over the "
        f"{REPLAY_BUDGET_BYTES / 2**30:.0f} GiB sanity ceiling."
    )


@pytest.mark.parametrize("config_path", DQN_CONFIGS, ids=lambda p: p.name)
def test_dqn_buffer_not_larger_than_the_run(config_path: Path) -> None:
    """A buffer bigger than the whole run can never fill — pure wasted RAM."""
    cfg = yaml.safe_load(config_path.read_text())
    buffer_size = cfg["hyperparams"]["buffer_size"]
    total_timesteps = cfg["training"]["total_timesteps"]
    assert buffer_size <= total_timesteps, (
        f"{config_path.name}: buffer_size={buffer_size:,} exceeds "
        f"total_timesteps={total_timesteps:,}"
    )
