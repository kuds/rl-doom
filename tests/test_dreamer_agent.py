"""Unit tests for ``rl_doom.agents.dreamer``.

These exercise the test-friendly paths only — config building, helper
functions for episode scanning / metric parsing / artefact synthesis. The
full ``train_dreamer`` driver requires the upstream ``NM512/dreamerv3-torch``
checkout *and* the ViZDoom binary, neither of which is assumed in CI; that
end-to-end path is exercised in the notebook instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rl_doom.agents.dreamer import (
    _DOOM_FIXED_OVERRIDES,
    UPSTREAM_PIN,
    _build_config,
    _import_port,
    _read_metrics_jsonl,
    _recursive_update,
    _scan_episode_npzs,
    _synthesize_evaluations_npz,
    _to_namespace,
)

# ---------------------------------------------------------------------------
# Constants + import contract
# ---------------------------------------------------------------------------


def test_upstream_pin_is_a_full_sha() -> None:
    """The pin should be a 40-char hex SHA so ``git checkout`` is unambiguous."""
    assert len(UPSTREAM_PIN) == 40
    int(UPSTREAM_PIN, 16)  # raises if not hex


def test_import_port_raises_friendly_error_on_missing_checkout(tmp_path: Path) -> None:
    """``_import_port`` should mention the pin + clone command on failure."""
    with pytest.raises(RuntimeError) as exc:
        _import_port(tmp_path)
    msg = str(exc.value)
    assert UPSTREAM_PIN in msg
    assert "git clone" in msg
    # Lists which files were missing — useful when someone forgets configs.yaml.
    for required in ("models.py", "tools.py", "configs.yaml"):
        assert required in msg


def test_import_port_partial_checkout_still_raises(tmp_path: Path) -> None:
    """If models.py exists but tools.py doesn't, we should still complain."""
    (tmp_path / "models.py").write_text("# stub")
    with pytest.raises(RuntimeError) as exc:
        _import_port(tmp_path)
    assert "tools.py" in str(exc.value)


# ---------------------------------------------------------------------------
# Recursive merge + namespace helpers
# ---------------------------------------------------------------------------


def test_recursive_update_overrides_and_preserves_unrelated_keys() -> None:
    base: dict = {"a": 1, "b": {"c": 2, "d": 3}, "e": [1, 2]}
    _recursive_update(base, {"a": 10, "b": {"c": 20}, "f": "new"})
    assert base == {
        "a": 10,
        "b": {"c": 20, "d": 3},  # unrelated nested key preserved
        "e": [1, 2],
        "f": "new",
    }


def test_recursive_update_replaces_dict_with_scalar() -> None:
    """Updating a dict-valued key with a non-dict value should replace, not merge."""
    base: dict = {"a": {"x": 1}}
    _recursive_update(base, {"a": "scalar"})
    assert base == {"a": "scalar"}


def test_to_namespace_attribute_access() -> None:
    ns = _to_namespace({"foo": 1, "bar": "hi"})
    assert ns.foo == 1 and ns.bar == "hi"


# ---------------------------------------------------------------------------
# Config builder (uses a fake configs.yaml so the test is hermetic)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_port(tmp_path: Path) -> Path:
    """Lay out a minimal port checkout so ``_build_config`` can read configs.yaml.

    We don't need the actual Python sources here — ``_build_config`` only
    reads ``configs.yaml`` directly and never imports ``models``/``tools``
    (that's :func:`_import_port`'s job, exercised separately).
    """
    (tmp_path / "configs.yaml").write_text(
        """
defaults:
  steps: 1000000
  envs: 1
  action_repeat: 2
  batch_size: 16
  batch_length: 64
  train_ratio: 512
  imag_horizon: 15
  size: [64, 64]
  grayscale: false
  pretrain: 100
  prefill: 2500
  log_every: 10000
  eval_every: 10000
  eval_episode_num: 10
  reset_every: 0
  expl_until: 0
  expl_behavior: greedy
  dataset_size: 1000000
  parallel: false
  compile: true
  video_pred_log: true
  reward_EMA: true
  actor: {dist: 'onehot', std: 'none'}

atari100k:
  steps: 400000
  action_repeat: 4
  train_ratio: 1024
  actor: {dist: 'onehot', std: 'none'}
  video_pred_log: true
""".strip(),
    )
    # Drop empty stubs so :func:`_import_port` (used elsewhere in tests) sees
    # a complete checkout if it walks this dir later. ``_build_config`` itself
    # does not need them.
    for name in ("models.py", "networks.py", "tools.py", "exploration.py"):
        (tmp_path / name).write_text("# test stub")
    return tmp_path


def test_build_config_merges_preset_over_defaults(fake_port: Path) -> None:
    cfg = _build_config(
        port_path=fake_port,
        scenario="defend_the_center",
        env_cfg={"resize_shape": [64, 64], "frame_skip": 4, "grayscale": False},
        hyperparams={"preset": "atari100k", "batch_size": 16, "batch_length": 64,
                     "train_ratio": 512, "imag_horizon": 15},
        eval_cfg={"eval_freq": 5000, "n_episodes": 7},
        training_cfg={"total_timesteps": 50000},
        seed=42,
        device="cpu",
        logdir=fake_port / "logs",
    )
    # atari100k preset wins over defaults.
    assert cfg.train_ratio == 512  # YAML hyperparam beats preset's 1024
    # env_cfg overrides win over preset.
    assert cfg.action_repeat == 4
    assert cfg.size == (64, 64)
    # per-call wiring.
    assert cfg.steps == 50000
    assert cfg.eval_every == 5000
    assert cfg.eval_episode_num == 7
    assert cfg.seed == 42
    assert cfg.device == "cpu"
    # Doom-fixed overrides win over everything.
    assert cfg.envs == 1
    assert cfg.parallel is False
    assert cfg.compile is False
    assert cfg.video_pred_log is False
    assert cfg.expl_behavior == "greedy"
    # Scenario tag for provenance.
    assert cfg.scenario == "defend_the_center"


def test_build_config_unknown_preset_raises(fake_port: Path) -> None:
    with pytest.raises(ValueError) as exc:
        _build_config(
            port_path=fake_port,
            scenario="basic",
            env_cfg={},
            hyperparams={"preset": "no_such_preset"},
            eval_cfg={},
            training_cfg={"total_timesteps": 1000},
            seed=0,
            device="cpu",
            logdir=fake_port / "logs",
        )
    msg = str(exc.value)
    assert "no_such_preset" in msg
    assert "atari100k" in msg  # mentions the available presets


def test_build_config_grayscale_propagates(fake_port: Path) -> None:
    cfg = _build_config(
        port_path=fake_port,
        scenario="basic",
        env_cfg={"grayscale": True, "resize_shape": [64, 64], "frame_skip": 4},
        hyperparams={"preset": "atari100k"},
        eval_cfg={},
        training_cfg={"total_timesteps": 1000},
        seed=0,
        device="cpu",
        logdir=fake_port / "logs",
    )
    assert cfg.grayscale is True


def test_doom_fixed_overrides_pin_single_env() -> None:
    """Sanity check: the fixed-override dict locks ``envs=1`` + ``parallel=False``."""
    assert _DOOM_FIXED_OVERRIDES["envs"] == 1
    assert _DOOM_FIXED_OVERRIDES["parallel"] is False


# ---------------------------------------------------------------------------
# Helpers — episode npz scan, metrics.jsonl parser, evaluations synthesis
# ---------------------------------------------------------------------------


def test_scan_episode_npzs_aggregates_rewards_and_lengths(tmp_path: Path) -> None:
    np.savez(tmp_path / "ep1.npz", reward=np.array([0.0, 1.0, 2.0, 3.0]))
    np.savez(tmp_path / "ep2.npz", reward=np.array([0.0, -1.0]))
    rewards, lengths = _scan_episode_npzs(tmp_path)
    assert rewards == [6.0, -1.0]
    assert lengths == [3, 1]


def test_scan_episode_npzs_skips_corrupted_file(tmp_path: Path) -> None:
    np.savez(tmp_path / "ok.npz", reward=np.array([0.0, 1.0]))
    (tmp_path / "broken.npz").write_bytes(b"not a npz file")
    rewards, lengths = _scan_episode_npzs(tmp_path)
    assert rewards == [1.0]
    assert lengths == [1]


def test_scan_episode_npzs_handles_missing_dir(tmp_path: Path) -> None:
    """Pre-training scan returns empty lists rather than raising."""
    rewards, lengths = _scan_episode_npzs(tmp_path / "does-not-exist")
    assert rewards == []
    assert lengths == []


def test_read_metrics_jsonl_alignment(tmp_path: Path) -> None:
    p = tmp_path / "metrics.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"step": 1000, "kl_loss": 0.5, "model_loss": 1.2}),
                json.dumps({"step": 2000, "kl_loss": 0.4, "model_loss": 1.0,
                            "value_loss": 3.1}),
            ],
        ),
    )
    series = _read_metrics_jsonl(p)
    assert series["kl_loss"] == [0.5, 0.4]
    assert series["kl_loss__step"] == [1000.0, 2000.0]
    # value_loss appears only on the second row, so its step list is shorter.
    assert series["value_loss"] == [3.1]
    assert series["value_loss__step"] == [2000.0]


def test_read_metrics_jsonl_skips_blank_and_invalid(tmp_path: Path) -> None:
    p = tmp_path / "metrics.jsonl"
    p.write_text(
        "\n".join(
            [
                "",                                          # blank
                "not json",                                  # invalid
                json.dumps({"step": 1000, "kl_loss": 0.5}),
                json.dumps({"step": 2000, "kl_loss": [1, 2]}),  # non-numeric value
            ],
        ),
    )
    series = _read_metrics_jsonl(p)
    assert series["kl_loss"] == [0.5]


def test_read_metrics_jsonl_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _read_metrics_jsonl(tmp_path / "nope.jsonl") == {}


def test_synthesize_evaluations_npz_matches_sb3_schema(tmp_path: Path) -> None:
    """The npz layout must match what SB3's ``EvalCallback`` writes.

    ``rl_doom.sb3_utils._plot_eval_performance`` and
    ``rl_doom.summary.generate_run_summary`` consume ``timesteps`` (1-D),
    ``results`` (n_evals, n_eval_eps), and ``ep_lengths`` (same shape) —
    keep them in sync or those plotters silently break on dreamer runs.
    """
    out = tmp_path / "evaluations.npz"
    _synthesize_evaluations_npz(
        [
            (1000, [10.0, 20.0], [50, 60]),
            (2000, [30.0, 40.0], [70, 80]),
        ],
        out,
    )
    with np.load(out) as data:
        assert data["timesteps"].tolist() == [1000, 2000]
        assert data["results"].tolist() == [[10.0, 20.0], [30.0, 40.0]]
        assert data["ep_lengths"].tolist() == [[50, 60], [70, 80]]


def test_synthesize_evaluations_npz_skips_when_history_empty(tmp_path: Path) -> None:
    out = tmp_path / "evaluations.npz"
    _synthesize_evaluations_npz([], out)
    assert not out.exists(), "should not write an npz when there's no eval history"
