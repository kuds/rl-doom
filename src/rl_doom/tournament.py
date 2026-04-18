"""Round-robin tournaments for ViZDoom self-play checkpoints.

Given a directory of SB3 ``.zip`` snapshots (e.g. the opponent pool produced
by :class:`rl_doom.self_play.SelfPlaySnapshotCallback`), this module plays
each pair head-to-head in a 1v1 deathmatch, tallies frag scores, and derives
an Elo ranking.

Design choices:

* **1v1 only.** Round-robin over 4-player team matches adds two axes of
  teammate pairing that obscure per-policy skill. A separate 2v2 driver can
  be layered on later using the same :class:`MatchResult` primitive.
* **Frag-count scoring, not per-step reward.** A match result is the total
  frag count each side scored; the winner is whoever has more frags. Ties
  become 0.5/0.5 Elo updates. This mirrors how deathmatch is naturally scored
  and keeps the signal interpretable.
* **Seat-flipping.** By default each pair plays twice - once with checkpoint
  A as ``player_0`` and once as ``player_1`` - so map asymmetries don't bias
  the ranking. Results are normalised back to the canonical (A, B) ordering
  before Elo updates.
* **Lazy imports of ``vizdoom`` / ``matplotlib``.** The pure Elo math and
  round-robin pairing logic are importable without the native binaries so
  unit tests can exercise them standalone.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from rl_doom.multiplayer_env import DoomMultiplayerEnv


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """Outcome of a single 1v1 match.

    Frag counts are stored in the canonical (A, B) order regardless of which
    seat each checkpoint actually occupied in the game, so downstream Elo /
    aggregation code doesn't have to care about seat swaps.
    """

    player_a: str
    player_b: str
    frags_a: int
    frags_b: int
    steps: int = 0

    @property
    def winner(self) -> str | None:
        if self.frags_a > self.frags_b:
            return self.player_a
        if self.frags_b > self.frags_a:
            return self.player_b
        return None


@dataclass
class EloRating:
    """Current Elo state for a single checkpoint."""

    name: str
    rating: float
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def _default_loader() -> Callable[[str], Any]:
    """Lazy import of ``stable_baselines3.PPO.load``.

    Matches :func:`rl_doom.self_play._default_loader` so tournament and
    self-play share the same checkpoint interchange format.
    """
    from stable_baselines3 import PPO

    return PPO.load


# ---------------------------------------------------------------------------
# Match play
# ---------------------------------------------------------------------------


def play_match(
    model_a: Any,
    model_b: Any,
    *,
    name_a: str = "A",
    name_b: str = "B",
    env: "DoomMultiplayerEnv | None" = None,
    env_kwargs: dict[str, Any] | None = None,
    deterministic: bool = True,
    max_steps: int | None = None,
    seed: int | None = None,
) -> MatchResult:
    """Run a single 1v1 deathmatch between two policies.

    Parameters
    ----------
    model_a, model_b : Any
        Objects exposing ``.predict(obs, deterministic=...) -> (action, state)``
        (SB3 models, :class:`rl_doom.self_play.RandomPolicy`, or any duck).
        ``model_a`` plays ``player_0``, ``model_b`` plays ``player_1``.
    name_a, name_b : str
        Labels recorded on the returned :class:`MatchResult`.
    env : DoomMultiplayerEnv | None
        Pre-built 1v1 env. If ``None``, builds one via
        :func:`rl_doom.multiplayer_env.make_1v1_env` and closes it on exit.
    env_kwargs : dict | None
        Keyword args forwarded to :func:`make_1v1_env` when ``env`` is not
        provided.
    deterministic : bool
        Forwarded to ``.predict``. Deterministic by default to keep tournament
        results reproducible.
    max_steps : int | None
        Hard cap on env steps; ``None`` lets the match's native timelimit
        terminate it.
    seed : int | None
        Reset seed for the underlying env.
    """
    owns_env = env is None
    if owns_env:
        from rl_doom.multiplayer_env import make_1v1_env

        env = make_1v1_env(**(env_kwargs or {}))
    assert env is not None  # for type-checkers

    try:
        obs, _ = env.reset(seed=seed)
        steps = 0
        while env.agents:
            action_a, _ = model_a.predict(
                obs["player_0"], deterministic=deterministic,
            )
            action_b, _ = model_b.predict(
                obs["player_1"], deterministic=deterministic,
            )
            obs, _, terms, truncs, infos = env.step(
                {
                    "player_0": int(np.asarray(action_a).item()),
                    "player_1": int(np.asarray(action_b).item()),
                },
            )
            steps += 1
            if all(terms.values()) or all(truncs.values()):
                break
            if max_steps is not None and steps >= max_steps:
                break

        frags_a = int(infos["player_0"].get("frags", 0))
        frags_b = int(infos["player_1"].get("frags", 0))
        return MatchResult(
            player_a=name_a,
            player_b=name_b,
            frags_a=frags_a,
            frags_b=frags_b,
            steps=steps,
        )
    finally:
        if owns_env:
            env.close()


# ---------------------------------------------------------------------------
# Round-robin
# ---------------------------------------------------------------------------


def run_round_robin(
    checkpoint_paths: list[Path],
    *,
    matches_per_pair: int = 2,
    flip_seats: bool = True,
    loader: Callable[[str], Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    match_kwargs: dict[str, Any] | None = None,
    env_factory: Callable[..., "DoomMultiplayerEnv"] | None = None,
) -> list[MatchResult]:
    """Play every unordered pair in ``checkpoint_paths`` head-to-head.

    When ``flip_seats`` is ``True`` and ``matches_per_pair`` is even, exactly
    half the games have checkpoint A as ``player_0`` and the other half as
    ``player_1``. Results are always reported in canonical (A, B) order -
    i.e. ``frags_a`` is *always* checkpoint A's frag count regardless of
    which seat it played.

    A single env instance is reused across matches (reset between games) so
    we don't pay the ViZDoom init cost per pair.
    """
    if matches_per_pair < 1:
        raise ValueError(
            f"matches_per_pair must be >= 1, got {matches_per_pair}",
        )

    load_fn = loader or _default_loader()
    # Load each checkpoint once; reuse across every pairing.
    policies: dict[str, Any] = {p.stem: load_fn(str(p)) for p in checkpoint_paths}
    names = [p.stem for p in checkpoint_paths]

    # Build one env for the whole sweep so ViZDoom doesn't re-init per match.
    if env_factory is None:
        from rl_doom.multiplayer_env import make_1v1_env as _factory
        env_factory = _factory
    env = env_factory(**(env_kwargs or {}))

    results: list[MatchResult] = []
    base_kwargs: dict[str, Any] = dict(match_kwargs or {})
    base_kwargs.pop("env", None)  # caller can't override - we own the env.

    try:
        for name_a, name_b in itertools.combinations(names, 2):
            model_a = policies[name_a]
            model_b = policies[name_b]
            for match_idx in range(matches_per_pair):
                # Swap seats on odd indices when flip_seats is on; this yields
                # a balanced split for even ``matches_per_pair`` and an
                # A-favoured seat split by one match for odd.
                swap = flip_seats and (match_idx % 2 == 1)
                if swap:
                    res = play_match(
                        model_b, model_a,
                        name_a=name_b, name_b=name_a,
                        env=env,
                        **base_kwargs,
                    )
                    # Normalise back to canonical (A, B) order.
                    res = MatchResult(
                        player_a=name_a,
                        player_b=name_b,
                        frags_a=res.frags_b,
                        frags_b=res.frags_a,
                        steps=res.steps,
                    )
                else:
                    res = play_match(
                        model_a, model_b,
                        name_a=name_a, name_b=name_b,
                        env=env,
                        **base_kwargs,
                    )
                results.append(res)
    finally:
        env.close()

    return results


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for player A vs B."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def compute_elo(
    results: list[MatchResult],
    *,
    initial: float = 1200.0,
    k: float = 32.0,
) -> dict[str, EloRating]:
    """Compute Elo ratings from a list of match results.

    Processes matches in the given order (order matters for Elo because later
    games pick up earlier rating updates). Every unique name appearing as
    ``player_a`` or ``player_b`` gets an initial rating of ``initial``.

    A win scores 1, a loss 0, a draw 0.5. Ties on frag count are draws.
    """
    ratings: dict[str, EloRating] = {}

    def _ensure(name: str) -> EloRating:
        if name not in ratings:
            ratings[name] = EloRating(name=name, rating=initial)
        return ratings[name]

    for res in results:
        a = _ensure(res.player_a)
        b = _ensure(res.player_b)

        if res.frags_a > res.frags_b:
            sa, sb = 1.0, 0.0
            a.wins += 1
            b.losses += 1
        elif res.frags_b > res.frags_a:
            sa, sb = 0.0, 1.0
            a.losses += 1
            b.wins += 1
        else:
            sa = sb = 0.5
            a.draws += 1
            b.draws += 1

        ea = _expected_score(a.rating, b.rating)
        eb = 1.0 - ea
        a.rating += k * (sa - ea)
        b.rating += k * (sb - eb)

    return ratings


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_results_csv(results: list[MatchResult], path: Path) -> None:
    """Serialise match results to a CSV for later analysis.

    Using plain string writes (not :mod:`csv`) because the columns are all
    primitives and the file is tiny; keeps the dependency graph minimal.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["player_a,player_b,frags_a,frags_b,steps"]
    for r in results:
        lines.append(
            f"{r.player_a},{r.player_b},{r.frags_a},{r.frags_b},{r.steps}",
        )
    path.write_text("\n".join(lines) + "\n")


def _write_elo_csv(ratings: dict[str, EloRating], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["name,rating,wins,losses,draws,games"]
    # Sort descending by rating so the CSV is readable in raw form.
    for r in sorted(ratings.values(), key=lambda x: x.rating, reverse=True):
        lines.append(
            f"{r.name},{r.rating:.2f},{r.wins},{r.losses},{r.draws},{r.games}",
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------


@dataclass
class TournamentReport:
    """Aggregate output of :func:`tournament`."""

    checkpoints: list[Path]
    results: list[MatchResult]
    ratings: dict[str, EloRating]
    results_csv: Path | None = None
    elo_csv: Path | None = None
    heatmap_png: Path | None = None

    @property
    def leaderboard(self) -> list[EloRating]:
        """Ratings sorted from strongest to weakest."""
        return sorted(
            self.ratings.values(), key=lambda x: x.rating, reverse=True,
        )

    def score_matrix(self) -> tuple[list[str], np.ndarray]:
        """Pairwise win-rate matrix aligned to the checkpoint list.

        Entry ``[i, j]`` is the fraction of frags player ``i`` scored when
        playing player ``j`` (frags_i / (frags_i + frags_j)), aggregated over
        every match between them. The diagonal is left as ``NaN``.
        """
        names = [p.stem for p in self.checkpoints]
        n = len(names)
        numer = np.zeros((n, n), dtype=np.float64)
        denom = np.zeros((n, n), dtype=np.float64)
        idx_of = {name: i for i, name in enumerate(names)}
        for r in self.results:
            i, j = idx_of[r.player_a], idx_of[r.player_b]
            numer[i, j] += r.frags_a
            numer[j, i] += r.frags_b
            denom[i, j] += r.frags_a + r.frags_b
            denom[j, i] += r.frags_a + r.frags_b
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.where(denom > 0, numer / denom, np.nan)
        # NaN the diagonal for readability.
        for i in range(n):
            mat[i, i] = np.nan
        return names, mat


def _discover_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Return sorted ``.zip`` checkpoints under ``checkpoint_dir``.

    Sort order is lexical, which matches the ``step_{:09d}.zip`` pattern
    emitted by :class:`rl_doom.self_play.SelfPlaySnapshotCallback` and keeps
    the leaderboard rows in training-time order.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {checkpoint_dir}")
    return sorted(checkpoint_dir.glob("*.zip"))


def _render_heatmap(
    names: list[str],
    matrix: np.ndarray,
    path: Path,
) -> None:
    """Render a square heatmap of pairwise frag share.

    Lazy-imports matplotlib so the rest of the module stays usable in
    headless / minimal-deps environments.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(max(4.0, 0.4 * len(names)), max(4.0, 0.4 * len(names))),
    )
    im = ax.imshow(
        matrix,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Frag share (row player vs column player)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def tournament(
    checkpoint_dir: str | Path,
    *,
    matches_per_pair: int = 2,
    flip_seats: bool = True,
    out_dir: str | Path | None = None,
    loader: Callable[[str], Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    match_kwargs: dict[str, Any] | None = None,
    env_factory: Callable[..., "DoomMultiplayerEnv"] | None = None,
    initial_rating: float = 1200.0,
    k_factor: float = 32.0,
    write_heatmap: bool = True,
) -> TournamentReport:
    """Run a round-robin tournament over every ``.zip`` in ``checkpoint_dir``.

    If ``out_dir`` is given, writes ``results.csv`` and ``elo.csv`` (and a
    ``heatmap.png`` when ``write_heatmap`` is true) under it. Returns a
    :class:`TournamentReport` with in-memory copies of everything plus the
    paths of any artifacts written.
    """
    ckpt_dir = Path(checkpoint_dir)
    checkpoints = _discover_checkpoints(ckpt_dir)
    if len(checkpoints) < 2:
        raise ValueError(
            f"Need >= 2 checkpoints in {ckpt_dir} for a tournament; "
            f"found {len(checkpoints)}.",
        )

    results = run_round_robin(
        checkpoints,
        matches_per_pair=matches_per_pair,
        flip_seats=flip_seats,
        loader=loader,
        env_kwargs=env_kwargs,
        match_kwargs=match_kwargs,
        env_factory=env_factory,
    )
    ratings = compute_elo(results, initial=initial_rating, k=k_factor)

    report = TournamentReport(
        checkpoints=checkpoints,
        results=results,
        ratings=ratings,
    )

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results_csv = out_path / "results.csv"
        elo_csv = out_path / "elo.csv"
        _write_results_csv(results, results_csv)
        _write_elo_csv(ratings, elo_csv)
        report.results_csv = results_csv
        report.elo_csv = elo_csv
        if write_heatmap:
            names, matrix = report.score_matrix()
            heatmap = out_path / "heatmap.png"
            try:
                _render_heatmap(names, matrix, heatmap)
                report.heatmap_png = heatmap
            except ImportError:
                # matplotlib not available - skip silently; CSVs still written.
                report.heatmap_png = None

    return report


__all__ = [
    "EloRating",
    "MatchResult",
    "TournamentReport",
    "compute_elo",
    "play_match",
    "run_round_robin",
    "tournament",
]
