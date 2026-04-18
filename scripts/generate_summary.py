#!/usr/bin/env python3
"""CLI wrapper for :func:`rl_doom.summary.generate_run_summary`.

Usage (from repo root or Colab):
    python scripts/generate_summary.py                       # all runs
    python scripts/generate_summary.py training_jobs/.../<run_id>

The heavy lifting lives in ``rl_doom.summary`` so notebooks can invoke it
in-process (``from rl_doom.summary import generate_run_summary``) without
spawning a subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make ``rl_doom`` importable when invoked directly from a fresh clone
# before ``pip install -e .`` has been run (e.g. first Colab cell).
sys.path.insert(0, str(REPO_ROOT / "src"))

from rl_doom.summary import discover_runs, generate_run_summary  # noqa: E402

TRAINING_JOBS = REPO_ROOT / "training_jobs"


def main() -> None:
    if len(sys.argv) > 1:
        run_dirs = [Path(p).resolve() for p in sys.argv[1:]]
    else:
        run_dirs = discover_runs(TRAINING_JOBS)

    if not run_dirs:
        print("No training runs found. Run a training notebook first.")
        sys.exit(0)

    summaries = []
    for run_dir in run_dirs:
        summary = generate_run_summary(run_dir)
        summaries.append(summary)
        out_path = run_dir / "stage_summary.txt"
        out_path.write_text(summary)
        print(f"Wrote {out_path}")

    combined = "\n\n".join(summaries)
    combined_path = REPO_ROOT / "stage_summary.txt"
    combined_path.write_text(combined + "\n")
    print(f"\nCombined summary written to {combined_path}")
    print()
    print(combined)


if __name__ == "__main__":
    main()
