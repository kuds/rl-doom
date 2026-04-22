"""rl-doom package root.

Exposes ``__version__`` so run artifacts (``config.json``, ``stage_summary.txt``)
can record exactly which version of the codebase produced them. The value is
pulled from installed package metadata so it stays in lock-step with
``pyproject.toml`` without a second source of truth; the fallback covers
editable/source checkouts where the distribution is not installed.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("rl-doom")
except PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = ["__version__"]
