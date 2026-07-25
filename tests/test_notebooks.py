"""Structural checks on the notebooks.

The notebooks are the primary user-facing entry point — README's Quick Start
says "run the notebooks in order" — but five of six could not run outside
Colab. Their setup cells carried a comment reading "Uncomment the block below
when running on Google Colab" directly above code that was already
uncommented, so locally they did `os.chdir("/content/rl-doom/notebooks")` and
`from google.colab import drive` and raised.

Nothing executes the notebooks in CI (they need GPU-hours), so these are
static checks on the parts that were actually broken.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
assert NOTEBOOKS, f"no notebooks found under {NOTEBOOK_DIR}"

# Paths that only exist inside a Colab runtime.
COLAB_ONLY_PATHS = ("/content/rl-doom",)

# The guarded helpers that are allowed to mention Colab, because checking
# `in_colab()` is the whole point of them.
GUARDED_HELPERS = ("setup_colab", "setup_google_drive")


def _unguarded_colab_imports(source: str) -> list[str]:
    """``google.colab`` imports not protected by a try/except ImportError.

    An AST walk rather than a substring search, so the "disconnect the runtime
    when finished" cells — which do guard themselves with try/except — are
    correctly treated as fine.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = any(
            handler.type is None
            or (isinstance(handler.type, ast.Name)
                and handler.type.id in {"ImportError", "ModuleNotFoundError", "Exception"})
            for handler in node.handlers
        )
        if not catches_import_error:
            continue
        for child in ast.walk(node):
            guarded.add(id(child))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            name = node.module or ""
        elif isinstance(node, ast.Import):
            name = ",".join(alias.name for alias in node.names)
        else:
            continue
        if "google.colab" in name and id(node) not in guarded:
            offenders.append(name)
    return offenders


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text())
    out = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src: Any = cell["source"]
        out.append("".join(src) if isinstance(src, list) else src)
    return out


def _uncommented(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_parse(path: Path) -> None:
    """Every code cell must be syntactically valid Python."""
    for i, source in enumerate(_code_cells(path)):
        # IPython magics/shell escapes are not Python; skip those cells.
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"{path.name} code cell {i} does not parse: {exc}") from exc


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_unguarded_colab_only_code(path: Path) -> None:
    """Colab-only operations must sit behind a runtime check, not a comment."""
    for i, source in enumerate(_code_cells(path)):
        live = _uncommented(source)
        if any(helper in live for helper in GUARDED_HELPERS):
            continue  # the helpers no-op off Colab by construction

        offenders = _unguarded_colab_imports(live)
        assert not offenders, (
            f"{path.name} code cell {i} imports {offenders} without a "
            f"try/except ImportError guard, so the notebook raises locally"
        )
        for needle in COLAB_ONLY_PATHS:
            assert needle not in live, (
                f"{path.name} code cell {i} references the Colab-only path "
                f"{needle!r} outside a guarded helper"
            )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_uncomment_this_instructions(path: Path) -> None:
    """A cell telling the reader to uncomment code that is already live.

    This is the specific trap that made the notebooks look Colab-optional
    while being Colab-only.
    """
    for i, source in enumerate(_code_cells(path)):
        lines = source.splitlines()
        for j, line in enumerate(lines):
            if not line.lstrip().startswith("#"):
                continue
            # An *instruction* to uncomment, not any mention of the word —
            # "nothing to uncomment or edit" is a legitimate reassurance.
            if not re.search(r"\buncomment\s+(the|this|below)\b", line, re.IGNORECASE):
                continue
            following = [
                x for x in lines[j + 1:]
                if x.strip() and not x.lstrip().startswith("#")
            ]
            assert not following, (
                f"{path.name} code cell {i} says {line.strip()!r} but the code "
                f"below it is already uncommented: {following[0].strip()!r}"
            )


def test_setup_helpers_are_a_noop_off_colab() -> None:
    """The guard itself: off Colab these must not touch anything."""
    from rl_doom.utils import in_colab, setup_colab, setup_google_drive

    assert in_colab() is False, "test env should not look like Colab"
    assert setup_colab() is False
    assert setup_google_drive() is None
