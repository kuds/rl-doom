"""DreamerV3 agent driver wrapping the NM512/dreamerv3-torch port.

The upstream port has no ``setup.py`` / ``pyproject.toml`` and so cannot be
installed via pip. The notebook ``git clone``s it and adds the checkout to
``sys.path``; this module then imports the four port files we actually need
(``models``, ``networks``, ``tools``, ``exploration``) lazily via
:func:`_import_port` and re-implements the thin ``Dreamer`` ``nn.Module``
driver inline so we avoid pulling in upstream's ``envs.wrappers`` (which
imports the legacy ``gym`` package) and ``ruamel.yaml`` deps.

See ``DREAMER_PLAN.md`` for the integration plan.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

# Single source of truth for the upstream commit pin. The notebook reads this
# constant when ``git clone``-ing so the cloned tree and the imported code can
# never disagree.
UPSTREAM_PIN: str = "6ef8646d807cd10ce0c88e10a7e943211e7fc44c"

# Names of the four upstream modules we actually consume. Kept as a tuple so
# :func:`_import_port` can validate the checkout before mutating ``sys.path``.
_REQUIRED_PORT_FILES: tuple[str, ...] = (
    "models.py",
    "networks.py",
    "tools.py",
    "exploration.py",
    "configs.yaml",
)


def _import_port(port_path: str | Path) -> dict[str, ModuleType]:
    """Import the four upstream modules we need, returning them in a dict.

    ``port_path`` is the directory containing the cloned ``dreamerv3-torch``
    repo (it should contain ``models.py``, ``tools.py``, etc.). The path is
    inserted at the front of ``sys.path`` so the upstream's intra-package
    imports (``import models`` etc.) resolve to the clone, not to anything
    else that happens to share those names.

    Raises ``RuntimeError`` with a friendly message — including the pinned
    commit SHA — when the checkout is missing or incomplete.
    """
    port_path = Path(port_path).expanduser().resolve()
    missing = [
        name for name in _REQUIRED_PORT_FILES if not (port_path / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"DreamerV3 port not found at {port_path!s} "
            f"(missing: {', '.join(missing)}). Clone the upstream repo first:\n"
            f"  git clone https://github.com/NM512/dreamerv3-torch {port_path}\n"
            f"  git -C {port_path} checkout {UPSTREAM_PIN}\n"
            f"and then pass its path as ``port_path``.",
        )
    if str(port_path) not in sys.path:
        sys.path.insert(0, str(port_path))
    import importlib

    return {name: importlib.import_module(name) for name in (
        "models", "networks", "tools", "exploration",
    )}


def _load_port_configs(port_path: str | Path) -> dict[str, Any]:
    """Read the upstream's ``configs.yaml`` via PyYAML (no ruamel dep).

    The file ships verbatim with the port and contains ``defaults`` plus the
    suite-specific presets (``atari100k``, ``crafter``, etc.). We load it with
    :func:`yaml.safe_load` rather than pulling in ``ruamel.yaml`` like the
    upstream CLI does — the file is plain YAML and round-trips fine.
    """
    import yaml

    text = (Path(port_path).expanduser() / "configs.yaml").read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected mapping at top-level of configs.yaml, got {type(data).__name__}",
        )
    return data


def _recursive_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    """In-place ``base <- update`` deep merge, matching upstream's helper."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value


def _to_namespace(d: dict[str, Any]) -> SimpleNamespace:
    """Convert the merged config dict to ``SimpleNamespace`` (attr access)."""
    return SimpleNamespace(**d)
