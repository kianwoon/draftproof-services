"""
pytest configuration for the poc/ test suite.

Problem: poc/rewrite_v6/__init__.py transitively imports the ML stack
(transformers → scipy) which is broken on this machine (scipy dlopen error).
Most test files only need the lightweight scan.py / plan.py / text.py modules,
NOT the full pipeline with GPU inference.

Fix: pre-register stub package objects for `poc` and `poc.rewrite_v6` in
sys.modules BEFORE any test module is collected.  Python then satisfies
``from poc.rewrite_v6.scan import …`` by loading scan.py directly, never
executing __init__.py.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = POC_DIR.parent

# Make sure the project root and poc/ itself are importable roots.
for p in (str(PROJECT_ROOT), str(POC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _stub_package(name: str, path: Path) -> types.ModuleType:
    """Register a lightweight namespace-package stub, bypassing __init__.py."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]          # type: ignore[assignment]
    mod.__package__ = name
    mod.__spec__ = None                 # type: ignore[assignment]
    sys.modules[name] = mod
    return mod


# Stub poc (no __init__.py exists, but pytest needs it as a package)
_stub_package("poc", POC_DIR)

# Stub poc.rewrite_v6 — prevents __init__.py from running (avoids ML stack)
_stub_package("poc.rewrite_v6", POC_DIR / "rewrite_v6")
