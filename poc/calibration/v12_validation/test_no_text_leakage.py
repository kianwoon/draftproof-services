"""Guard: committed v12 JSON artifacts are numbers-only — no corpus text.

The §12 corpus is SCoCESLE-derived (no-redistribution license) and the
generated variants embed those essays. Every committed baseline / lock /
candidate JSON must therefore carry metrics and provenance strings only.
Mechanical rules (owner-mandated 2026-07-08):
  1. no string value longer than MAX_STRING_LEN except under an exempt
     provenance key (or nested anywhere below one);
  2. no key named like document/essay content at any depth.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

MAX_STRING_LEN = 300
EXEMPT_KEYS = {"_notes", "_tuning_provenance", "provenance"}
FORBIDDEN_KEYS = {"text", "document_text", "essay", "content", "source_text"}


def _committed_jsons() -> list[Path]:
    """Every JSON that git would commit from this dir: top-level artifacts
    plus candidates/. The gitignored corpus/ and captures/ dirs are exactly
    what this test exists to keep OUT of the committed files, so they are
    not scanned."""
    paths = sorted(HERE.glob("*.json")) + sorted((HERE / "candidates").glob("*.json"))
    return [p for p in paths if p.is_file()]


def _violations(node, path: str, exempt: bool) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key) in FORBIDDEN_KEYS:
                out.append(f"{path}.{key}: forbidden key name")
            out.extend(_violations(value, f"{path}.{key}", exempt or str(key) in EXEMPT_KEYS))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(_violations(value, f"{path}[{i}]", exempt))
    elif isinstance(node, str) and not exempt and len(node) > MAX_STRING_LEN:
        out.append(f"{path}: string of {len(node)} chars (max {MAX_STRING_LEN} outside {sorted(EXEMPT_KEYS)})")
    return out


@pytest.mark.parametrize("json_path", _committed_jsons(), ids=lambda p: p.name)
def test_committed_json_is_numbers_only(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    problems = _violations(data, json_path.name, exempt=False)
    assert not problems, "possible corpus-text leakage:\n" + "\n".join(problems)


def test_scan_found_artifacts():
    # The scan must never silently pass because it found nothing to check.
    assert _committed_jsons(), "no committed v12 JSON artifacts found — glob broken?"
