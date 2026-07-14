"""Tests for provenance-label correctness in v7_deberta_academic_calibrate.py.

Regression: the "checkpoint" field in the result JSON was a hardcoded string naming
desklib/ai-text-detector-academic-v1.01, even though the script scores via whatever
Modal endpoint DRAFTPROOF_MODAL_ENDPOINT_URL points at. A sweep run against a
fine-tuned staging checkpoint (configured via DRAFTPROOF_MODAL_CHECKPOINT) still
produced a report labeled with the old checkpoint name -- misleading provenance in
a calibration artifact.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calibration.v7_deberta_academic_calibrate as mod  # noqa: E402


def test_checkpoint_label_uses_env_override(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_MODAL_CHECKPOINT", "x/y")
    label = mod._checkpoint_label()
    assert "x/y" in label


def test_checkpoint_label_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_MODAL_CHECKPOINT", raising=False)
    label = mod._checkpoint_label()
    assert "desklib/ai-text-detector-academic-v1.01" in label
