"""Residual checker = rewrite pass 2. Re-scans the REWRITTEN draft and re-runs the writer on
paragraphs the fresh re-scan flags; unflagged paragraphs keep their pass-1 text (invariant:
never revert to the original submitted text)."""
import os
from poc.rewrite_v6 import direct_rewrite


def test_kill_switch_default_on_and_off(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_RESIDUAL_FIX", raising=False)
    assert direct_rewrite.residual_fix_enabled() is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", off)
        assert direct_rewrite.residual_fix_enabled() is False
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    assert direct_rewrite.residual_fix_enabled() is True
