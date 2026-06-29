import os
from app._composers.authenticity_dashboard import maybe_attach


def test_backfill_uses_predictability_for_ci(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_AUTHENTICITY_DASHBOARD", "1")
    badge = {"ai_likelihood_score": 32.0, "tier": "AMBER", "confidence": "low",
             "critical_thinking_control": {"score": 90.0}}
    pred = {"all_sentences": [{"predictability_risk": r} for r in (0.1, 0.5, 0.9)]}
    out = maybe_attach(badge, predictability=pred)
    assert out["ai_assistance"]["ci"]["tentative"] is True
