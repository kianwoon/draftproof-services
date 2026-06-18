"""Tests for the additive two-score policy risk composer.

Pure function over already-computed signals; we feed synthetic grounding_diagnosis
buckets + critical_thinking dimensions and assert the renormalised weighted math.
"""
from detect.policy_risk import (
    MODEL_VERSION,
    WEIGHTS_ALLOWED,
    WEIGHTS_RESTRICTED,
    _band,
    ALLOWED_BANDS,
    RESTRICTED_BANDS,
    score_policy_risk,
)


def _inputs(surface=None, grounding=None, voice=None, judgment=None, specificity=None):
    """Build (grounding_diagnosis, critical_thinking_control) carrying the 5 base signals."""
    gd = {"buckets": {}}
    if surface is not None:
        gd["buckets"]["llm_patterning"] = {"score": surface}
    if grounding is not None:
        gd["buckets"]["concrete_grounding"] = {"score": grounding}
    if voice is not None:
        gd["buckets"]["authorship_trace"] = {"score": voice}
    ctc = {"dimensions": {}}
    if judgment is not None:
        ctc["dimensions"]["student_judgement"] = {"gap": judgment}
    if specificity is not None:
        ctc["dimensions"]["specific_context"] = {"gap": specificity}
    return gd, ctc


def _run(**kw):
    gd, ctc = _inputs(**kw)
    return score_policy_risk(grounding_diagnosis=gd, critical_thinking_control=ctc)


# ── weights ─────────────────────────────────────────────────────────────────

def test_text_derivable_weights_sum_after_omission():
    # AI-Allowed drops declaration_gap (.15) -> remaining .85; AI-Restricted drops
    # process_defensibility_gap (.10) -> remaining .90.
    allowed_text = sum(w for k, w in WEIGHTS_ALLOWED.items() if k != "declaration_gap")
    restricted_text = sum(w for k, w in WEIGHTS_RESTRICTED.items() if k != "process_defensibility_gap")
    assert round(allowed_text, 4) == 0.85
    assert round(restricted_text, 4) == 0.90


# ── worked example (spec values, with renormalisation) ───────────────────────

def test_spec_worked_example_renormalised():
    # spec signals: surface 70, grounding 65, voice 50, judgment 60, specificity 55
    r = _run(surface=70, grounding=65, voice=50, judgment=60, specificity=55)
    # AI-Allowed = (.30*65 + .25*60 + .20*55 + .10*70) / .85 = 52.5 / .85 = 61.76
    assert abs(r["ai_allowed"]["score"] - 61.76) < 0.05
    assert r["ai_allowed"]["level"] == "high"          # 51-75
    # AI-Restricted = (.30*70 + .25*50 + .20*65 + .15*55) / .90 = 54.75 / .90 = 60.83
    assert abs(r["ai_restricted"]["score"] - 60.83) < 0.05
    assert r["ai_restricted"]["level"] == "high"        # 46-70


def test_main_issue_is_top_weighted_contributor():
    # grounding dominates AI-Allowed (.30 weight, high value); surface dominates AI-Restricted.
    r = _run(surface=80, grounding=90, voice=20, judgment=20, specificity=20)
    assert r["ai_allowed"]["main_issue"] == "grounding_gap"
    assert r["ai_restricted"]["main_issue"] == "surface_ai_text_signal"


# ── confirm nudge ─────────────────────────────────────────────────────────────

def test_confirm_delta_math():
    r = _run(surface=60, grounding=60, voice=60, judgment=60, specificity=60)
    # all signals 60 -> baseline 60 for both. Allowed confirm delta = 60 * .15/1.0 = 9.0;
    # restricted = 60 * .10/1.0 = 6.0.
    assert r["ai_allowed"]["score"] == 60.0
    assert abs(r["ai_allowed"]["confirm_delta"] - 9.0) < 0.05
    assert r["ai_allowed"]["confirm_factor"] == "declaration_gap"
    assert abs(r["ai_restricted"]["confirm_delta"] - 6.0) < 0.05
    assert r["ai_restricted"]["confirm_factor"] == "process_defensibility_gap"


# ── bands ────────────────────────────────────────────────────────────────────

def test_allowed_bands():
    assert _band(25, ALLOWED_BANDS) == "low"
    assert _band(25.1, ALLOWED_BANDS) == "moderate"
    assert _band(50, ALLOWED_BANDS) == "moderate"
    assert _band(75, ALLOWED_BANDS) == "high"
    assert _band(75.1, ALLOWED_BANDS) == "severe"


def test_restricted_bands_are_stricter():
    assert _band(20, RESTRICTED_BANDS) == "low"
    assert _band(20.1, RESTRICTED_BANDS) == "moderate"
    assert _band(45, RESTRICTED_BANDS) == "moderate"
    assert _band(70, RESTRICTED_BANDS) == "high"
    assert _band(70.1, RESTRICTED_BANDS) == "severe"


# ── missing signals + abstain ─────────────────────────────────────────────────

def test_missing_signal_is_dropped_and_renormalised():
    # judgment missing -> AI-Allowed renormalises over grounding/specificity/surface only.
    r = _run(surface=40, grounding=40, voice=40, specificity=40)  # no judgment
    # avail weights = .30+.20+.10 = .60; all values 40 -> score 40.
    assert r["ai_allowed"]["score"] == 40.0
    assert r["base_signals"]["judgment_gap"] is None


def test_no_data_is_unknown():
    r = score_policy_risk(grounding_diagnosis={}, critical_thinking_control={})
    assert r["ai_allowed"]["level"] == "unknown"
    assert r["ai_allowed"]["score"] is None
    assert r["ai_restricted"]["level"] == "unknown"


def test_shape_and_model():
    r = _run(surface=30, grounding=30, voice=30, judgment=30, specificity=30)
    assert r["model"] == MODEL_VERSION
    assert set(r["base_signals"]) == {
        "surface_ai_text_signal", "grounding_gap", "authorship_voice_gap",
        "judgment_gap", "prompt_specificity_gap",
    }
    assert "ai_allowed" in r and "ai_restricted" in r
