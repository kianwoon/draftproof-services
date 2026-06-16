"""ML-free unit tests for the first-person grounding fix.

Covers all three layers without the LLM / scipy stack:
  A. _DIVERSIFIED_SYSTEM is basis-driven (no "often strongest" first-person bias).
  B. select_author_proxy_routes stops auto-injecting first-person on generic verbs.
  C. the selection-time fabrication penalty makes an honest candidate beat a fabricated one.

Full-pipeline behaviour (does the LLM actually emit an honest candidate to pick) is CI/production-only.
"""
from types import SimpleNamespace

from rewrite_v6.author_proxy_routes import select_author_proxy_routes
from rewrite_v6.direct_rewrite import (
    _DIVERSIFIED_SYSTEM,
    _choose_scored_lane,
    _fabrication_penalty,
    _has_added_first_person_experience,
    _lane_selector_trace,
)


# ── A. diversified prompt is basis-driven ──────────────────────────────────────

def test_diversified_system_is_basis_driven_not_first_person_default():
    assert "often strongest" not in _DIVERSIFIED_SYSTEM
    assert "ITS BASIS" in _DIVERSIFIED_SYSTEM
    assert "ATTRIBUTE and qualify" in _DIVERSIFIED_SYSTEM
    # fabrication of experience for second-hand claims is explicitly forbidden
    assert "NEVER invent first-hand" in _DIVERSIFIED_SYSTEM
    # first-person demoted to one mode, not the default
    assert "First-person observation is ONE mode" in _DIVERSIFIED_SYSTEM


# ── B. route selection no longer forces first-person on generic verbs ───────────

_HEADLINE_PARA = (
    "Microsoft reported that AI tools saved roughly 500 million dollars, yet adoption "
    "sits near 3.3 percent as teams review and apply these systems."
)
_FIRST_HAND_PARA = "When I reviewed our rollout last term, I saw teams struggle to apply the tool."


def _modes(routes):
    return [r["mode"] for r in routes]


def test_headline_para_does_not_inject_first_person():
    # action verbs (review/apply) present, but NO first-person pronoun -> no observed_process
    routes = select_author_proxy_routes(_HEADLINE_PARA, diagnosis=None,
                                        finding_tags=["author_anchor_gap", "generic_assertion_risk"])
    modes = _modes(routes)
    assert "observed_process" not in modes
    # the non-first-person alternatives carry the grounding instead
    assert any(m in modes for m in ("source_use", "condition_trigger", "actor_interaction", "decision_moment"))


def test_first_hand_para_still_offers_first_person():
    routes = select_author_proxy_routes(_FIRST_HAND_PARA, diagnosis=None, finding_tags=["author_anchor_gap"])
    assert "observed_process" in _modes(routes)


# ── C. fabrication detector + penalty + selection ──────────────────────────────

def test_added_first_person_detected_against_headline_source():
    src = "Reports say AI tools saved firms roughly 500 million dollars."
    fabricated = "In my own consulting work, I saw a firm capture those 500 million dollars in savings."
    assert _has_added_first_person_experience(fabricated, src) is True


def test_detector_catches_documented_production_fabrications():
    # allow-hardcode: test fixtures (sample inputs asserting the structural regex behaviour), not a
    # production matching list. The detector itself uses only agnostic first-person-anecdote syntax.
    # These are the exact phrasings observed in production that the original regex missed.
    src = "Reports say AI tools saved firms roughly 500 million dollars."
    for fabricated in (
        "a client of mine captured those savings",
        "a colleague of mine confirmed the figure",
        "a firm I helped saw the savings firsthand",
        "a rollout I tracked delivered exactly that",
        "an essay I wrote made the same point",
    ):
        assert _has_added_first_person_experience(fabricated, src) is True, fabricated


def test_analytical_first_person_is_not_flagged():
    # allow-hardcode: test fixtures. The grounding we ENCOURAGE -- the author's own reasoning /
    # attribution -- must NOT trip the penalty, so these assert the detector stays quiet on it.
    src = "Reports say AI tools saved firms roughly 500 million dollars."
    for analytical in (
        "I argue that the figure reflects efficiency, not hype.",
        "Microsoft's data shows the saving; I read this as a productivity signal.",
        "The widely cited 3.3 percent adoption rate suggests caution.",
    ):
        assert _has_added_first_person_experience(analytical, src) is False, analytical


def test_no_penalty_when_source_already_first_person():
    src = "When I taught this unit, I watched students lean on AI."
    candidate = "In my classroom, I saw the same reliance take hold."
    # source is genuinely first-hand -> adding first-person is not fabrication
    assert _has_added_first_person_experience(candidate, src) is False


def test_attribution_candidate_is_not_flagged():
    src = "Reports say AI tools saved firms roughly 500 million dollars."
    attribution = "Microsoft's own figures put the saving near 500 million dollars, though adoption lags."
    assert _has_added_first_person_experience(attribution, src) is False


def test_penalty_default_and_env_tunable(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_FABRICATION_PENALTY", raising=False)
    assert _fabrication_penalty() == 25.0
    monkeypatch.setenv("DRAFTPROOF_V6_FABRICATION_PENALTY", "10")
    assert _fabrication_penalty() == 10.0
    monkeypatch.setenv("DRAFTPROOF_V6_FABRICATION_PENALTY", "0")  # disable
    assert _fabrication_penalty() == 0.0
    monkeypatch.setenv("DRAFTPROOF_V6_FABRICATION_PENALTY", "not-a-number")
    assert _fabrication_penalty() == 25.0


def test_penalty_makes_honest_candidate_win_selection(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_FABRICATION_PENALTY", "25")
    # fabricated candidate has the LOWER raw risk (fabrication lowers the score) ...
    raw_fabricated, raw_honest = 20.0, 30.0
    penalty = _fabrication_penalty()
    fabricated = (raw_fabricated + penalty, 0, "diversified", SimpleNamespace(rewritten_text="In my work, I saw it."))
    honest = (raw_honest, 1, "control", SimpleNamespace(rewritten_text="Reports widely cite the figure."))
    best_score, _attempt, best_lane, _doc = _choose_scored_lane([fabricated, honest])
    # ... but after the +25 penalty (45 vs 30) the honest attribution candidate wins
    assert best_lane == "control"
    assert best_score == raw_honest


def test_lane_selector_trace_records_penalty_flag():
    trace = _lane_selector_trace(45.0, 0, "diversified", selected=False, fabrication_penalty_applied=True)
    assert trace["fabrication_penalty_applied"] is True
    # default stays False for clean candidates
    assert _lane_selector_trace(30.0, 1, "control", selected=True)["fabrication_penalty_applied"] is False
