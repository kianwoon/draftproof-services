"""Guard: rewrite-scan compaction must keep the badge fields the report UI reads.

The rewritten scan's badge feeds the report page's dual-headline AI-likelihood block
(DraftProof vs Turnitin/external). If compaction drops `external_detector_estimate`,
the rewritten side can't show the Turnitin estimate — which users need (a rewrite does
not beat perplexity detectors).
"""

from app.rewrite_scan_compaction import compact_rewrite_scan_summary, SCAN_BADGE_KEYS


def test_compaction_retains_external_detector_estimate():
    scan = {
        "ai_risk_badge": {
            "ai_likelihood_score": 36.34,
            "tier": "AMBER",
            "authorship_rating_label": "Possible AI-Assisted",
            "external_detector_estimate": {"score": 55.0, "band": "high"},
        }
    }
    out = compact_rewrite_scan_summary(scan)
    assert out["ai_risk_badge"]["external_detector_estimate"] == {"score": 55.0, "band": "high"}


def test_external_detector_estimate_is_in_keep_list():
    # The dual headline depends on this; lock it in the contract.
    assert "external_detector_estimate" in SCAN_BADGE_KEYS


def test_compaction_keeps_core_badge_fields():
    scan = {
        "ai_risk_badge": {
            "ai_likelihood_score": 42.0,
            "tier": "AMBER",
            "authorship_rating_label": "Possible AI-Assisted",
            "writing_quality_score": 65.0,
        }
    }
    badge = compact_rewrite_scan_summary(scan)["ai_risk_badge"]
    assert badge["ai_likelihood_score"] == 42.0
    assert badge["authorship_rating_label"] == "Possible AI-Assisted"
    assert badge["tier"] == "AMBER"


def test_compaction_retains_policy_and_submission_risk():
    # Regression: these additive composer outputs were NOT in the keep-list, so
    # compaction stripped them before storage. The page survived via API read-time
    # re-enrichment, but the worker-rendered rewrite PDF fell back to the raw AI-score
    # framing -> page/PDF divergence. The PDF must show the same policy scores as the page.
    scan = {
        "ai_risk_badge": {
            "ai_likelihood_score": 35.0,
            "tier": "acceptable",
            "policy_risk": {
                "ai_allowed": {"score": 63, "level": "high"},
                "ai_restricted": {"score": 57, "level": "high"},
            },
            "submission_risk": {"overall": {"level": "high"}},
            "grounding_diagnosis": {"buckets": {}},
            "critical_thinking_control": {"dimensions": {}},
        }
    }
    badge = compact_rewrite_scan_summary(scan)["ai_risk_badge"]
    assert badge["policy_risk"]["ai_allowed"]["score"] == 63
    assert badge["policy_risk"]["ai_restricted"]["level"] == "high"
    assert badge["submission_risk"]["overall"]["level"] == "high"
    assert "grounding_diagnosis" in badge
    assert "critical_thinking_control" in badge


def test_policy_composer_fields_in_keep_list():
    # Lock the contract: PDF + page both read these off the stored badge.
    for key in ("policy_risk", "submission_risk", "grounding_diagnosis", "critical_thinking_control"):
        assert key in SCAN_BADGE_KEYS

def test_compaction_retains_v7_fused_evidence():
    # Regression (rewrite 9a29e56a, 2026-07-07): the rewrite's before/after scans stored
    # only the bare ai_likelihood_score — the number WAS V7-fused, but tier_authority &
    # friends were stripped, so every surface (page, PDF, stored JSON) rendered the
    # composite-era framing and the rewrite comparison looked "still V6". The fused
    # provenance must survive storage; tier_authority also feeds render_rewrite's
    # deep-scan KPI (_deep_scan_pct reads badge.tier_authority.proportion).
    scan = {
        "ai_risk_badge": {
            "ai_likelihood_score": 6.82,
            "tier": "green",
            "tier_authority": {
                "source": "v7_fused",
                "fused_score": 6.82,
                "composite_score": 4.0,
                "proportion": 0.087,
                "paragraphs": [
                    {"index": 0, "sentence_count": 5, "flagged_count": 1,
                     "proportion": 0.2, "band": "amber"},
                ],
            },
            "tier_authority_status": {"enabled": True, "applied": True},
            "authorship_breakdown": {"paragraph_count": 4},
            "ai_signal_deberta": {"available": True},
            "signal_source": "deberta_authoritative",
        }
    }
    badge = compact_rewrite_scan_summary(scan)["ai_risk_badge"]
    assert badge["tier_authority"]["source"] == "v7_fused"
    assert badge["tier_authority"]["proportion"] == 0.087
    assert badge["tier_authority"]["paragraphs"][0]["band"] == "amber"
    assert badge["tier_authority_status"] == {"enabled": True, "applied": True}
    assert badge["authorship_breakdown"] == {"paragraph_count": 4}
    assert badge["ai_signal_deberta"] == {"available": True}
    assert badge["signal_source"] == "deberta_authoritative"


def test_v7_keys_locked_in_contract():
    for key in ("tier_authority", "tier_authority_status", "authorship_breakdown",
                "ai_signal_deberta", "signal_source"):
        assert key in SCAN_BADGE_KEYS, key
