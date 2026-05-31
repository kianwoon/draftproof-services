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
