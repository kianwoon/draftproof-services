"""Dual-headline AI-likelihood rendering (page + PDF share this band table)."""
from report.render import _ai_likelihood_bands


def test_draftproof_tier_maps_to_color():
    out = _ai_likelihood_bands({"ai_likelihood_score": 42.0, "tier": "AMBER"})
    assert out["draftproof"] == {"score": 42, "tier": "AMBER", "color": "#d97706"}


def test_all_draftproof_tiers():
    colors = {"GREEN": "#16a34a", "AMBER": "#d97706", "ORANGE": "#ea580c", "RED": "#dc2626"}
    for tier, color in colors.items():
        out = _ai_likelihood_bands({"ai_likelihood_score": 50, "tier": tier})
        assert out["draftproof"]["color"] == color


def test_external_bands_map_to_label_and_color():
    cases = {
        "low": ("unlikely to be flagged", "#16a34a"),
        "elevated": ("possibly flagged", "#d97706"),
        "high": ("likely to be flagged", "#dc2626"),
    }
    for band, (label, color) in cases.items():
        out = _ai_likelihood_bands({
            "ai_likelihood_score": 42, "tier": "AMBER",
            "external_detector_estimate": {"score": 59.8, "band": band},
        })
        assert out["external"]["label"] == label
        assert out["external"]["color"] == color
        assert out["external"]["score"] == 60  # rounded


def test_missing_external_returns_none():
    out = _ai_likelihood_bands({"ai_likelihood_score": 42, "tier": "AMBER"})
    assert out["external"] is None
    assert out["draftproof"]["score"] == 42


def test_missing_score_returns_none_draftproof():
    assert _ai_likelihood_bands({})["draftproof"] is None
    assert _ai_likelihood_bands(None)["draftproof"] is None
