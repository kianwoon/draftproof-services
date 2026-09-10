"""P1 voice-fidelity check tests (voice_check.voice_fidelity)."""

from .voice_check import voice_fidelity
from .voice_profile import voice_options_var
from . import production

NORMAL = (
    "The migration project required careful coordination between three teams over several months. "
    "We underestimated how much legacy documentation was missing, which delayed the first milestone. "
    "Our team eventually rebuilt the dependency map by interviewing the original maintainers directly. "
    "This experience taught me that institutional knowledge often lives in people, not repositories. "
    "Future projects should budget explicit time for that discovery work before committing to dates."
)

SHORT = "This is short. Only three sentences here. Not enough voice data yet."


def test_fidelity_full_on_identical_text():
    out = voice_fidelity(NORMAL, NORMAL)
    assert out["abstain"] is False
    assert out["score"] == 1.0
    assert all(a["drift"] == 0 for a in out["drift_axes"])
    assert out["source_profile"]["n_sentences"] >= 4


def test_fidelity_below_one_on_register_shifted_rewrite():
    academic = (
        "The implementation of the migration initiative necessitated meticulous interdepartmental "
        "coordination over a protracted temporal horizon. One must acknowledge that the paucity of "
        "legacy documentation substantially impeded the initial project milestone. "
        "Notwithstanding these impediments, the dependency topology was subsequently reconstructed "
        "via structured interviews with the original maintainers. It is thus demonstrable that "
        "institutional knowledge resides predominantly within personnel rather than repositories. "
        "Consequently, prudent project governance mandates explicit temporal allocations for such "
        "discovery endeavours prior to schedule commitment."
    )
    out = voice_fidelity(NORMAL, academic)
    assert out["abstain"] is False
    assert out["score"] < 1.0
    axes = {a["axis"] for a in out["drift_axes"]}
    assert axes == {"formality", "sentence_length_mix", "first_person", "hedging"}


def test_abstain_on_short_text():
    out = voice_fidelity(SHORT, NORMAL)
    assert out["abstain"] is True
    assert out["score"] == 1.0
    assert voice_fidelity(NORMAL, "")["abstain"] is True
    assert voice_fidelity("", "")["abstain"] is True


def test_disabled_voice_options_skips(monkeypatch):
    token = voice_options_var.set({"disabled": True})
    try:
        out = voice_fidelity(NORMAL, NORMAL)
        assert out["abstain"] is True
        assert out["score"] == 1.0
    finally:
        voice_options_var.reset(token)


def test_fail_open_on_internal_error():
    from . import voice_check
    orig = voice_check.build_voice_profile
    voice_check.build_voice_profile = lambda t: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        out = voice_fidelity(NORMAL, NORMAL)
        assert out["abstain"] is True
        assert out["score"] == 1.0
    finally:
        voice_check.build_voice_profile = orig


def test_production_attaches_voice_fidelity_and_profile(monkeypatch):
    """run_rewrite_pipeline_v6 attaches voice_fidelity + voice_profile to the result.

    LLM-free: stub the per-paragraph writer + all post stages so no API key or
    network is needed. Failure here must mean the voice wiring broke, not that
    credentials are missing.
    """
    import types
    from . import production as prod
    from .scan import scan_text

    _scan = scan_text(NORMAL)
    doc = types.SimpleNamespace(
        rewritten_text=NORMAL,
        pass_trace=[],
        initial_scan=_scan,
        final_scan=_scan,
        passes=[],
        final_text_before_quality_repair=None,
        quality_repair=None,
        naturalisation_repair=None,
        predictability_showcase=None,
    )
    monkeypatch.setattr(prod, "run_direct_rewrite_all", lambda *a, **k: doc)
    monkeypatch.setattr(prod, "run_v6_rewrite_all", lambda *a, **k: doc)
    monkeypatch.setattr(prod, "scan_text_with_report", lambda *a, **k: _scan)
    monkeypatch.setattr(prod, "scan_text_preserve_blocks", lambda *a, **k: _scan)
    monkeypatch.setattr(prod, "extract_paragraph_diagnoses", lambda *a, **k: [])
    # Post-writer stages are module-level imports inside production (not prod
    # attributes): disable via their kill switches so no LLM gateway is built.
    monkeypatch.setenv("DRAFTPROOF_V6_HIGHLIGHT_TOPK_REPAIR", "0")
    monkeypatch.setenv("DRAFTPROOF_V6_EXTERNAL_GUARD", "0")
    monkeypatch.setenv("DRAFTPROOF_V6_BRACKET_GROUNDING", "0")
    monkeypatch.setenv("DRAFTPROOF_V6_REGISTER_COACHING", "0")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_VERDICT_REFRAME", "0")
    monkeypatch.setattr(production, "direct_rewrite_enabled", lambda: True)
    detect_json = {
        "input_text": NORMAL,
        "raw_input_text": NORMAL,
        "findings": {},
        "final_risk": 40.0,
    }
    envelope = production.run_rewrite_pipeline_v6(
        detect_json=detect_json,
        output_dir="/tmp/vpv_voice_test_out",
    )
    assert envelope["voice_fidelity"]["abstain"] is False  # identical stub text => scored
    assert envelope["voice_fidelity"]["score"] == 1.0
    assert envelope["voice_profile"] is not None
    assert envelope["voice_profile"]["abstain"] is False
