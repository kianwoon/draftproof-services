"""Grounding-aware residual checker. scan_text (pass 2's re-scan) surfaces only STRUCTURAL tells and
gives a pure-generic paragraph 0 findings (verified experimentally), so a generic leftover (e.g. a
pass-1 source_preserved paragraph) sails through unfixed. Pass 2 therefore also consults the
grounding signals (lived_detail / generic_assertion) directly. Thresholds chosen from measured data:
generic A => lived_gap 0.80 / gen_assert 0.90 (flag); grounded C => lived_gap 0.20 / gen_assert 0.65
(do NOT flag)."""
from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6.pipeline import DocumentResult
from poc.rewrite_v6.scan import scan_text as real_scan_text

# exact texts measured in Experiment 2
GENERIC = ("Education systems change faster than many schools can comfortably manage. Education in "
           "the past was mostly built around the classroom. The system relied on textbooks and "
           "teachers. Knowledge was received by students from trusted sources. The model still "
           "exists. The model no longer fully reflects how young people learn today.")
GROUNDED = ("When I taught ninth grade last spring, eleven of my twenty-eight students could not open "
            "the shared Google Doc because their school laptops still ran the 2019 image. I logged "
            "each failure in a spreadsheet and brought it to our Tuesday department meeting.")


def _clear(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_RESIDUAL_LIVED_GAP", raising=False)
    monkeypatch.delenv("DRAFTPROOF_V6_RESIDUAL_GENERIC_ASSERTION", raising=False)
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")


def _doc(text):
    return DocumentResult(initial_scan=real_scan_text("orig paragraph here."),
                          final_scan=real_scan_text(text), passes=[],
                          rewritten_text=text, pass_trace=[])


def test_grounding_gap_flags_generic_not_grounded(monkeypatch):
    _clear(monkeypatch)
    assert dr._grounding_gap(GENERIC) is True
    assert dr._grounding_gap(GROUNDED) is False


def test_grounding_gap_thresholds_env_tunable(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_LIVED_GAP", "0.99")
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_GENERIC_ASSERTION", "0.99")
    assert dr._grounding_gap(GENERIC) is False   # both signals now below the raised bar


def test_residual_findings_synthesizes_grounding_finding_when_scan_clean(monkeypatch):
    _clear(monkeypatch)
    s = real_scan_text(GENERIC)
    found = dr._residual_findings(s, s.paragraphs[0])
    assert found and "source_grounding" in found[0].tags


def test_residual_findings_empty_for_grounded(monkeypatch):
    _clear(monkeypatch)
    s = real_scan_text(GROUNDED)
    assert dr._residual_findings(s, s.paragraphs[0]) == []


def test_generic_leftover_is_refixed_by_grounding_aware_pass2(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(dr, "_clean_candidate",
                        lambda *a, **k: ("Regrounded with a concrete classroom scene I taught.", []))
    out = dr._apply_residual_fix(_doc(GENERIC), gateway=None, cancellation_check=None)
    # scan_text gives GENERIC 0 structural findings; the grounding signal catches it -> re-fixed
    assert out.rewritten_text == "Regrounded with a concrete classroom scene I taught."


def test_grounded_paragraph_not_refixed(monkeypatch):
    _clear(monkeypatch)
    called = []
    monkeypatch.setattr(dr, "_clean_candidate",
                        lambda *a, **k: (called.append(1), ("X", []))[1])
    out = dr._apply_residual_fix(_doc(GROUNDED), gateway=None, cancellation_check=None)
    assert out.rewritten_text == GROUNDED   # unchanged
    assert called == []                     # writer never invoked
