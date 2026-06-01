"""QC reviewer reliability. gpt-oss intermittently (~1 in 5) runs away past max_tokens
(finish_reason=length), cutting the JSON mid-object so json.loads fails and EVERY correction is
lost -> review_document silently returned 0 corrections AND _apply_reviewer wrote no trace entry, so
the failure was invisible (artifact shows qc_reviewer:0) and cross-paragraph monoculture shipped
unfixed. Fix: salvage complete corrections from a truncated response, retry on total failure, and
ALWAYS record a qc_reviewer trace entry."""
import types

from poc.rewrite_v6 import document_reviewer as dr
from poc.rewrite_v6 import direct_rewrite as drw
from poc.rewrite_v6.pipeline import DocumentResult
from poc.rewrite_v6.scan import scan_text as real_scan_text


# ---- _salvage_corrections -------------------------------------------------------------------
def test_salvage_recovers_complete_objects_from_truncated_json():
    raw = (
        '{"corrections": ['
        '{"original": "The system relied on textbooks.", "revised": "In my class I shared three core textbooks."}, '
        '{"original": "The model still exists.", "revised": "That model lingers in my district even now."}, '
        '{"original": "A third sentence", "revised": "incomplete value cut o'  # truncated mid-object
    )
    out = dr._salvage_corrections(raw)
    assert len(out) == 2
    assert out[0]["original"] == "The system relied on textbooks."
    assert out[1]["revised"].startswith("That model lingers")


def test_salvage_handles_braces_inside_string_values():
    raw = '{"corrections": [{"original": "a b c", "revised": "use {curly} braces ok"}]}'
    out = dr._salvage_corrections(raw)
    assert len(out) == 1 and out[0]["revised"] == "use {curly} braces ok"


def test_salvage_returns_empty_without_corrections_key():
    assert dr._salvage_corrections('{"foo": [1,2,3]}') == []
    assert dr._salvage_corrections("") == []


def test_salvage_handles_escaped_quotes_in_values():
    # an escaped quote inside a value must not end the string early (brace/depth stays correct)
    raw = r'{"corrections": [{"original": "a b c", "revised": "she said \"go\" twice"}, {"original": "d", "revised": "cut'
    out = dr._salvage_corrections(raw)
    assert len(out) == 1 and out[0]["revised"] == 'she said "go" twice'


# ---- _request_corrections: retry + salvage -------------------------------------------------
def _resp(text):
    return types.SimpleNamespace(raw_content=text, content=text)


class _Gateway:
    def __init__(self, scripted):
        self._scripted = list(scripted)  # each item: str (raw) or Exception
        self.calls = 0

    def chat(self, *a, **k):
        self.calls += 1
        item = self._scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return _resp(item)


def test_request_corrections_salvages_truncated_response_without_retry():
    truncated = (
        '{"corrections": [{"original": "x", "revised": "one two three"}, '
        '{"original": "y", "revised": "four five six"}, {"original": "z", "revised": "cut o'
    )
    gw = _Gateway([truncated])
    corrs, skipped = dr._request_corrections(gw, "prompt", None)
    assert gw.calls == 1                       # salvage, no retry needed
    assert len(corrs) == 2 and skipped == "salvaged_partial"


def test_request_corrections_retries_on_chat_error_then_succeeds():
    good = '{"corrections": [{"original": "a", "revised": "alpha beta gamma"}]}'
    gw = _Gateway([RuntimeError("boom"), good])
    corrs, skipped = dr._request_corrections(gw, "prompt", None)
    assert gw.calls == 2 and len(corrs) == 1 and skipped is None


def test_request_corrections_reports_failure_after_all_attempts():
    gw = _Gateway(["not json at all", "still {nonsense"])
    corrs, skipped = dr._request_corrections(gw, "prompt", None)
    assert corrs == [] and skipped in {"bad_json", "llm_error"}


def test_request_corrections_accepts_valid_empty_list_without_retry():
    # a clean parse with an empty corrections list means "nothing to fix" -- NOT an error; don't retry
    gw = _Gateway(['{"corrections": []}'])
    corrs, skipped = dr._request_corrections(gw, "prompt", None)
    assert corrs == [] and skipped is None and gw.calls == 1


# ---- review_document applies salvaged corrections ------------------------------------------
def test_review_document_applies_salvaged_corrections(monkeypatch):
    monkeypatch.setattr(dr, "_score", lambda t: 10.0)  # constant -> safety guard always passes
    text = "The system relied on textbooks. The model still exists in my school."
    truncated = (
        '{"corrections": ['
        '{"original": "The system relied on textbooks.", "revised": "In my class I shared three core textbooks."}, '
        '{"original": "Nonexistent sentence here", "revised": "this will not splice anywh'  # truncated
    )
    monkeypatch.setattr(dr, "_request_corrections", lambda *a, **k: (dr._salvage_corrections(truncated), "salvaged_partial"))
    res = dr.review_document(text, gateway=object())
    assert len(res.corrections) == 1                       # the one that spliced
    assert "three core textbooks" in res.text
    assert res.skipped == "salvaged_partial"


# ---- _apply_reviewer ALWAYS traces ---------------------------------------------------------
def _doc(text):
    return DocumentResult(initial_scan=real_scan_text("orig."), final_scan=real_scan_text(text),
                          passes=[], rewritten_text=text, pass_trace=[])


def test_apply_reviewer_traces_even_when_zero_corrections(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_REVIEWER", "1")
    monkeypatch.setattr(dr, "review_document",
                        lambda *a, **k: dr.ReviewResult(text=k.get("text", a[0] if a else ""),
                                                        corrections=[], skipped="llm_error"))
    doc = _doc("A para.\n\nB para.")
    out = drw._apply_reviewer(doc, gateway=None, cancellation_check=None)
    entry = next(e for e in out.pass_trace if e.get("selected_source") == "qc_reviewer")
    assert entry["status"] == "llm_error" and entry["applied"] == 0
    assert out.rewritten_text == "A para.\n\nB para."     # text unchanged on failure


def test_apply_reviewer_traces_accepted_when_corrections(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_REVIEWER", "1")
    fixed = "A para improved.\n\nB para."
    monkeypatch.setattr(dr, "review_document",
                        lambda *a, **k: dr.ReviewResult(text=fixed,
                                                        corrections=[dr.Correction(original="A para.", revised="A para improved.")]))
    out = drw._apply_reviewer(_doc("A para.\n\nB para."), gateway=None, cancellation_check=None)
    entry = next(e for e in out.pass_trace if e.get("selected_source") == "qc_reviewer")
    assert entry["status"] == "accepted" and entry["applied"] == 1
    assert out.rewritten_text == fixed
