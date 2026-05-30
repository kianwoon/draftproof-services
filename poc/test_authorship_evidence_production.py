"""run_rewrite_pipeline_v6 must build the evidence block from detect_json and attach it
(with verbatim preserved_ideas) to the rewrite report it returns."""
from poc.report.authorship_evidence import build_authorship_evidence, preserved_idea_spans


def test_evidence_built_from_detect_json_signals():
    detect_json = {"authorship_concern": {"signals": {"genericity": 0.0, "specificity": 0.69,
                   "source_grounding": 1.0}, "confidence": "medium"}, "false_positives": []}
    sig = detect_json["authorship_concern"]["signals"]
    block = build_authorship_evidence(sig, detect_json.get("false_positives"),
                                      detect_json["authorship_concern"].get("confidence", "low"))
    assert {m["signal"] for m in block["present_markers"]} == {"genericity"}
    assert {t["signal"] for t in block["thin_signals"]} == {"specificity", "source_grounding"}


def test_preserved_ideas_capture_unchanged_sentences():
    original = "I worked in one classroom for a decade. Everything is changing fast now."
    final = "I worked in one classroom for a decade. The pace of change is concrete and measurable."
    kept = [p["text"] for p in preserved_idea_spans(original, final)]
    assert "I worked in one classroom for a decade." in kept
