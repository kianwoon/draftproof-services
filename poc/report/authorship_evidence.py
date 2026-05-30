"""Authorship-evidence builder. Re-presents ALREADY-MEASURED authorship_concern signals as an
honest, descriptive inventory -- never a verdict or score.

Direction (C1): signals are risk-oriented. LOW risk = human-authorship marker PRESENT;
HIGH risk = thin signal surfaced as an ACTION. Confirmed against detect.scoring.derive_concern_tier.
Exclusions (C2): predictability / surprisal_risk / topk_pattern_risk are intrinsic LLM token-floor
stats -- not human-authorship evidence and not honestly mitigable -- so they never appear here.
"""
from __future__ import annotations

from typing import Any

try:
    from detect.layer3_scoring import split_sentences
except ImportError:  # package-qualified import path
    from poc.detect.layer3_scoring import split_sentences

SCHEMA_VERSION = "authorship_evidence.v1"
PRESENT_MAX = 0.35   # risk <= this -> genuine human marker
THIN_MIN = 0.55      # risk >= this -> thin signal -> action

# risk-oriented signals only (low = human-present). label, thin-action. (C2 trio omitted.)
SIGNAL_TABLE: dict[str, tuple[str, str]] = {
    "genericity":        ("Original, non-generic phrasing",        "Replace boilerplate phrasing with your own wording"),
    "specificity":       ("Concrete, specific detail",             "Add concrete examples, names, numbers, or scenarios you know"),
    "source_grounding":  ("Claims tied to sources",                "Tie your key claims to a source or citation"),
    "citation_integrity":("Citations support the claims",          "Add or verify citations for your claims"),
    "draft_evolution":   ("Signs of genuine revision",             "Keep or show your drafting history where possible"),
    "structural_reuse":  ("Original structure",                    "Reshape any reused or templated structure into your own"),
    "burstiness":        ("Natural sentence-length variance",      "Let sentence lengths vary as your natural voice would"),
}


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _summary_line(n_present: int, n_thin: int, confidence: str) -> str:
    soft = " (short submission -- read as indicative)" if confidence == "low" else ""
    if n_present and n_thin:
        return f"Your draft shows {n_present} human-authorship marker(s); {n_thin} area(s) would strengthen it.{soft}"
    if n_present:
        return f"Your draft shows {n_present} human-authorship marker(s).{soft}"
    if n_thin:
        return f"{n_thin} area(s) would strengthen your draft's authorship evidence.{soft}"
    return f"Authorship signals are inconclusive for this submission.{soft}"


def build_authorship_evidence(
    signals: dict[str, Any] | None,
    false_positives: list[dict[str, Any]] | None = None,
    confidence: str = "low",
    preserved_ideas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    signals = signals or {}
    present, mixed, thin = [], [], []
    for sig, (label, action) in SIGNAL_TABLE.items():
        raw = signals.get(sig)
        if raw is None:
            continue
        risk = float(raw)
        if risk <= PRESENT_MAX:
            present.append({"signal": sig, "label": label, "risk": round(risk, 3)})
        elif risk >= THIN_MIN:
            thin.append({"signal": sig, "action": action, "risk": round(risk, 3)})
        else:
            mixed.append({"signal": sig, "label": label, "risk": round(risk, 3)})

    human_spans = []
    for fp in (false_positives or []):
        text = (fp.get("sentence") or "").strip()
        if text:
            human_spans.append({
                "sentence_id": fp.get("sentence_id"),
                "text": text,
                "reason": fp.get("reason"),
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "present_markers": present,
        "mixed_markers": mixed,
        "thin_signals": thin,
        "human_recognized_spans": human_spans,
        "preserved_ideas": list(preserved_ideas or []),
        "confidence": confidence or "low",
        "summary_line": _summary_line(len(present), len(thin), confidence or "low"),
    }


def preserved_idea_spans(original_text: str, final_text: str, *, min_words: int = 3) -> list[dict[str, str]]:
    """Sentences from the original kept VERBATIM in the final text (the user's surviving words)."""
    final_norm = {_norm(s) for s in split_sentences(final_text or "")}
    out: list[dict[str, str]] = []
    for sent in split_sentences(original_text or ""):
        clean = (sent or "").strip()
        if len(clean.split()) >= min_words and _norm(clean) in final_norm:
            out.append({"text": clean})
    return out


def paragraph_authorship_targets(evidence: dict[str, Any] | None, paragraph_text: str) -> dict[str, Any]:
    """Per-paragraph protect+target brief for the writer, derived from the evidence block."""
    if not evidence:
        return {}
    para_set = {_norm(s) for s in split_sentences(paragraph_text or "")}
    protected = [
        span["text"] for span in evidence.get("human_recognized_spans", [])
        if (span.get("text") or "").strip() and _norm(span["text"]) in para_set
    ]
    targets = [t["action"] for t in evidence.get("thin_signals", []) if t.get("action")]
    out: dict[str, Any] = {}
    if protected:
        out["protected_spans"] = protected[:6]
    if targets:
        out["grounding_targets"] = targets[:4]
    return out


def authorship_boost_enabled() -> bool:
    import os
    return os.environ.get("DRAFTPROOF_AUTHORSHIP_BOOST", "1").strip().lower() not in {"0", "false", "no", "off"}
