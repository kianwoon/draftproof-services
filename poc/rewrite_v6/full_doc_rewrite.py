from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable


def full_doc_rewrite_enabled() -> bool:
    value = os.environ.get("DRAFTPROOF_V6_FULL_DOC_REWRITE", "").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return bool(os.environ.get("DRAFTPROOF_V6_FULL_DOC_REWRITE_MODEL"))


@dataclass(frozen=True)
class FullDocRewriteResult:
    text: str
    changed: bool = False
    status: str = "disabled"
    reasons: list[str] = field(default_factory=list)
    score_before: float | None = None
    score_after: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_trace(self) -> dict[str, Any]:
        return {
            "selected_source": "full_doc_rewrite",
            "status": self.status,
            "changed": self.changed,
            "reasons": list(self.reasons),
            "score_before": self.score_before,
            "score_after": self.score_after,
            "metrics": dict(self.metrics),
            # When accepted, the whole document was re-written in one pass, so the
            # per-paragraph entries earlier in pass_trace no longer map to the final
            # text. Flag it so trace consumers don't read stale paragraph provenance.
            "supersedes_paragraph_provenance": bool(self.changed),
        }


_SYSTEM = (
    "You revise a full document with a light copy-editing touch. Preserve the author's meaning, "
    "voice, details, numbers, first-person perspective, and claim strength. Return only the revised "
    "document."
)


def build_prompt(text: str) -> str:
    return (
        "Rewrite the full document for clarity and flow while preserving the exact meaning, voice, "
        "and level of detail.\n\n"
        "Rules:\n"
        "- Do not add new ideas.\n"
        "- Do not remove specific details.\n"
        "- Do not make the document more formal than the original.\n"
        "- Preserve the first-person perspective.\n"
        "- Keep the claim strength the same.\n"
        "- Return only the rewritten document.\n\n"
        f"DOCUMENT:\n{text}"
    )


def apply_full_doc_rewrite(
    text: str,
    *,
    gateway: Any | None,
    cancellation_check: Callable[[], None] | None = None,
) -> tuple[str, FullDocRewriteResult]:
    if not full_doc_rewrite_enabled():
        return text, FullDocRewriteResult(text=text, status="disabled")
    if gateway is None:
        return text, FullDocRewriteResult(text=text, status="no_gateway", reasons=["missing_gateway"])
    if cancellation_check:
        cancellation_check()
    try:
        candidate = _request_candidate(text, gateway)
    except Exception:
        return text, FullDocRewriteResult(text=text, status="error", reasons=["llm_error"])
    accepted, result = _evaluate_candidate(text, candidate)
    if not accepted:
        return text, result
    return result.text, result


def _request_candidate(text: str, gateway: Any) -> str:
    response = gateway.chat(
        build_prompt(text),
        system=_SYSTEM,
        temperature=0.2,
        top_p=0.85,
        max_tokens=_max_output_tokens(text),
        app_label="FullDocRewrite",
    )
    raw = getattr(response, "raw_content", "") or getattr(response, "content", "") or ""
    return _clean_candidate(raw)


def _clean_candidate(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*(rewritten document|revised document|rewrite):\s*", "", text, flags=re.I)
    return text.strip()


def _evaluate_candidate(original: str, candidate: str) -> tuple[bool, FullDocRewriteResult]:
    candidate = _clean_candidate(candidate)
    base_score = _score(original)
    candidate_score = _score(candidate) if candidate else None
    metrics = _metrics(original, candidate)
    reasons: list[str] = []
    if not candidate:
        reasons.append("empty_candidate")
    if candidate.strip() == original.strip():
        reasons.append("no_change")
    if not metrics["word_count_in_band"]:
        reasons.append("word_count_out_of_band")
    if not metrics["numbers_preserved"]:
        reasons.append("numbers_changed")
    elif not metrics["number_multiset_preserved"]:
        # Unique set matched but a duplicated figure was dropped/added
        # (e.g. "5 sites and 5 controls" -> "5 sites and controls").
        reasons.append("number_count_changed")
    if not metrics["paragraph_count_preserved"]:
        reasons.append("paragraph_count_changed")
    if not metrics["ends_complete"]:
        reasons.append("truncated_candidate")
    if candidate_score is None or candidate_score == float("inf"):
        reasons.append("candidate_unscorable")
    elif candidate_score > base_score + _score_regression_tolerance():
        reasons.append("score_regressed")
    if reasons:
        return False, FullDocRewriteResult(
            text=original,
            changed=False,
            status="rejected",
            reasons=reasons,
            score_before=base_score,
            score_after=candidate_score,
            metrics=metrics,
        )
    return True, FullDocRewriteResult(
        text=candidate,
        changed=True,
        status="accepted",
        score_before=base_score,
        score_after=candidate_score,
        metrics=metrics,
    )


def _metrics(original: str, candidate: str) -> dict[str, Any]:
    original_words = len(original.split())
    candidate_words = len(candidate.split())
    lower, upper = _word_count_band(original_words)
    return {
        "original_words": original_words,
        "candidate_words": candidate_words,
        "word_count_in_band": lower <= candidate_words <= upper,
        "numbers_preserved": _unique_numbers(original) == _unique_numbers(candidate),
        "number_multiset_preserved": _numbers(original) == _numbers(candidate),
        "paragraph_count_preserved": _paragraph_count(original) == _paragraph_count(candidate),
        # A max_tokens=None request can silently truncate mid-sentence; if the
        # original ended on terminal punctuation, the candidate must too.
        "ends_complete": _ends_complete(original, candidate),
    }


_SENTENCE_END = tuple(".!?…")
_CLOSERS = "\"')]}”’"


def _ends_complete(original: str, candidate: str) -> bool:
    if not _ends_on_terminal(original):
        return True  # original itself isn't sentence-terminated; don't penalize
    return _ends_on_terminal(candidate)


def _ends_on_terminal(text: str) -> bool:
    stripped = str(text or "").rstrip().rstrip(_CLOSERS).rstrip()
    return bool(stripped) and stripped.endswith(_SENTENCE_END)


def _word_count_band(word_count: int) -> tuple[int, int]:
    spread = _float_env("DRAFTPROOF_V6_FULL_DOC_REWRITE_WORD_BAND", 0.08)
    spread = max(0.0, min(0.25, spread))
    return int(word_count * (1.0 - spread)), int(word_count * (1.0 + spread)) + 1


def _paragraph_count(text: str) -> int:
    return len([p for p in re.split(r"\n\s*\n+", str(text or "")) if p.strip()])


def _numbers(text: str) -> list[str]:
    return sorted(token.replace(" ", "") for token in re.findall(r"\d+(?:[.,]\d+)?\s*%?", str(text or "")))


def _unique_numbers(text: str) -> list[str]:
    return sorted(set(_numbers(text)))


def _score(text: str) -> float:
    try:
        from .direct_rewrite import _document_ai_risk, _rhythm_risk
        return float(_document_ai_risk(text) + 25.0 * _rhythm_risk(text))
    except Exception:
        return float("inf")


def _max_output_tokens(text: str) -> int:
    """Cap output to a generous multiple of the input so a truncated candidate
    is unlikely, while still bounding cost. The full-doc rewrite preserves length
    (~+8% word band), so ~2 tokens/word plus a fixed floor is ample headroom.
    Env override: DRAFTPROOF_V6_FULL_DOC_REWRITE_MAX_TOKENS (0/blank = auto)."""
    override = os.environ.get("DRAFTPROOF_V6_FULL_DOC_REWRITE_MAX_TOKENS", "").strip()
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass
    words = len(str(text or "").split())
    factor = _float_env("DRAFTPROOF_V6_FULL_DOC_REWRITE_TOKEN_FACTOR", 2.0)
    factor = max(1.5, min(4.0, factor))
    return max(1024, int(words * factor) + 256)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _score_regression_tolerance() -> float:
    value = _float_env("DRAFTPROOF_V6_FULL_DOC_REWRITE_SCORE_TOLERANCE", 0.0)
    return max(0.0, min(15.0, value))
