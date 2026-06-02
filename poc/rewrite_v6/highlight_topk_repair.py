from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class HighlightTopkRepairResult:
    changed: bool
    applied: list[dict[str, Any]]
    skipped: list[dict[str, Any]]

    def to_trace(self) -> dict[str, Any]:
        return {
            "selected_source": "highlight_topk_repair",
            "status": "accepted" if self.changed else "no_change",
            "applied": len(self.applied),
            "skipped": self.skipped[:12],
            "fixes": self.applied[:12],
        }


def highlight_topk_repair_enabled() -> bool:
    return os.environ.get("DRAFTPROOF_V6_HIGHLIGHT_TOPK_REPAIR", "1").strip().lower() not in {"0", "false", "no", "off"}


def apply_highlight_topk_repair(
    text: str,
    highlights: dict[str, Any] | None,
    *,
    gateway: Any,
    cancellation_check: Callable[[], None] | None = None,
) -> tuple[str, HighlightTopkRepairResult]:
    """Repair the exact high-top-k sentences highlighted on /rewrite.

    The pass is intentionally narrow: one sentence in, one sentence out. It accepts only if the
    scanner's local top-k ratio drops and generic grammar/meaning gates pass.
    """
    original_text = str(text or "")
    if not highlight_topk_repair_enabled() or not original_text.strip():
        return original_text, HighlightTopkRepairResult(False, [], [])
    targets = _target_sentences(original_text, highlights, limit=_max_sentences())
    if not targets:
        return original_text, HighlightTopkRepairResult(False, [], [])

    current = original_text
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for target in targets:
        if cancellation_check:
            cancellation_check()
        sentence = target["sentence"]
        if sentence not in current:
            skipped.append({"sentence": sentence[:180], "reason": "sentence_not_found"})
            continue
        options = _request_options(gateway, target)
        accepted = _best_option(sentence, options, gateway)
        if accepted is None:
            skipped.append({"sentence": sentence[:180], "reason": "no_safe_lower_topk_option"})
            continue
        revised, report = accepted
        current = current.replace(sentence, revised, 1)
        applied.append({
            "original": sentence,
            "revised": revised,
            "topk_before": report["topk_before"],
            "topk_after": report["topk_after"],
            "predictable_word_runs": target["predictable_word_runs"],
        })
    return current, HighlightTopkRepairResult(bool(applied), applied, skipped)


def _max_sentences() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_HIGHLIGHT_TOPK_REPAIR_MAX_SENTENCES", "").strip()
    try:
        value = int(raw)
        if value > 0:
            return min(value, 8)
    except (TypeError, ValueError):
        pass
    return 2


def _target_sentences(text: str, highlights: dict[str, Any] | None, *, limit: int) -> list[dict[str, Any]]:
    sentence_spans = _spans(highlights.get("sentences") if isinstance(highlights, dict) else [], len(text))
    word_spans = _spans(highlights.get("words") if isinstance(highlights, dict) else [], len(text))
    if not sentence_spans and word_spans:
        sentence_spans = _sentence_spans_covering_words(text, word_spans)
    rows: list[dict[str, Any]] = []
    for start, end in sentence_spans:
        sentence = text[start:end].strip()
        if len(sentence.split()) < 6:
            continue
        runs = [
            text[max(ws, start):min(we, end)].strip()
            for ws, we in word_spans
            if ws < end and we > start and text[max(ws, start):min(we, end)].strip()
        ]
        rows.append({
            "start": start,
            "end": end,
            "sentence": sentence,
            "predictable_word_runs": _dedupe(runs)[:12],
            "topk": _sentence_topk(sentence),
        })
    rows.sort(key=lambda row: (row["topk"], len(row["predictable_word_runs"]), len(row["sentence"])), reverse=True)
    return rows[:max(1, int(limit or 1))]


def _sentence_spans_covering_words(text: str, word_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    try:
        from poc.detect.document_structure import structured_sentence_segments
    except ImportError:
        from detect.document_structure import structured_sentence_segments

    rows: list[tuple[int, int, int]] = []
    for segment in structured_sentence_segments(text):
        start = segment.get("start_char")
        end = segment.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            continue
        count = sum(1 for ws, we in word_spans if ws < end and we > start)
        if count:
            rows.append((start, end, count))
    rows.sort(key=lambda row: (row[2], row[1] - row[0]), reverse=True)
    return [(start, end) for start, end, _count in rows]


def _spans(value: Any, length: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            start, end = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        if 0 <= start < end <= length:
            rows.append((start, end))
    return rows


_SYSTEM = (
    "Rewrite one exact high-top-k sentence. Keep one sentence, same meaning, facts, numbers, names, "
    "and clean grammar. Do not split into two sentences. Do not add new facts. Return JSON only."
)


def _request_options(gateway: Any, target: dict[str, Any]) -> list[str]:
    from .json_io import parse_json

    payload = {
        "task": "repair_exact_high_topk_sentence",
        "sentence": target["sentence"],
        "predictable_word_runs": target["predictable_word_runs"],
        "rules": [
            "Return 2-4 alternatives for the same single sentence.",
            "Change the clause route around predictable_word_runs; do not just swap one synonym.",
            "Keep all facts, names, numbers, classroom/source details, and stance.",
            "Do not split the sentence or make it choppy.",
            "Do not add typos, grammar errors, slang, or fake human mistakes.",
        ],
        "output_schema": {"alternatives": ["single-sentence rewrite"]},
    }
    response = gateway.chat(
        "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False),
        system=_SYSTEM,
        temperature=0.45,
        top_p=0.9,
        max_tokens=2500,
        response_format={"type": "json_object"},
        app_label="HighlightTopkRepair",
    )
    raw = getattr(response, "raw_content", "") or getattr(response, "content", "") or ""
    data = parse_json(raw)
    values = data.get("alternatives") if isinstance(data, dict) else []
    return _dedupe([str(value).strip() for value in values if str(value or "").strip()])


def _best_option(original: str, options: list[str], gateway: Any) -> tuple[str, dict[str, Any]] | None:
    best: tuple[str, dict[str, Any]] | None = None
    before = _sentence_topk(original)
    for option in options:
        report = _safe_option_report(original, option, before)
        if not report["accepted"]:
            continue
        verification = _verify_safe_replacement(original, option, gateway)
        report["verification"] = verification
        if not verification.get("accepted"):
            report["accepted"] = False
            report["reasons"].append("llm_safety_verifier_rejected")
            continue
        if best is None or report["topk_after"] < best[1]["topk_after"]:
            best = (option, report)
    return best


def _verify_safe_replacement(original: str, candidate: str, gateway: Any) -> dict[str, Any]:
    if not _verifier_enabled():
        return {"accepted": True, "status": "disabled"}
    payload = {
        "task": "verify_single_sentence_replacement",
        "original_sentence": original,
        "replacement_sentence": candidate,
        "checks": [
            "replacement is one complete grammatical sentence",
            "replacement preserves the same meaning, facts, numbers, names, and stance",
            "replacement does not sound awkward, scrambled, or mechanically rearranged",
        ],
        "output_schema": {
            "grammar_safe": True,
            "meaning_safe": True,
            "natural_sentence": True,
            "reason": "short reason if any check fails",
        },
    }
    try:
        response = gateway.chat(
            "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False),
            system="You are a strict copy editor. Return JSON only.",
            temperature=0.0,
            top_p=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
            app_label="HighlightTopkRepairVerifier",
        )
        from .json_io import parse_json
        data = parse_json(getattr(response, "raw_content", "") or getattr(response, "content", "") or "")
    except Exception as exc:
        return {"accepted": False, "status": f"failed:{type(exc).__name__}"}
    if not isinstance(data, dict):
        return {"accepted": False, "status": "invalid_response"}
    accepted = bool(data.get("grammar_safe") is True and data.get("meaning_safe") is True and data.get("natural_sentence") is True)
    return {
        "accepted": accepted,
        "grammar_safe": data.get("grammar_safe"),
        "meaning_safe": data.get("meaning_safe"),
        "natural_sentence": data.get("natural_sentence"),
        "reason": str(data.get("reason") or "")[:240],
    }


def _verifier_enabled() -> bool:
    return os.environ.get("DRAFTPROOF_V6_HIGHLIGHT_TOPK_VERIFY", "1").strip().lower() not in {"0", "false", "no", "off"}


def _safe_option_report(original: str, candidate: str, before: float | None = None) -> dict[str, Any]:
    from .direct_rewrite import _has_broken_grammar
    from .selector_diagnostics import _severe_polarity_inversion
    from .text import Paragraph

    c = str(candidate or "").strip()
    report: dict[str, Any] = {"accepted": False, "reasons": [], "topk_before": before, "topk_after": None}
    if not c or c == original.strip():
        report["reasons"].append("empty_or_unchanged")
        return report
    if len(_split_sentences(c)) != len(_split_sentences(original)):
        report["reasons"].append("sentence_count_changed")
        return report
    ow = len(original.split())
    if not (0.75 * ow <= len(c.split()) <= 1.35 * ow + 3):
        report["reasons"].append("word_count_outside_band")
        return report
    if _numbers(original) != _numbers(c):
        report["reasons"].append("number_changed")
        return report
    coverage = _content_coverage(original, c)
    if coverage < _content_coverage_floor():
        report["reasons"].append("content_coverage_low")
        return report
    if _compound_token_coverage(original, c) < 1.0:
        report["reasons"].append("compound_token_changed")
        return report
    if _has_broken_grammar(c):
        report["reasons"].append("broken_grammar")
        return report
    if _severe_polarity_inversion(c, Paragraph(id="tk", index=0, text=original, sentences=[])):
        report["reasons"].append("polarity_inversion")
        return report
    after = _sentence_topk(c)
    report["topk_before"] = before if isinstance(before, (int, float)) else _sentence_topk(original)
    report["topk_after"] = after
    if after >= report["topk_before"]:
        report["reasons"].append("topk_not_lower")
        return report
    report["accepted"] = True
    return report


def _content_coverage_floor() -> float:
    raw = os.environ.get("DRAFTPROOF_V6_HIGHLIGHT_TOPK_CONTENT_COVERAGE", "").strip()
    try:
        value = float(raw)
        if 0.0 <= value <= 1.0:
            return value
    except (TypeError, ValueError):
        pass
    return 0.72


def _sentence_topk(sentence: str) -> float:
    try:
        from poc.predictability.scanner import PredictabilityScanner
    except ImportError:
        from predictability.scanner import PredictabilityScanner

    scanner = PredictabilityScanner(model_name="gpt2")
    result = scanner.scan_sentence(str(sentence or ""))
    tokens = getattr(result, "token_results", None) or []
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if getattr(token, "top_10", False))
    return round(hits / max(len(tokens), 1), 4)


def _split_sentences(text: str) -> list[str]:
    return [row.strip() for row in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if row.strip()]


def _numbers(text: str) -> Counter:
    return Counter(re.findall(r"\d[\d,\.]*%?", str(text or "")))


def _stem(word: str) -> str:
    value = word.casefold()
    for suffix in ("ing", "ed", "es", "s", "d"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 3:
            return value[: -len(suffix)]
    return value


def _content_key(text: str) -> Counter:
    return Counter(_stem(word) for word in re.findall(r"[A-Za-z]{4,}", str(text or "")))


def _content_coverage(original: str, candidate: str) -> float:
    source = _content_key(original)
    if not source:
        return 1.0
    candidate_key = _content_key(candidate)
    retained = sum(min(count, candidate_key.get(term, 0)) for term, count in source.items())
    return retained / max(sum(source.values()), 1)


_COMPOUND_RE = re.compile(r"[A-Za-z0-9]+(?:[-‑–—][A-Za-z0-9]+)+")


def _compound_token_key(text: str) -> Counter:
    tokens: list[str] = []
    for compound in _COMPOUND_RE.findall(str(text or "")):
        tokens.extend(_stem(part) for part in re.findall(r"[A-Za-z0-9]+", compound))
    return Counter(tokens)


def _compound_token_coverage(original: str, candidate: str) -> float:
    source = _compound_token_key(original)
    if not source:
        return 1.0
    candidate_key = _compound_token_key(candidate) + Counter(
        _stem(word) for word in re.findall(r"[A-Za-z0-9]+", str(candidate or ""))
    )
    retained = sum(min(count, candidate_key.get(token, 0)) for token, count in source.items())
    return retained / max(sum(source.values()), 1)


def _dedupe(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = re.sub(r"\s+", " ", text).casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows
