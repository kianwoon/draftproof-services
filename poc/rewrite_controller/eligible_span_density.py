"""Eligible prose span density diagnostics for detector-risk labeling.

This module is intentionally independent from the large rewrite pipeline. It
turns scanner output plus text into a simple contract:

* how much eligible prose still looks AI-like,
* whether that risk is concentrated in long contiguous spans, and
* whether the output needs real author/provenance context rather than another
  generic rewrite pass.
"""

from __future__ import annotations

import re
from typing import Any


SPAN_DENSITY_POLICY_VERSION = "eligible_span_density_v1"
MAX_UNSAFE_ELIGIBLE_WORD_RATIO = 35.0
MAX_LONGEST_UNSAFE_SPAN_WORDS = 140
MAX_UNSAFE_CLUSTER_COUNT = 4
MIN_MATERIAL_RATIO_DROP = 3.0
MIN_MATERIAL_LONGEST_SPAN_DROP = 25

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b[\w'-]+\b")
_URL_OR_REFERENCE_RE = re.compile(r"https?://|www\.|^\s*references?\s*$", re.I)
_TRANSITION_RE = re.compile(
    r"^\s*(?:however|therefore|furthermore|moreover|additionally|in addition|in conclusion|overall|"
    r"despite|at the same time|another important|one of the|this means|this shows|this highlights)\b",
    re.I,
)
_GENERIC_RE = re.compile(
    r"\b(?:one of the|important|significant|major role|plays? a|influential|known for|wide range|"
    r"has become|this shows|this reflects|strong influence|global impact|central part|key feature)\b",
    re.I,
)
_CANONICAL_RE = re.compile(
    r"\b(?:\d{4}|constitution|civil war|world war|declared independence|was founded|"
    r"united nations|north atlantic treaty organization|nato|nasa)\b",
    re.I,
)
_AUTHOR_PROVENANCE_RE = re.compile(
    r"\b(?:draft|revision|feedback|outline|notes?|source document|supporting document|"
    r"teacher feedback|class discussion|my example|i noticed|i used|we checked|in class)\b",
    re.I,
)


def _num(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else float(default)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text or "")))


def _split_sentences(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    return [sentence.strip() for sentence in _SENTENCE_RE.split(value) if sentence.strip()]


def _predictability_rows(text: str, report_dict: dict | None) -> list[dict[str, Any]]:
    predictability = (report_dict or {}).get("predictability") if isinstance(report_dict, dict) else {}
    source_rows = []
    if isinstance(predictability, dict):
        source_rows = predictability.get("all_sentences") or predictability.get("sentences") or []
    rows: list[dict[str, Any]] = []
    if isinstance(source_rows, list) and source_rows:
        for index, item in enumerate(source_rows):
            if not isinstance(item, dict):
                continue
            sentence = str(item.get("sentence") or item.get("text") or "").strip()
            if not sentence:
                continue
            rows.append({
                "sentence_index": index,
                "sentence_id": item.get("sentence_id") or f"s{index + 1:03d}",
                "sentence": sentence,
                "top10_ratio": _num(item.get("top10_ratio"), _num(item.get("top_10_ratio"))),
                "top50_ratio": _num(item.get("top50_ratio"), _num(item.get("top_50_ratio"))),
                "predictability_risk": _num(item.get("predictability_risk"), _num(item.get("risk"), _num(item.get("score")))),
            })
    if rows:
        return rows
    return [
        {
            "sentence_index": index,
            "sentence_id": f"s{index + 1:03d}",
            "sentence": sentence,
            "top10_ratio": 0.0,
            "top50_ratio": 0.0,
            "predictability_risk": 0.0,
        }
        for index, sentence in enumerate(_split_sentences(text))
    ]


def _human_anchor_score(report_dict: dict | None) -> float:
    badge = (report_dict or {}).get("ai_risk_badge") if isinstance(report_dict, dict) else {}
    transform = (badge or {}).get("transformation_classification") if isinstance(badge, dict) else {}
    features = (transform or {}).get("features") if isinstance(transform, dict) else {}
    value = features.get("human_anchor_score") if isinstance(features, dict) else None
    if isinstance(value, (int, float)):
        return round(float(value) * 100.0 if abs(float(value)) <= 1 else float(value), 3)
    writing = (badge or {}).get("writing_components") if isinstance(badge, dict) else {}
    lived = _num((writing or {}).get("lived_detail_risk"), 100.0)
    domain = _num((writing or {}).get("domain_grounding_strength"), 0.0)
    return round(0.80 * (100.0 - lived) + 0.20 * domain, 3)


def build_eligible_span_density_contract(text: str, report_dict: dict | None) -> dict[str, Any]:
    """Return detector-risk density diagnostics for eligible prose spans."""

    rows = []
    total_eligible_words = 0
    unsafe_words = 0
    unsafe_indexes: set[int] = set()
    for row in _predictability_rows(text, report_dict):
        sentence = str(row.get("sentence") or "").strip()
        words = _word_count(sentence)
        eligible = bool(words >= 8 and not _URL_OR_REFERENCE_RE.search(sentence))
        canonical = bool(_CANONICAL_RE.search(sentence))
        transition = bool(_TRANSITION_RE.search(sentence))
        generic_hits = len(_GENERIC_RE.findall(sentence))
        top10 = _num(row.get("top10_ratio"))
        top50 = _num(row.get("top50_ratio"))
        predictability = _num(row.get("predictability_risk"))
        risk_score = (
            top10 * 4.0
            + top50 * 1.2
            + predictability * 2.0
            + generic_hits * 0.75
            + (1.35 if transition else 0.0)
            - (1.3 if canonical else 0.0)
        )
        unsafe = bool(
            eligible
            and not canonical
            and (
                risk_score >= 4.1
                or (top10 >= 0.62 and (generic_hits > 0 or transition))
                or (generic_hits >= 2 and top10 >= 0.45)
            )
        )
        if eligible:
            total_eligible_words += words
        if unsafe:
            unsafe_words += words
            unsafe_indexes.add(int(row.get("sentence_index") or 0))
        rows.append({
            "sentence_index": int(row.get("sentence_index") or 0),
            "sentence_id": row.get("sentence_id"),
            "word_count": words,
            "eligible": eligible,
            "unsafe": unsafe,
            "canonical_fact_preserved": canonical,
            "transition_risk": transition,
            "generic_hits": generic_hits,
            "top10_ratio": round(top10, 4),
            "top50_ratio": round(top50, 4),
            "predictability_risk": round(predictability, 4),
            "risk_score": round(risk_score, 3),
            "preview": sentence[:220],
        })

    clusters = []
    active: list[dict[str, Any]] = []
    for row in rows:
        if row["sentence_index"] not in unsafe_indexes:
            if active:
                clusters.append(_cluster_row(active))
                active = []
            continue
        active.append(row)
    if active:
        clusters.append(_cluster_row(active))
    clusters.sort(key=lambda item: (int(item.get("word_count") or 0), float(item.get("risk_score") or 0.0)), reverse=True)
    longest = int(clusters[0]["word_count"]) if clusters else 0
    ratio = round((unsafe_words / max(1, total_eligible_words)) * 100.0, 3)
    ai_components = (((report_dict or {}).get("ai_risk_badge") or {}).get("ai_components") or {}) if isinstance(report_dict, dict) else {}
    external_inputs = {
        "ai_likelihood": _num(((report_dict or {}).get("ai_risk_badge") or {}).get("ai_likelihood_score")),
        "topk_calibrated_risk": _num(ai_components.get("topk_calibrated_risk"), _num(ai_components.get("topk_pattern"))),
        "qualifying_text_ai_density": _num(ai_components.get("qualifying_text_ai_density")),
    }
    human_anchor = _human_anchor_score(report_dict)
    provenance_markers = len(_AUTHOR_PROVENANCE_RE.findall(str(text or "")))
    safe = bool(
        ratio <= MAX_UNSAFE_ELIGIBLE_WORD_RATIO
        and longest <= MAX_LONGEST_UNSAFE_SPAN_WORDS
        and len(clusters) <= MAX_UNSAFE_CLUSTER_COUNT
    )
    needs_author_context = bool(
        not safe
        and human_anchor < 45.0
        and provenance_markers <= 1
    )
    return {
        "version": SPAN_DENSITY_POLICY_VERSION,
        "safe": safe,
        "thresholds": {
            "max_unsafe_eligible_word_ratio": MAX_UNSAFE_ELIGIBLE_WORD_RATIO,
            "max_longest_unsafe_span_words": MAX_LONGEST_UNSAFE_SPAN_WORDS,
            "max_unsafe_cluster_count": MAX_UNSAFE_CLUSTER_COUNT,
        },
        "eligible_sentence_count": sum(1 for row in rows if row.get("eligible")),
        "eligible_word_count": total_eligible_words,
        "unsafe_sentence_count": len(unsafe_indexes),
        "unsafe_word_count": unsafe_words,
        "unsafe_eligible_word_ratio": ratio,
        "longest_unsafe_span_words": longest,
        "unsafe_cluster_count": len(clusters),
        "top_unsafe_clusters": clusters[:6],
        "top_sentence_targets": sorted(
            [row for row in rows if row.get("unsafe")],
            key=lambda item: float(item.get("risk_score") or 0.0),
            reverse=True,
        )[:12],
        "human_anchor_score": human_anchor,
        "provenance_marker_count": provenance_markers,
        "needs_author_context": needs_author_context,
        "ceiling_reason": (
            "unsafe eligible prose remains concentrated and author/provenance context is insufficient"
            if needs_author_context else ""
        ),
        "external_detector_inputs": external_inputs,
        "recommended_actions": _recommended_actions(safe, needs_author_context, clusters),
    }


def _cluster_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "start_sentence": rows[0].get("sentence_index"),
        "end_sentence": rows[-1].get("sentence_index"),
        "sentence_count": len(rows),
        "word_count": sum(int(row.get("word_count") or 0) for row in rows),
        "risk_score": round(sum(float(row.get("risk_score") or 0.0) for row in rows), 3),
        "generic_hits": sum(int(row.get("generic_hits") or 0) for row in rows),
        "transition_count": sum(1 for row in rows if row.get("transition_risk")),
        "preview": " ".join(str(row.get("preview") or "") for row in rows[:3])[:320],
    }


def _recommended_actions(safe: bool, needs_author_context: bool, clusters: list[dict[str, Any]]) -> list[str]:
    if safe:
        return ["preserve_density_safe_output"]
    actions = ["target_longest_unsafe_cluster", "break_contiguous_generic_prose"]
    if clusters and int(clusters[0].get("word_count") or 0) > MAX_LONGEST_UNSAFE_SPAN_WORDS:
        actions.append("split_or_reframe_longest_cluster")
    if needs_author_context:
        actions.append("request_or_use_author_provenance_context")
    else:
        actions.append("bounded_cluster_patch")
    return actions


def compare_eligible_span_density(
    before_text: str,
    before_report: dict | None,
    after_text: str,
    after_report: dict | None,
) -> dict[str, Any]:
    before = build_eligible_span_density_contract(before_text, before_report)
    after = build_eligible_span_density_contract(after_text, after_report)
    ratio_drop = round(float(before.get("unsafe_eligible_word_ratio") or 0.0) - float(after.get("unsafe_eligible_word_ratio") or 0.0), 3)
    longest_drop = int(before.get("longest_unsafe_span_words") or 0) - int(after.get("longest_unsafe_span_words") or 0)
    cluster_drop = int(before.get("unsafe_cluster_count") or 0) - int(after.get("unsafe_cluster_count") or 0)
    improved = bool(
        ratio_drop >= MIN_MATERIAL_RATIO_DROP
        or longest_drop >= MIN_MATERIAL_LONGEST_SPAN_DROP
        or (ratio_drop > 0.0 and cluster_drop > 0)
    )
    regressed = bool(ratio_drop < -1.0 or longest_drop < -20)
    return {
        "version": "eligible_span_density_comparison_v1",
        "before": before,
        "after": after,
        "safe": bool(after.get("safe")),
        "improved": improved,
        "regressed": regressed,
        "unsafe_eligible_word_ratio_drop": ratio_drop,
        "longest_unsafe_span_words_drop": longest_drop,
        "unsafe_cluster_count_drop": cluster_drop,
        "needs_author_context": bool(after.get("needs_author_context")),
        "reason": (
            "eligible_span_density_safe"
            if after.get("safe")
            else "eligible_span_density_improved_but_unsafe"
            if improved
            else "eligible_span_density_unsafe"
        ),
    }
