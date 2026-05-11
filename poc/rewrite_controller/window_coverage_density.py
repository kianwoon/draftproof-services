"""Sliding-window coverage optimizer primitives.

Turnitin-like AI detectors can aggregate sentence predictions from overlapping
5-10 sentence windows.  This module maps unsafe window coverage back to the
sentences that contribute to many risky windows, then builds scoped JSON patch
tasks for those high-leverage sentences.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from detect.turnitin_like import turnitin_like_ai_profile_from_report

from .eligible_span_density import build_eligible_span_density_contract
from .segment_window_density import (
    is_canonical_fact_sentence,
    protected_anchor_terms,
    sentence_density_rows,
    split_sentences,
)


WINDOW_COVERAGE_CONTROLLER_VERSION = "window_coverage_density_optimizer_v1"
WINDOW_COVERAGE_FAMILIES = (
    "COVERAGE_GENERIC_COMPRESS",
    "COVERAGE_TEMPLATE_BREAK",
    "COVERAGE_CADENCE_SHIFT",
    "COVERAGE_WINDOW_BRIDGE_REORDER",
    "COVERAGE_MULTI_SENTENCE_HYBRID",
)

_WORD_RE = re.compile(r"\b[\w'-]+\b")
_GENERIC_RE = re.compile(
    r"\b(?:one of the|another important|important feature|plays? a|major role|significant|"
    r"influential|known for|has become|this shows|this reflects|wide range|global impact|"
    r"central part|key feature|important part|many people|in conclusion|overall|"
    r"strong influence|major economy|cultural impact|important in society)\b",
    re.I,
)
_TRANSITION_RE = re.compile(
    r"^\s*(?:however|therefore|furthermore|moreover|additionally|in addition|in conclusion|"
    r"overall|despite|at the same time|another important|one of the|this means|this shows|"
    r"this highlights|on the other hand)\b",
    re.I,
)
_ANCHOR_STOPWORDS = {
    "Another",
    "Although",
    "Furthermore",
    "However",
    "Many",
    "Moreover",
    "One",
    "Overall",
    "Some",
    "Technology",
    "The",
    "This",
    "Therefore",
}
_SINGLE_WORD_ANCHOR_ALLOW = {
    "Apple",
    "Britain",
    "Constitution",
    "Google",
    "Hollywood",
    "Microsoft",
    "NASA",
    "Tesla",
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text or "")))


def _targeted_drivers_for_family(family: str) -> list[str]:
    if family == "COVERAGE_GENERIC_COMPRESS":
        return ["qualifying_text_ai_density", "semantic_uniformity", "rewrite_smoothness"]
    if family == "COVERAGE_TEMPLATE_BREAK":
        return ["rewrite_smoothness", "semantic_uniformity", "signal_agreement"]
    if family == "COVERAGE_CADENCE_SHIFT":
        return ["ai_likelihood", "topk_calibrated_risk", "rewrite_smoothness"]
    if family == "COVERAGE_WINDOW_BRIDGE_REORDER":
        return ["semantic_uniformity", "patchwork_expansion", "qualifying_text_ai_density"]
    return [
        "ai_likelihood",
        "topk_calibrated_risk",
        "qualifying_text_ai_density",
        "rewrite_smoothness",
    ]


def _row_generic_pressure(row: dict[str, Any]) -> float:
    sentence = str(row.get("sentence") or "")
    return (
        len(_GENERIC_RE.findall(sentence)) * 1.15
        + (1.5 if _TRANSITION_RE.search(sentence) else 0.0)
        + (0.75 if _word_count(sentence) >= 22 else 0.0)
    )


def _patch_anchor_terms(text: str, *, limit: int = 12) -> list[str]:
    anchors = []
    for term in protected_anchor_terms(text, limit=limit * 2):
        value = str(term or "").strip()
        if not value or value in _ANCHOR_STOPWORDS:
            continue
        first = value.split()[0] if value.split() else ""
        if first in _ANCHOR_STOPWORDS:
            continue
        if (
            len(value.split()) == 1
            and value not in _SINGLE_WORD_ANCHOR_ALLOW
            and not re.search(r"\d", value)
            and not value.isupper()
        ):
            continue
        anchors.append(value)
        if len(anchors) >= limit:
            break
    return anchors


def _window_risk(window_rows: list[dict[str, Any]], components: dict[str, Any]) -> dict[str, Any]:
    editable_rows = [
        row for row in window_rows
        if row.get("editable") and not row.get("canonical_fact_preserve")
    ]
    noncanonical_rows = [
        row for row in window_rows
        if not row.get("canonical_fact_preserve")
    ]
    total_words = sum(int(row.get("word_count") or 0) for row in window_rows)
    unsafe_words = sum(int(row.get("word_count") or 0) for row in editable_rows)
    generic_hits = sum(int(row.get("generic_hits") or 0) for row in editable_rows)
    transition_count = sum(1 for row in editable_rows if row.get("transition_risk"))
    canonical_count = sum(1 for row in window_rows if row.get("canonical_fact_preserve"))
    ai_pressure = _num(components.get("ai_likelihood")) / 100.0
    topk_pressure = _num(components.get("topk_calibrated_risk")) / 100.0
    smooth_pressure = _num(components.get("rewrite_smoothness")) / 100.0
    semantic_pressure = _num(components.get("semantic_uniformity")) / 100.0
    row_risk = sum(float(row.get("risk_score") or 0.0) for row in editable_rows)
    all_row_risk = sum(float(row.get("risk_score") or 0.0) for row in noncanonical_rows)
    unsafe_ratio = (unsafe_words / max(1, total_words)) * 100.0
    risk_score = (
        row_risk * 0.72
        + all_row_risk * 0.18
        + unsafe_ratio * 0.11
        + generic_hits * 1.0
        + transition_count * 1.4
        + ai_pressure * 5.0
        + topk_pressure * 4.0
        + smooth_pressure * 2.0
        + semantic_pressure * 1.5
        - canonical_count * 1.35
    )
    unsafe = bool(
        editable_rows
        and (
            risk_score >= 18.0
            or unsafe_ratio >= 38.0
            or (generic_hits + transition_count >= 2 and risk_score >= 13.0)
        )
    )
    return {
        "risk_score": round(max(0.0, risk_score), 3),
        "unsafe": unsafe,
        "unsafe_word_count": unsafe_words,
        "unsafe_word_ratio": round(unsafe_ratio, 3),
        "generic_hits": generic_hits,
        "transition_count": transition_count,
        "editable_sentence_count": len(editable_rows),
        "canonical_fact_count": canonical_count,
    }


def build_window_coverage_map(
    text: str,
    report_dict: dict | None,
    *,
    min_size: int = 5,
    max_size: int = 10,
    window_limit: int = 24,
    sentence_limit: int = 16,
) -> dict[str, Any]:
    """Return unsafe sliding-window coverage diagnostics."""

    rows = sentence_density_rows(text, report_dict)
    profile = turnitin_like_ai_profile_from_report(report_dict or {})
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    density = build_eligible_span_density_contract(text, report_dict)
    if not rows:
        return {
            "version": WINDOW_COVERAGE_CONTROLLER_VERSION,
            "windows": [],
            "top_coverage_sentences": [],
            "unsafe_window_count": 0,
            "ai_sentence_vote_ratio": 0.0,
            "eligible_span_density": density,
            "turnitin_like_score": profile.get("score"),
        }

    min_size = max(2, int(min_size or 5))
    max_size = max(min_size, int(max_size or 10))
    all_windows: list[dict[str, Any]] = []
    coverage: dict[int, dict[str, Any]] = {}
    eligible_indexes = {
        int(row.get("sentence_index") or 0)
        for row in rows
        if int(row.get("word_count") or 0) >= 8
    }
    for row in rows:
        idx = int(row.get("sentence_index") or 0)
        coverage[idx] = {
            "sentence_index": idx,
            "sentence": row.get("sentence"),
            "classification": row.get("classification"),
            "editable": bool(row.get("editable") and not row.get("canonical_fact_preserve")),
            "canonical_fact_preserve": bool(row.get("canonical_fact_preserve")),
            "protected_anchor_terms": _patch_anchor_terms(str(row.get("sentence") or ""), limit=12),
            "word_count": int(row.get("word_count") or 0),
            "risk_score": float(row.get("risk_score") or 0.0),
            "generic_pressure": round(_row_generic_pressure(row), 3),
            "unsafe_window_count": 0,
            "coverage_risk": 0.0,
            "max_window_risk": 0.0,
        }

    for start in range(len(rows)):
        for size in range(min_size, max_size + 1):
            end = start + size
            if end > len(rows):
                continue
            window_rows = rows[start:end]
            risk = _window_risk(window_rows, components)
            window = {
                "window_id": f"cw{start + 1:03d}_{end:03d}",
                "start_sentence": start,
                "end_sentence": end - 1,
                "sentence_count": size,
                **risk,
                "editable_sentence_indexes": [
                    int(row.get("sentence_index") or 0)
                    for row in window_rows
                    if row.get("editable") and not row.get("canonical_fact_preserve")
                ],
                "preview": " ".join(str(row.get("sentence") or "") for row in window_rows)[:420],
            }
            all_windows.append(window)
            if risk.get("unsafe"):
                for row in window_rows:
                    idx = int(row.get("sentence_index") or 0)
                    item = coverage[idx]
                    item["unsafe_window_count"] += 1
                    item["coverage_risk"] += float(risk.get("risk_score") or 0.0)
                    item["max_window_risk"] = max(
                        float(item.get("max_window_risk") or 0.0),
                        float(risk.get("risk_score") or 0.0),
                    )

    unsafe_windows = [row for row in all_windows if row.get("unsafe")]
    unsafe_windows.sort(
        key=lambda row: (
            float(row.get("risk_score") or 0.0),
            int(row.get("editable_sentence_count") or 0),
            int(row.get("unsafe_word_count") or 0),
        ),
        reverse=True,
    )
    top_sentences = []
    for item in coverage.values():
        if not item.get("editable") or int(item.get("unsafe_window_count") or 0) <= 0:
            continue
        leverage = (
            float(item.get("coverage_risk") or 0.0)
            + int(item.get("unsafe_window_count") or 0) * 2.75
            + float(item.get("risk_score") or 0.0)
            + float(item.get("generic_pressure") or 0.0)
        )
        top_sentences.append({
            **item,
            "coverage_risk": round(float(item.get("coverage_risk") or 0.0), 3),
            "max_window_risk": round(float(item.get("max_window_risk") or 0.0), 3),
            "coverage_leverage": round(leverage, 3),
        })
    top_sentences.sort(
        key=lambda row: (
            float(row.get("coverage_leverage") or 0.0),
            int(row.get("unsafe_window_count") or 0),
            float(row.get("risk_score") or 0.0),
        ),
        reverse=True,
    )
    voted_sentence_count = sum(
        1
        for idx in eligible_indexes
        if int((coverage.get(idx) or {}).get("unsafe_window_count") or 0) > 0
    )
    vote_ratio = round((voted_sentence_count / max(1, len(eligible_indexes))) * 100.0, 3)
    return {
        "version": WINDOW_COVERAGE_CONTROLLER_VERSION,
        "turnitin_like_score": profile.get("score"),
        "components": {
            key: components.get(key)
            for key in (
                "ai_likelihood",
                "topk_calibrated_risk",
                "semantic_uniformity",
                "rewrite_smoothness",
                "patchwork_expansion",
                "signal_agreement",
            )
        },
        "eligible_span_density": density,
        "window_policy": {
            "min_size": min_size,
            "max_size": max_size,
            "stride": 1,
        },
        "window_count": len(all_windows),
        "unsafe_window_count": len(unsafe_windows),
        "ai_sentence_vote_ratio": vote_ratio,
        "voted_sentence_count": voted_sentence_count,
        "eligible_sentence_count": len(eligible_indexes),
        "windows": unsafe_windows[: max(0, int(window_limit or 24))],
        "top_coverage_sentences": top_sentences[: max(0, int(sentence_limit or 16))],
    }


def compare_window_coverage_density(
    before_text: str,
    before_report: dict | None,
    after_text: str,
    after_report: dict | None,
) -> dict[str, Any]:
    before = build_window_coverage_map(before_text, before_report)
    after = build_window_coverage_map(after_text, after_report)
    unsafe_window_drop = int(before.get("unsafe_window_count") or 0) - int(after.get("unsafe_window_count") or 0)
    vote_ratio_drop = round(
        _num(before.get("ai_sentence_vote_ratio")) - _num(after.get("ai_sentence_vote_ratio")),
        3,
    )
    return {
        "version": "window_coverage_comparison_v1",
        "before": before,
        "after": after,
        "unsafe_window_count_drop": unsafe_window_drop,
        "ai_sentence_vote_ratio_drop": vote_ratio_drop,
        "improved": bool(unsafe_window_drop > 0 or vote_ratio_drop > 0.001),
        "safe": bool(
            _num(after.get("ai_sentence_vote_ratio")) <= 20.0
            and int(after.get("unsafe_window_count") or 0) == 0
        ),
    }


def window_coverage_tasks(
    text: str,
    report_dict: dict | None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    task_limit = max(0, int(limit or 5))
    coverage_map = build_window_coverage_map(text, report_dict, sentence_limit=20)
    top_sentences = coverage_map.get("top_coverage_sentences") or []
    if not top_sentences or task_limit <= 0:
        return []
    tasks: list[dict[str, Any]] = []

    def add_task(family: str, sentence_indexes: list[int], suffix: str) -> None:
        indexes = sorted({int(index) for index in sentence_indexes})[:4]
        if len(indexes) < 2 or len(tasks) >= task_limit:
            return
        related_windows = [
            window for window in coverage_map.get("windows") or []
            if any(index in set(window.get("editable_sentence_indexes") or []) for index in indexes)
        ][:8]
        tasks.append({
            "family": family,
            "task_id": f"{family.lower()}_{suffix}",
            "editable_sentence_indexes": indexes,
            "targeted_drivers": _targeted_drivers_for_family(family),
            "coverage_sentences": [
                row for row in top_sentences
                if int(row.get("sentence_index") or -1) in indexes
            ],
            "related_windows": related_windows,
            "base_unsafe_window_count": coverage_map.get("unsafe_window_count"),
            "base_ai_sentence_vote_ratio": coverage_map.get("ai_sentence_vote_ratio"),
        })

    anchor_light = [
        int(row.get("sentence_index"))
        for row in top_sentences
        if row.get("sentence_index") is not None and not (row.get("protected_anchor_terms") or [])
    ]
    anchor_rich = [
        int(row.get("sentence_index"))
        for row in top_sentences
        if row.get("sentence_index") is not None and (row.get("protected_anchor_terms") or [])
    ]
    top_indexes = anchor_light + [idx for idx in anchor_rich if idx not in set(anchor_light)]
    add_task("COVERAGE_MULTI_SENTENCE_HYBRID", top_indexes[:4], "1")
    add_task("COVERAGE_GENERIC_COMPRESS", top_indexes[:3], "2")
    add_task("COVERAGE_TEMPLATE_BREAK", top_indexes[1:5], "3")
    add_task("COVERAGE_CADENCE_SHIFT", top_indexes[::2][:4], "4")
    add_task("COVERAGE_WINDOW_BRIDGE_REORDER", top_indexes[2:6], "5")

    cursor = 0
    while len(tasks) < task_limit and len(top_indexes) >= 2:
        family = WINDOW_COVERAGE_FAMILIES[len(tasks) % len(WINDOW_COVERAGE_FAMILIES)]
        add_task(family, top_indexes[cursor : cursor + 4], f"extended_{len(tasks) + 1}")
        cursor = (cursor + 1) % max(1, len(top_indexes) - 1)
    return tasks[:task_limit]


def window_coverage_candidate_prompt(text: str, report_dict: dict | None, task: dict[str, Any]) -> str:
    profile = turnitin_like_ai_profile_from_report(report_dict or {})
    schema = {
        "strategy": task.get("family"),
        "targeted_drivers": task.get("targeted_drivers") or [],
        "fact_inventory_preserved": True,
        "protected_anchors_preserved": True,
        "unsupported_new_facts": False,
        "sentence_patches": [
            {
                "sentence_index": 0,
                "replacement_text": "replacement for this sentence only"
            }
        ],
    }
    return (
        "DraftProof WINDOW_COVERAGE_DENSITY_CANDIDATE.\n"
        "Objective: reduce unsafe sliding-window coverage and Turnitin-like AI score.\n"
        "Return only valid JSON. Do not rewrite the whole document.\n\n"
        f"Candidate family: {task.get('family')}\n"
        f"Targeted drivers: {json.dumps(task.get('targeted_drivers') or [], ensure_ascii=False)}\n"
        f"Current formula profile: {json.dumps(profile, ensure_ascii=False)[:3600]}\n\n"
        "Hard rules:\n"
        "- Patch only sentence_index values listed as editable below.\n"
        "- Patch 2 to 4 sentences total; do not patch the whole paragraph or document.\n"
        "- For every patched sentence, preserve its protected_anchor_terms exactly.\n"
        "- Preserve canonical fact sentences, dates, numbers, named entities, citations, and required claims.\n"
        "- No personal voice, fake evidence, fake dates, fake people, fake sources, or broad rewrite.\n"
        "- Do not polish the prose into cleaner essay language.\n"
        "- Prefer compression, transition breaking, clause-order variation, and uneven cadence.\n\n"
        f"Protected anchors: {json.dumps(protected_anchor_terms(text), ensure_ascii=False)[:2400]}\n\n"
        f"Coverage task: {json.dumps(task, ensure_ascii=False, indent=2)[:9000]}\n\n"
        "Return JSON matching this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def extract_window_coverage_payload(raw: str) -> tuple[dict[str, Any] | None, str]:
    text = str(raw or "").strip()
    if not text:
        return None, "empty_response"
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
    try:
        payload = json.loads(text)
    except Exception as exc:
        return None, f"invalid_json {exc}"
    if not isinstance(payload, dict):
        return None, "json_not_object"
    if payload.get("unsupported_new_facts") is True:
        return None, "unsupported_new_facts_declared"
    for key in ("fact_inventory_preserved", "protected_anchors_preserved"):
        if payload.get(key) is False:
            return None, f"{key}_false"
    patches = payload.get("sentence_patches")
    if not isinstance(patches, list) or not patches:
        return None, "missing_sentence_patches"
    valid = []
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        try:
            sentence_index = int(patch.get("sentence_index"))
        except (TypeError, ValueError):
            continue
        replacement = str(patch.get("replacement_text") or "").strip()
        if replacement:
            valid.append({"sentence_index": sentence_index, "replacement_text": replacement})
    if len(valid) < 2:
        return None, "too_few_valid_sentence_patches"
    payload["sentence_patches"] = valid[:4]
    return payload, ""


def assemble_window_coverage_candidate(
    source_text: str,
    payload: dict[str, Any],
    task: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], str]:
    source = str(source_text or "")
    sentences = split_sentences(source)
    if not sentences:
        return "", [], "empty_source"
    editable = {int(index) for index in (task.get("editable_sentence_indexes") or [])}
    next_text = source
    applied: list[dict[str, Any]] = []
    touched: set[int] = set()
    for patch in payload.get("sentence_patches") or []:
        try:
            index = int(patch.get("sentence_index"))
        except (TypeError, ValueError):
            continue
        if index in touched or index < 0 or index >= len(sentences) or index not in editable:
            continue
        original = sentences[index]
        if is_canonical_fact_sentence(original):
            continue
        replacement = str(patch.get("replacement_text") or "").strip()
        if not replacement or replacement == original or _word_count(replacement) < 3:
            continue
        lost_anchor = ""
        for anchor in _patch_anchor_terms(original, limit=12):
            if str(anchor) not in replacement:
                lost_anchor = str(anchor)
                break
        if lost_anchor:
            return "", applied, f"protected_anchor_lost {lost_anchor}"
        replaced = next_text.replace(original, replacement, 1)
        if replaced == next_text:
            continue
        next_text = replaced
        touched.add(index)
        applied.append({
            "sentence_index": index,
            "original_word_count": _word_count(original),
            "replacement_word_count": _word_count(replacement),
        })
    candidate = next_text.strip()
    if len(applied) < 2:
        return "", applied, "insufficient_applicable_sentence_patches"
    if candidate == source.strip():
        return "", applied, "unchanged_after_sentence_patches"
    return candidate, applied, ""


def window_coverage_patchwork_budget(
    source_text: str,
    candidate_text: str,
    applied: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sentences = split_sentences(source_text)
    if applied:
        edited = len({int(row.get("sentence_index") or 0) for row in applied if isinstance(row, dict)})
    else:
        source_sentences = [re.sub(r"\s+", " ", item).strip() for item in sentences]
        candidate_sentences = [re.sub(r"\s+", " ", item).strip() for item in split_sentences(candidate_text)]
        matcher = SequenceMatcher(None, source_sentences, candidate_sentences, autojunk=False)
        edited = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")
    total = max(1, len(sentences))
    ratio = edited / total
    max_edited = max(2, min(10, int(total * 0.20)))
    max_ratio = 0.30 if total <= 10 else 0.20
    accepted = bool(edited <= max_edited and ratio <= max_ratio)
    return {
        "version": "window_coverage_patchwork_budget_v1",
        "accepted": accepted,
        "edited_sentence_count": edited,
        "edited_sentence_ratio": round(ratio, 3),
        "max_edited_sentences": max_edited,
        "max_edited_sentence_ratio": max_ratio,
        "reason": "within_patchwork_budget" if accepted else "window_coverage_patchwork_budget_exceeded",
    }
