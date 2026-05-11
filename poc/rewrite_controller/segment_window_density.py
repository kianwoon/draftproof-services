"""Segment-window density controller primitives.

The eligible-span detector risk is often concentrated across several adjacent
sentences.  This module ranks overlapping 5-10 sentence windows and builds
scoped JSON patch prompts so the rewrite pipeline can attack contiguous density
without broad document rewriting.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from detect.turnitin_like import turnitin_like_ai_profile_from_report


SEGMENT_WINDOW_CONTROLLER_VERSION = "segment_window_density_controller_v1"
WINDOW_FAMILIES = (
    "WINDOW_TEXTURE_REBUILD",
    "WINDOW_COMPRESS_REFRAME",
    "WINDOW_DENSITY_BREAK_HYBRID",
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b[\w'-]+\b")
_GENERIC_RE = re.compile(
    r"\b(?:one of the|another important|important feature|plays? a|major role|significant|"
    r"influential|known for|has become|this shows|this reflects|wide range|global impact|"
    r"central part|key feature|important part|many people|in conclusion|overall)\b",
    re.I,
)
_TRANSITION_RE = re.compile(
    r"^\s*(?:however|therefore|furthermore|moreover|additionally|in addition|in conclusion|"
    r"overall|despite|at the same time|another important|one of the|this means|this shows|"
    r"this highlights|on the other hand)\b",
    re.I,
)
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9&.-]{2,}|[A-Z]{2,})(?:\s+[A-Z][a-zA-Z0-9&.-]{2,}){0,5}\b")
_YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_FACTUAL_EVENT_RE = re.compile(
    r"\b(?:was founded|were founded|declared|established|created|formed|signed|"
    r"became|joined|launched|developed|invented|is located|are located|consists of|"
    r"includes|contains|led to|resulted in)\b",
    re.I,
)
_FACTUAL_NOUN_RE = re.compile(
    r"\b(?:constitution|independence|colonies|government|war|treaty|amendment|"
    r"court|congress|parliament|university|company|agency|organization|movement)\b",
    re.I,
)
_ENTITY_STOPWORDS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "But",
    "For",
    "From",
    "In",
    "It",
    "On",
    "Or",
    "So",
    "That",
    "The",
    "This",
    "When",
    "While",
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text or "")))


def split_sentences(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    return [sentence.strip() for sentence in _SENTENCE_RE.split(value) if sentence.strip()]


def is_canonical_fact_sentence(sentence: str) -> bool:
    value = str(sentence or "").strip()
    if not value:
        return False
    if _YEAR_RE.search(value) or _ACRONYM_RE.search(value):
        return True
    entities = []
    for match in _ENTITY_RE.findall(value):
        first_token = str(match).split()[0]
        if first_token not in _ENTITY_STOPWORDS:
            entities.append(match)
    if len(entities) >= 2:
        return True
    if entities and _FACTUAL_EVENT_RE.search(value):
        return True
    return bool(_FACTUAL_EVENT_RE.search(value) and _FACTUAL_NOUN_RE.search(value))


def _predictability_rows(text: str, report_dict: dict | None) -> dict[int, dict[str, Any]]:
    predictability = (report_dict or {}).get("predictability") if isinstance(report_dict, dict) else {}
    source_rows = []
    if isinstance(predictability, dict):
        source_rows = predictability.get("all_sentences") or predictability.get("sentences") or []
    rows: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(source_rows or []):
        if not isinstance(item, dict):
            continue
        sentence = str(item.get("sentence") or item.get("text") or "").strip()
        if not sentence:
            continue
        try:
            sentence_index = int(item.get("sentence_index", index))
        except (TypeError, ValueError):
            sentence_index = index
        rows[sentence_index] = {
            "sentence": sentence,
            "top10_ratio": _num(item.get("top10_ratio"), _num(item.get("top_10_ratio"))),
            "top50_ratio": _num(item.get("top50_ratio"), _num(item.get("top_50_ratio"))),
            "predictability_risk": _num(item.get("predictability_risk"), _num(item.get("risk"), _num(item.get("score")))),
        }
    return rows


def _sentence_classification(sentence: str) -> str:
    if is_canonical_fact_sentence(sentence):
        return "canonical_fact_preserve"
    if _TRANSITION_RE.search(sentence):
        return "template_transition_target"
    if _GENERIC_RE.search(sentence):
        return "generic_expansion_target"
    if _word_count(sentence) >= 24:
        return "low_value_density_target"
    return "preserve"


def sentence_density_rows(text: str, report_dict: dict | None) -> list[dict[str, Any]]:
    sentences = split_sentences(text)
    predictability = _predictability_rows(text, report_dict)
    profile = turnitin_like_ai_profile_from_report(report_dict or {})
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    ai_pressure = _num(components.get("ai_likelihood")) / 100.0
    topk_pressure = _num(components.get("topk_calibrated_risk")) / 100.0
    smooth_pressure = _num(components.get("rewrite_smoothness")) / 100.0
    rows: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        pred = predictability.get(index, {})
        classification = _sentence_classification(sentence)
        generic_hits = len(_GENERIC_RE.findall(sentence))
        transition = bool(_TRANSITION_RE.search(sentence))
        canonical = classification == "canonical_fact_preserve"
        top10 = _num(pred.get("top10_ratio"))
        top50 = _num(pred.get("top50_ratio"))
        predictability_risk = _num(pred.get("predictability_risk"))
        risk_score = (
            top10 * 4.0
            + top50 * 1.2
            + predictability_risk * 2.0
            + generic_hits * 1.15
            + (1.5 if transition else 0.0)
            + ai_pressure * 1.6
            + topk_pressure * 1.2
            + smooth_pressure * 0.8
            - (2.2 if canonical else 0.0)
        )
        editable = bool(
            classification in {
                "generic_expansion_target",
                "template_transition_target",
                "low_value_density_target",
            }
            and not canonical
        )
        rows.append({
            "sentence_index": index,
            "sentence": sentence,
            "word_count": _word_count(sentence),
            "classification": classification,
            "editable": editable,
            "canonical_fact_preserve": canonical,
            "generic_hits": generic_hits,
            "transition_risk": transition,
            "top10_ratio": round(top10, 4),
            "top50_ratio": round(top50, 4),
            "predictability_risk": round(predictability_risk, 4),
            "risk_score": round(max(0.0, risk_score), 3),
        })
    return rows


def build_segment_density_windows(
    text: str,
    report_dict: dict | None,
    *,
    min_size: int = 5,
    max_size: int = 10,
    limit: int = 6,
) -> list[dict[str, Any]]:
    rows = sentence_density_rows(text, report_dict)
    if not rows:
        return []
    raw_windows: list[dict[str, Any]] = []
    min_size = max(2, int(min_size or 5))
    max_size = max(min_size, int(max_size or 10))
    for start in range(len(rows)):
        for size in range(min_size, max_size + 1):
            end = start + size
            if end > len(rows):
                continue
            window_rows = rows[start:end]
            editable_rows = [row for row in window_rows if row.get("editable")]
            if not editable_rows:
                continue
            canonical_count = sum(1 for row in window_rows if row.get("canonical_fact_preserve"))
            transition_count = sum(1 for row in editable_rows if row.get("transition_risk"))
            generic_hits = sum(int(row.get("generic_hits") or 0) for row in editable_rows)
            unsafe_words = sum(int(row.get("word_count") or 0) for row in editable_rows)
            total_words = sum(int(row.get("word_count") or 0) for row in window_rows)
            risk_score = (
                sum(float(row.get("risk_score") or 0.0) for row in editable_rows)
                + transition_count * 1.4
                + generic_hits * 0.45
                + (unsafe_words / max(1, total_words)) * 2.0
                - canonical_count * 1.8
            )
            if risk_score <= 0:
                continue
            raw_windows.append({
                "window_id": f"w{start + 1:03d}_{end:03d}",
                "start_sentence": start,
                "end_sentence": end - 1,
                "sentence_count": size,
                "editable_sentence_count": len(editable_rows),
                "canonical_fact_count": canonical_count,
                "unsafe_word_count": unsafe_words,
                "word_count": total_words,
                "risk_score": round(risk_score, 3),
                "editable_sentences": [
                    {
                        "sentence_index": row["sentence_index"],
                        "classification": row["classification"],
                        "risk_score": row["risk_score"],
                        "top10_ratio": row["top10_ratio"],
                        "generic_hits": row["generic_hits"],
                        "transition_risk": row["transition_risk"],
                        "sentence": row["sentence"],
                    }
                    for row in editable_rows
                ],
                "preview": " ".join(str(row.get("sentence") or "") for row in window_rows)[:420],
            })
    raw_windows.sort(
        key=lambda row: (
            float(row.get("risk_score") or 0.0),
            int(row.get("editable_sentence_count") or 0),
            -int(row.get("canonical_fact_count") or 0),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for row in raw_windows:
        indexes = set(range(int(row["start_sentence"]), int(row["end_sentence"]) + 1))
        if len(indexes & occupied) > max(1, int(row.get("sentence_count") or 1) // 3):
            continue
        selected.append(row)
        occupied.update(indexes)
        if len(selected) >= int(limit or 6):
            break
    return selected


def segment_window_tasks(text: str, report_dict: dict | None, *, limit: int = 3) -> list[dict[str, Any]]:
    windows = build_segment_density_windows(text, report_dict, limit=max(3, int(limit or 3)))
    families = WINDOW_FAMILIES[: max(0, int(limit or 3))]
    tasks: list[dict[str, Any]] = []
    for index, family in enumerate(families):
        if not windows:
            break
        if family == "WINDOW_DENSITY_BREAK_HYBRID" and len(windows) >= 2:
            target_windows = windows[:2]
        else:
            target_windows = [windows[min(index, len(windows) - 1)]]
        editable_indexes = []
        for window in target_windows:
            editable_indexes.extend(
                int(row["sentence_index"])
                for row in (window.get("editable_sentences") or [])
                if isinstance(row, dict) and row.get("sentence_index") is not None
            )
        tasks.append({
            "family": family,
            "task_id": f"{family.lower()}_{index + 1}",
            "windows": target_windows,
            "editable_sentence_indexes": sorted(set(editable_indexes)),
            "targeted_drivers": _targeted_drivers_for_family(family),
        })
    return tasks


def _targeted_drivers_for_family(family: str) -> list[str]:
    if family == "WINDOW_TEXTURE_REBUILD":
        return ["ai_likelihood", "topk_calibrated_risk", "rewrite_smoothness"]
    if family == "WINDOW_COMPRESS_REFRAME":
        return ["semantic_uniformity", "patchwork_expansion", "qualifying_text_ai_density"]
    return ["ai_likelihood", "topk_calibrated_risk", "qualifying_text_ai_density", "rewrite_smoothness"]


def protected_anchor_terms(text: str, *, limit: int = 60) -> list[str]:
    seen: set[str] = set()
    anchors: list[str] = []
    for value in re.findall(r"\b\d{2,4}\b", str(text or "")):
        if value not in seen:
            seen.add(value)
            anchors.append(value)
    for match in _ENTITY_RE.finditer(str(text or "")):
        value = " ".join(match.group(0).split()).strip(" ,.;:")
        if not value or value.lower() in seen or value in {"The", "This", "That", "Many", "Some"}:
            continue
        seen.add(value.lower())
        anchors.append(value)
        if len(anchors) >= int(limit):
            break
    return anchors[:limit]


def segment_window_candidate_prompt(text: str, report_dict: dict | None, task: dict[str, Any]) -> str:
    profile = turnitin_like_ai_profile_from_report(report_dict or {})
    rows_by_index = {row["sentence_index"]: row for row in sentence_density_rows(text, report_dict)}
    sentence_scope = []
    for window in task.get("windows") or []:
        start = int(window.get("start_sentence") or 0)
        end = int(window.get("end_sentence") or start)
        for index in range(start, end + 1):
            row = rows_by_index.get(index)
            if row:
                sentence_scope.append(row)
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
        "DraftProof SEGMENT_WINDOW_DENSITY_CANDIDATE.\n"
        "Objective: reduce Turnitin-like AI score and unsafe eligible prose density in the selected 5-10 sentence window only.\n"
        "Return only valid JSON. Do not rewrite the whole document.\n\n"
        f"Candidate family: {task.get('family')}\n"
        f"Targeted drivers: {json.dumps(task.get('targeted_drivers') or [], ensure_ascii=False)}\n"
        f"Current formula profile: {json.dumps(profile, ensure_ascii=False)[:3600]}\n\n"
        "Hard rules:\n"
        "- Patch only sentence_index values listed as editable below.\n"
        "- Preserve canonical fact sentences exactly unless they are not patched.\n"
        "- Preserve dates, numbers, named entities, citations, and core factual claims.\n"
        "- No personal voice, fake evidence, fake dates, fake people, fake sources, or broad document rewrite.\n"
        "- Prefer removing generic connectors, compressing generic expansion, clause-order variation, and uneven pacing.\n"
        "- Do not polish the window into cleaner essay prose.\n\n"
        f"Protected anchors: {json.dumps(protected_anchor_terms(text), ensure_ascii=False)[:2200]}\n\n"
        f"Window task: {json.dumps(task, ensure_ascii=False)[:5000]}\n\n"
        f"Sentence scope: {json.dumps(sentence_scope, ensure_ascii=False, indent=2)[:9000]}\n\n"
        "Return JSON matching this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def extract_segment_window_payload(raw: str) -> tuple[dict[str, Any] | None, str]:
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
    if not valid:
        return None, "missing_valid_sentence_patches"
    payload["sentence_patches"] = valid
    return payload, ""


def assemble_segment_window_candidate(
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
        if index in touched or index < 0 or index >= len(sentences):
            continue
        if index not in editable:
            continue
        original = sentences[index]
        if is_canonical_fact_sentence(original):
            continue
        replacement = str(patch.get("replacement_text") or "").strip()
        if not replacement or replacement == original:
            continue
        if _word_count(replacement) < 3:
            continue
        replaced = next_text.replace(original, replacement, 1)
        if replaced == next_text:
            continue
        next_text = replaced
        touched.add(index)
        applied.append({
            "sentence_index": index,
            "original_word_count": _word_count(original),
            "replacement_word_count": _word_count(replacement),
            "classification": _sentence_classification(original),
        })
    candidate = next_text.strip()
    if not applied:
        return "", [], "no_applicable_sentence_patches"
    if candidate == source.strip():
        return "", applied, "unchanged_after_sentence_patches"
    return candidate, applied, ""


def segment_patchwork_budget(source_text: str, candidate_text: str, applied: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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
    max_edited = max(1, min(8, int(total * 0.18)))
    accepted = bool(edited <= max_edited and ratio <= 0.18)
    return {
        "version": "segment_window_patchwork_budget_v1",
        "accepted": accepted,
        "edited_sentence_count": edited,
        "edited_sentence_ratio": round(ratio, 3),
        "max_edited_sentences": max_edited,
        "max_edited_sentence_ratio": 0.18,
        "reason": "within_patchwork_budget" if accepted else "patchwork_edit_budget_exceeded",
    }
