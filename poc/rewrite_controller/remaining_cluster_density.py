"""Remaining unsafe-cluster density controller primitives.

Segment-window repair is useful when one contiguous span is too long.  After
that span is split, the remaining risk can be distributed across several
generic clusters.  This module maps those clusters and builds scoped JSON patch
tasks so the pipeline can compress or rebuild cluster-level prose without
falling back to broad document rewriting.
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


REMAINING_CLUSTER_CONTROLLER_VERSION = "remaining_cluster_density_controller_v1"
CLUSTER_FAMILIES = (
    "CLUSTER_COMPRESS_GENERIC",
    "CLUSTER_REBUILD_ASYMMETRIC",
    "CLUSTER_REMOVE_LOW_VALUE",
    "CLUSTER_HYBRID",
)

_WORD_RE = re.compile(r"\b[\w'-]+\b")
_GENERIC_RE = re.compile(
    r"\b(?:one of the|another important|important feature|plays? a|major role|significant|"
    r"influential|known for|has become|this shows|this reflects|wide range|global impact|"
    r"central part|key feature|important part|many people|in conclusion|overall|"
    r"strong influence|major economy|cultural impact)\b",
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
    "Advocates",
    "Education",
    "Furthermore",
    "However",
    "Innovation",
    "Large",
    "Major",
    "Many",
    "Moreover",
    "Overall",
    "One",
    "Political",
    "Recently",
    "Some",
    "Technology",
    "This",
    "These",
    "Therefore",
    "University",
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
_REFERENCE_RE = re.compile(r"https?://|www\.|^\s*references?\s*$", re.I)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text or "")))


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


def _targeted_drivers_for_family(family: str) -> list[str]:
    if family == "CLUSTER_COMPRESS_GENERIC":
        return ["qualifying_text_ai_density", "semantic_uniformity", "rewrite_smoothness"]
    if family == "CLUSTER_REBUILD_ASYMMETRIC":
        return ["ai_likelihood", "topk_calibrated_risk", "rewrite_smoothness"]
    if family == "CLUSTER_REMOVE_LOW_VALUE":
        return ["qualifying_text_ai_density", "patchwork_expansion", "semantic_uniformity"]
    return [
        "qualifying_text_ai_density",
        "topk_calibrated_risk",
        "ai_likelihood",
        "rewrite_smoothness",
    ]


def _cluster_protected_anchor_terms(text: str, *, limit: int = 24) -> list[str]:
    anchors = []
    for term in protected_anchor_terms(text, limit=limit * 2):
        value = str(term or "").strip()
        if not value or value in _ANCHOR_STOPWORDS:
            continue
        first_token = value.split()[0] if value.split() else ""
        if first_token in _ANCHOR_STOPWORDS:
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


def _cluster_sentence_editable(sentence: str) -> bool:
    value = str(sentence or "").strip()
    if not value or _REFERENCE_RE.search(value):
        return False
    return bool(
        _TRANSITION_RE.search(value)
        or _GENERIC_RE.search(value)
        or _word_count(value) >= 16
    )


def build_remaining_cluster_map(
    text: str,
    report_dict: dict | None,
    *,
    limit: int = 6,
) -> dict[str, Any]:
    """Rank remaining unsafe clusters after segment-window passes."""

    sentences = split_sentences(text)
    density = build_eligible_span_density_contract(text, report_dict)
    sentence_rows = {int(row.get("sentence_index") or 0): row for row in sentence_density_rows(text, report_dict)}
    profile = turnitin_like_ai_profile_from_report(report_dict or {})
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    ai_pressure = _num(components.get("ai_likelihood")) / 100.0
    topk_pressure = _num(components.get("topk_calibrated_risk")) / 100.0
    smooth_pressure = _num(components.get("rewrite_smoothness")) / 100.0
    clusters: list[dict[str, Any]] = []

    for index, cluster in enumerate(density.get("top_unsafe_clusters") or [], start=1):
        if not isinstance(cluster, dict):
            continue
        try:
            start = int(cluster.get("start_sentence"))
            end = int(cluster.get("end_sentence"))
        except (TypeError, ValueError):
            continue
        if start < 0 or end < start or start >= len(sentences):
            continue
        end = min(end, len(sentences) - 1)
        rows = [sentence_rows.get(i) or {"sentence_index": i, "sentence": sentences[i]} for i in range(start, end + 1)]
        canonical_count = sum(1 for row in rows if is_canonical_fact_sentence(str(row.get("sentence") or "")))
        generic_hits = sum(len(_GENERIC_RE.findall(str(row.get("sentence") or ""))) for row in rows)
        transition_count = sum(1 for row in rows if _TRANSITION_RE.search(str(row.get("sentence") or "")))
        editable_rows = [row for row in rows if _cluster_sentence_editable(str(row.get("sentence") or ""))]
        protected = _cluster_protected_anchor_terms(" ".join(sentences[start : end + 1]), limit=24)
        unsafe_words = int(cluster.get("word_count") or sum(_word_count(str(row.get("sentence") or "")) for row in rows))
        cluster_risk = _num(cluster.get("risk_score"))
        pressure_score = (
            unsafe_words * 0.18
            + cluster_risk * 0.52
            + generic_hits * 1.2
            + transition_count * 1.5
            + ai_pressure * 5.0
            + topk_pressure * 5.0
            + smooth_pressure * 2.0
            - canonical_count * 3.0
        )
        if not editable_rows:
            recommended = "preserve"
        elif canonical_count > 0 and len(editable_rows) <= 1:
            recommended = "compress"
        elif transition_count or generic_hits >= 2:
            recommended = "rebuild"
        else:
            recommended = "compress"
        clusters.append({
            "cluster_id": f"rc{index:03d}_{start:03d}_{end:03d}",
            "start_sentence": start,
            "end_sentence": end,
            "sentence_count": end - start + 1,
            "unsafe_word_count": unsafe_words,
            "cluster_risk_score": round(cluster_risk, 3),
            "ranking_score": round(max(0.0, pressure_score), 3),
            "generic_hits": generic_hits,
            "transition_count": transition_count,
            "canonical_fact_count": canonical_count,
            "editable_sentence_indexes": [
                int(row.get("sentence_index") or 0)
                for row in editable_rows
            ],
            "protected_anchor_terms": protected,
            "recommended_strategy": recommended,
            "preview": " ".join(sentences[start : min(end + 1, start + 3)])[:420],
        })

    clusters.sort(
        key=lambda row: (
            float(row.get("ranking_score") or 0.0),
            int(row.get("unsafe_word_count") or 0),
            -int(row.get("canonical_fact_count") or 0),
        ),
        reverse=True,
    )
    return {
        "version": REMAINING_CLUSTER_CONTROLLER_VERSION,
        "eligible_span_density": density,
        "turnitin_like_score": profile.get("score"),
        "topk_calibrated_risk": components.get("topk_calibrated_risk"),
        "ai_likelihood": components.get("ai_likelihood"),
        "clusters": clusters[: max(0, int(limit or 6))],
    }


def remaining_cluster_tasks(
    text: str,
    report_dict: dict | None,
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    task_limit = max(0, int(limit or 4))
    cluster_map = build_remaining_cluster_map(text, report_dict, limit=max(4, task_limit))
    clusters = [
        row
        for row in cluster_map.get("clusters") or []
        if isinstance(row, dict) and row.get("editable_sentence_indexes")
    ]
    tasks: list[dict[str, Any]] = []
    if not clusters or task_limit <= 0:
        return tasks

    for index, family in enumerate(CLUSTER_FAMILIES):
        if len(tasks) >= task_limit:
            break
        target_clusters = [clusters[0]]
        if family == "CLUSTER_HYBRID" and len(clusters) > 1:
            target_clusters = clusters[:2]
        tasks.append({
            "family": family,
            "task_id": f"{family.lower()}_{len(tasks) + 1}",
            "clusters": target_clusters,
            "editable_sentence_indexes": sorted({
                int(sentence_index)
                for cluster in target_clusters
                for sentence_index in (cluster.get("editable_sentence_indexes") or [])
            }),
            "targeted_drivers": _targeted_drivers_for_family(family),
        })
    return tasks[:task_limit]


def remaining_cluster_candidate_prompt(text: str, report_dict: dict | None, task: dict[str, Any]) -> str:
    profile = turnitin_like_ai_profile_from_report(report_dict or {})
    sentences = split_sentences(text)
    cluster_scopes = []
    for cluster in task.get("clusters") or []:
        start = int(cluster.get("start_sentence") or 0)
        end = int(cluster.get("end_sentence") or start)
        cluster_scopes.append({
            "cluster_id": cluster.get("cluster_id"),
            "start_sentence": start,
            "end_sentence": end,
            "protected_anchor_terms": cluster.get("protected_anchor_terms") or [],
            "editable_sentence_indexes": cluster.get("editable_sentence_indexes") or [],
            "sentences": [
                {
                    "sentence_index": index,
                    "classification": _sentence_classification(sentences[index]) if 0 <= index < len(sentences) else "missing",
                    "sentence": sentences[index] if 0 <= index < len(sentences) else "",
                }
                for index in range(start, end + 1)
                if 0 <= index < len(sentences)
            ],
        })
    schema = {
        "strategy": task.get("family"),
        "targeted_drivers": task.get("targeted_drivers") or [],
        "fact_inventory_preserved": True,
        "protected_anchors_preserved": True,
        "unsupported_new_facts": False,
        "cluster_patches": [
            {
                "cluster_id": "rc001_000_004",
                "start_sentence": 0,
                "end_sentence": 4,
                "replacement_text": "replacement text for this cluster only"
            }
        ],
    }
    return (
        "DraftProof REMAINING_CLUSTER_DENSITY_CANDIDATE.\n"
        "Objective: reduce total Turnitin-like AI score, unsafe eligible density, calibrated Top-k, and AI likelihood inside selected remaining clusters only.\n"
        "Return only valid JSON. Do not rewrite the whole document.\n\n"
        f"Candidate family: {task.get('family')}\n"
        f"Targeted drivers: {json.dumps(task.get('targeted_drivers') or [], ensure_ascii=False)}\n"
        f"Current formula profile: {json.dumps(profile, ensure_ascii=False)[:3600]}\n\n"
        "Hard rules:\n"
        "- Patch only the selected cluster spans below.\n"
        "- Preserve dates, numbers, named entities, citations, and core factual claims.\n"
        "- Preserve canonical factual sentences unless they are repeated elsewhere in the replacement.\n"
        "- No personal voice, fake evidence, fake dates, fake people, fake sources, or broad document rewrite.\n"
        "- Prefer compressing repeated generic explanation, breaking claim-explain symmetry, limited sentence-order/pacing change, and removing empty transition prose.\n"
        "- Do not polish the cluster into cleaner essay prose.\n\n"
        f"Document protected anchors: {json.dumps(protected_anchor_terms(text), ensure_ascii=False)[:2200]}\n\n"
        f"Cluster task: {json.dumps(task, ensure_ascii=False)[:5000]}\n\n"
        f"Cluster scopes: {json.dumps(cluster_scopes, ensure_ascii=False, indent=2)[:10000]}\n\n"
        "Return JSON matching this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def extract_remaining_cluster_payload(raw: str) -> tuple[dict[str, Any] | None, str]:
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
    patches = payload.get("cluster_patches")
    if not isinstance(patches, list) or not patches:
        return None, "missing_cluster_patches"
    valid = []
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        try:
            start = int(patch.get("start_sentence"))
            end = int(patch.get("end_sentence"))
        except (TypeError, ValueError):
            continue
        replacement = str(patch.get("replacement_text") or "").strip()
        if replacement and end >= start:
            valid.append({
                "cluster_id": str(patch.get("cluster_id") or ""),
                "start_sentence": start,
                "end_sentence": end,
                "replacement_text": replacement,
            })
    if not valid:
        return None, "missing_valid_cluster_patches"
    payload["cluster_patches"] = valid
    return payload, ""


def _missing_required_terms(original: str, replacement: str, protected: list[str]) -> list[str]:
    missing = []
    replacement_lower = str(replacement or "").lower()
    for term in protected:
        value = str(term or "").strip()
        if not value:
            continue
        if value.lower() in str(original or "").lower() and value.lower() not in replacement_lower:
            missing.append(value)
    return missing


def assemble_remaining_cluster_candidate(
    source_text: str,
    payload: dict[str, Any],
    task: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], str]:
    source = str(source_text or "")
    sentences = split_sentences(source)
    if not sentences:
        return "", [], "empty_source"
    allowed_clusters = [
        (
            int(cluster.get("start_sentence") or 0),
            int(cluster.get("end_sentence") or 0),
            cluster,
        )
        for cluster in (task.get("clusters") or [])
        if isinstance(cluster, dict)
    ]
    next_text = source
    applied: list[dict[str, Any]] = []
    touched: set[tuple[int, int]] = set()
    for patch in payload.get("cluster_patches") or []:
        try:
            start = int(patch.get("start_sentence"))
            end = int(patch.get("end_sentence"))
        except (TypeError, ValueError):
            continue
        if (start, end) in touched:
            continue
        if start < 0 or end >= len(sentences) or end < start:
            continue
        cluster = None
        for cluster_start, cluster_end, allowed_cluster in allowed_clusters:
            if start >= cluster_start and end <= cluster_end:
                cluster = allowed_cluster
                break
        if cluster is None:
            continue
        original = " ".join(sentences[start : end + 1]).strip()
        replacement = str(patch.get("replacement_text") or "").strip()
        if not replacement or replacement == original:
            continue
        if _word_count(replacement) < max(8, min(18, int(_word_count(original) * 0.25))):
            continue
        missing = _missing_required_terms(
            original,
            replacement,
            list(cluster.get("protected_anchor_terms") or []),
        )
        if missing:
            return "", applied, "protected_anchor_lost " + ", ".join(missing[:6])
        if any(is_canonical_fact_sentence(sentence) for sentence in sentences[start : end + 1]):
            original_anchors = _cluster_protected_anchor_terms(original, limit=20)
            missing_canonical = _missing_required_terms(original, replacement, original_anchors)
            if missing_canonical:
                return "", applied, "canonical_fact_anchor_lost " + ", ".join(missing_canonical[:6])
        replaced = next_text.replace(original, replacement, 1)
        if replaced == next_text:
            continue
        next_text = replaced
        touched.add((start, end))
        applied.append({
            "cluster_id": cluster.get("cluster_id") or patch.get("cluster_id"),
            "start_sentence": start,
            "end_sentence": end,
            "original_word_count": _word_count(original),
            "replacement_word_count": _word_count(replacement),
            "protected_anchor_count": len(cluster.get("protected_anchor_terms") or []),
        })
    candidate = next_text.strip()
    if not applied:
        return "", [], "no_applicable_cluster_patches"
    if candidate == source.strip():
        return "", applied, "unchanged_after_cluster_patches"
    return candidate, applied, ""


def remaining_cluster_patchwork_budget(
    source_text: str,
    candidate_text: str,
    applied: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sentences = split_sentences(source_text)
    if applied:
        edited = sum(
            max(1, int(row.get("end_sentence") or 0) - int(row.get("start_sentence") or 0) + 1)
            for row in applied
            if isinstance(row, dict)
        )
    else:
        source_sentences = [re.sub(r"\s+", " ", item).strip() for item in sentences]
        candidate_sentences = [re.sub(r"\s+", " ", item).strip() for item in split_sentences(candidate_text)]
        matcher = SequenceMatcher(None, source_sentences, candidate_sentences, autojunk=False)
        edited = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")
    total = max(1, len(sentences))
    ratio = edited / total
    max_ratio = 0.50 if total <= 10 else 0.24
    max_edited = max(2, min(12, int(total * max_ratio)))
    accepted = bool(edited <= max_edited and ratio <= max_ratio)
    return {
        "version": "remaining_cluster_patchwork_budget_v1",
        "accepted": accepted,
        "edited_sentence_count": edited,
        "edited_sentence_ratio": round(ratio, 3),
        "max_edited_sentences": max_edited,
        "max_edited_sentence_ratio": max_ratio,
        "reason": "within_patchwork_budget" if accepted else "remaining_cluster_patchwork_budget_exceeded",
    }
