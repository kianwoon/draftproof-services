from __future__ import annotations

from typing import Any

from .text import Paragraph


def paragraph_sentence_plan(paragraph: Paragraph, beats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for beat in beats:
        by_source.setdefault(str(beat.get("source_sentence_id") or ""), []).append(beat)
    rows: list[dict[str, Any]] = []
    for sentence in paragraph.sentences:
        source_id = sentence.id
        slot_beats = by_source.get(source_id, [])
        if not slot_beats:
            rows.append({
                "slot_id": f"{source_id}_slot",
                "source_sentence_id": source_id,
                "coverage_beat_ids": [],
                "slot_goal": "carry the source sentence role without adding a new route",
                "sentence_route": "one ordinary sentence",
                "coverage_capsules": [sentence.text],
            })
            continue
        markers = {
            marker.casefold()
            for beat in slot_beats
            for marker in beat.get("polarity_markers", [])
        }
        tags = {tag for beat in slot_beats for tag in beat.get("finding_tags", [])}
        rows.append({
            "slot_id": f"{source_id}_slot",
            "source_sentence_id": source_id,
            "coverage_beat_ids": [str(beat.get("beat_id")) for beat in slot_beats if beat.get("beat_id")],
            "slot_goal": _sentence_slot_goal(tags, markers, slot_beats),
            "sentence_route": _sentence_slot_route(tags, markers, slot_beats),
            "coverage_capsules": [str(beat.get("coverage_capsule") or "") for beat in slot_beats if beat.get("coverage_capsule")],
            "starter_terms": _dedupe_terms([
                str(term)
                for beat in slot_beats
                for term in beat.get("starter_terms", [])
                if str(term).strip()
            ])[:6],
        })
    return rows


def _sentence_slot_goal(tags: set[str], markers: set[str], beats: list[dict[str, Any]]) -> str:
    if "not always" in markers:
        return "carry the positive side and limited side as one contrast route without repeating the limitation"
    if any("not only" in marker for marker in markers):
        return "carry the first side as insufficient by itself, then attach the also-matters side"
    if _has_grouped_capsule(beats):
        return "carry each grouped capsule as a paired relation; never rejoin the groups into a comma list"
    if tags & {"packed_list", "sentence_overload"} or len(beats) > 1:
        return "group related coverage beats into one or two connected prose sentences, not a repair trace"
    if tags & {"broad_claim", "unsupported_claim_gap", "author_anchor_gap", "context_anchor_gap"}:
        return "narrow the claim through a source/context anchor inside the same sentence"
    return "carry the source relation in plain prose"


def _sentence_slot_route(tags: set[str], markers: set[str], beats: list[dict[str, Any]]) -> str:
    if "not always" in markers:
        return "one contrast sentence; add a follow-up only for a separate remaining limited item"
    if any("not only" in marker for marker in markers):
        return "one not-only sentence plus an also-side clause or follow-up sentence"
    if _has_grouped_capsule(beats):
        return "one paired relation sentence plus one short follow-up, or two short paired-relation sentences"
    if tags & {"packed_list", "sentence_overload"} or len(beats) > 1:
        return "one sentence may map several coverage_beat_ids; avoid repeated one-item sentences"
    if tags & {"broad_claim", "unsupported_claim_gap", "author_anchor_gap", "context_anchor_gap"}:
        return "source anchor first, scoped interpretation second"
    return "plain source relation sentence"


def _has_grouped_capsule(beats: list[dict[str, Any]]) -> bool:
    return any(bool(beat.get("grouped_relation")) for beat in beats)


def _dedupe_terms(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows
