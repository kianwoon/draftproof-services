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
            "source_terms_to_carry": _slot_terms(slot_beats),
            "must_cover_terms": _exact_anchor_terms(_slot_terms(slot_beats)),
            "revoiceable_source_terms": _revoiceable_terms(_slot_terms(slot_beats)),
            "coverage_loss_failure": "Do not improve the score by dropping source meaning. Exact anchors must survive; revoiceable source terms may be rewritten in plain words.",
            "required_sentence_groups": _required_sentence_groups(tags, slot_beats),
            "starter_terms": _dedupe_terms([
                str(term)
                for beat in slot_beats
                for term in beat.get("starter_terms", [])
                if str(term).strip()
            ])[:6],
        })
    return rows


def coverage_loss_contract(sentence_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in sentence_plan:
        if not isinstance(slot, dict):
            continue
        for group in slot.get("required_sentence_groups") or []:
            if not isinstance(group, dict):
                continue
            rows.append({
                "source_sentence_id": slot.get("source_sentence_id"),
                "group_id": group.get("group_id"),
                "coverage_beat_ids": group.get("coverage_beat_ids", []),
                "must_cover_terms": group.get("must_cover_terms", []),
                "source_terms_to_carry": group.get("source_terms_to_carry", []),
                "failure": "candidate is invalid if this group is omitted or only implied by a broad summary",
            })
    return rows[:24]


def _sentence_slot_goal(tags: set[str], markers: set[str], beats: list[dict[str, Any]]) -> str:
    if "not always" in markers:
        return "carry the positive side and limited side as one contrast route without repeating the limitation"
    if any("not only" in marker for marker in markers):
        return "carry the first side as insufficient by itself, then attach the also-matters side"
    if _has_grouped_capsule(beats):
        return "carry each grouped capsule as a paired relation; never rejoin the groups into a comma list"
    if "sentence_overload" in tags and len(_slot_terms(beats)) > 12:
        return "split the required terms across two connected ordinary sentences before adding interpretation"
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
    if "sentence_overload" in tags and len(_slot_terms(beats)) > 12:
        return "two connected ordinary sentences; do not compress the slot into one long sentence"
    if tags & {"packed_list", "sentence_overload"} or len(beats) > 1:
        return "one sentence may map several coverage_beat_ids; avoid repeated one-item sentences"
    if tags & {"broad_claim", "unsupported_claim_gap", "author_anchor_gap", "context_anchor_gap"}:
        return "source anchor first, scoped interpretation second"
    return "plain source relation sentence"


def _has_grouped_capsule(beats: list[dict[str, Any]]) -> bool:
    return any(bool(beat.get("grouped_relation")) for beat in beats)


def _slot_terms(beats: list[dict[str, Any]]) -> list[str]:
    return _dedupe_terms([
        str(term)
        for beat in beats
        for term in beat.get("coverage_terms", [])
        if str(term).strip()
    ])[:24]


def _required_sentence_groups(tags: set[str], beats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ("sentence_overload" in tags and len(_slot_terms(beats)) > 12):
        return []
    groups: list[dict[str, Any]] = []
    for beat in beats:
        if not beat.get("beat_id"):
            continue
        terms = [str(term) for term in beat.get("coverage_terms", []) if str(term).strip()]
        for chunk_index, chunk in enumerate(_term_chunks(terms), start=1):
            groups.append({
                "coverage_beat_ids": [str(beat.get("beat_id"))],
                "group_id": f"{beat.get('beat_id')}_g{chunk_index:02d}",
                "source_terms_to_carry": chunk,
                "must_cover_terms": _exact_anchor_terms(chunk),
                "revoiceable_source_terms": _revoiceable_terms(chunk),
                "instruction": "cover this group in the final text; adjacent groups may share one ordinary sentence when exact anchors and plain meaning remain clear",
            })
    return groups


def _term_chunks(terms: list[str], *, size: int = 10) -> list[list[str]]:
    clean = [term for term in terms if str(term).strip()]
    if len(clean) <= 12:
        return [clean] if clean else []
    return [clean[index:index + size] for index in range(0, len(clean), size)]


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


def _exact_anchor_terms(terms: list[str]) -> list[str]:
    exact: list[str] = []
    for term in terms:
        if _is_exact_anchor(term):
            exact.append(term)
    return exact


def _revoiceable_terms(terms: list[str]) -> list[str]:
    return [term for term in terms if not _is_exact_anchor(term)]


def _is_exact_anchor(term: str) -> bool:
    text = str(term or "").strip()
    if not text:
        return False
    if text.casefold() in {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand"}:
        return True
    if any(char.isdigit() for char in text):
        return True
    if text.isupper() and len(text) >= 2:
        return True
    if "-" in text:
        return True
    if text[:1].isupper():
        return True
    return False
