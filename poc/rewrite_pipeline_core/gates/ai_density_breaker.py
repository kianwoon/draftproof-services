"""Low-level AI-density route helpers.

The full post-selection density controller still lives behind the public
``rewrite_pipeline`` facade.  These helpers are pure enough to extract first.
"""

from __future__ import annotations

import re


_AI_DENSITY_CANONICAL_FACT_RE = re.compile(
    r"(?:\b\d{3,4}\b|https?://|www\.|\[[^\]]+\]|\([^)]*\d{4}[^)]*\)|"
    r"\b[A-Z]{2,}[A-Z0-9-]{2,}\b)",
)

_AI_DENSITY_GENERIC_RE = re.compile(
    r"\b(?:important|significant|major|strong|influential|different|many|"
    r"various|complex|global|modern|today|society|culture|system|country|"
    r"world|success|challenge|opportunity|influence|impact|role|feature|"
    r"strength|development|diversity|economy|people)\b",
    re.I,
)

_AI_DENSITY_TRANSITION_RE = re.compile(
    r"^(?:Furthermore|Moreover|Additionally|In conclusion|Overall|Therefore|"
    r"However|At the same time|In addition|Despite|Another important|"
    r"One of the|This|These|It is important)\b",
    re.I,
)


def _ai_density_breaker_canonical_fact_sentence(sentence: str) -> bool:
    """Preserve canonical factual/anchor-heavy sentences in the add-on layer."""
    value = str(sentence or "").strip()
    if not value:
        return True
    if _AI_DENSITY_CANONICAL_FACT_RE.search(value):
        return True
    proper = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", value)
    if len(proper) >= 4 and not _AI_DENSITY_TRANSITION_RE.search(value):
        return True
    return False


def _ai_density_breaker_sentence_route(sentence: str) -> tuple[str, list[str]]:
    """Apply a small non-personal route change to generic/transition prose."""
    original = str(sentence or "").strip()
    if not original or _ai_density_breaker_canonical_fact_sentence(original):
        return original, []
    candidate = original
    operations: list[str] = []
    replacements = [
        (
            r"^One of the biggest strengths of (?P<subject>.+?) is (?P<claim>[^.]+)\.$",
            lambda m: f"For {m.group('subject').strip()}, {m.group('claim').strip()} remains one clear strength.",
            "ranked_claim_route",
        ),
        (
            r"^Another important feature of (?P<subject>.+?) is (?P<claim>[^.]+)\.$",
            lambda m: f"Another part of {m.group('subject').strip()} is {m.group('claim').strip()}.",
            "feature_route_reduce",
        ),
        (
            r"^The country has one of the largest (?P<asset>[^.]+?) in the world and is home to many (?P<group>[^.]+)\.$",
            lambda m: f"Its {m.group('asset').strip()} is large, and many {m.group('group').strip()} operate from there.",
            "largest_asset_route_reduce",
        ),
        (
            r"^At the same time,\s*(?P<body>.+)$",
            lambda m: f"Still, {m.group('body').strip()[0].lower() + m.group('body').strip()[1:]}",
            "transition_break",
        ),
        (
            r"^In addition to\s+(?P<body>.+)$",
            lambda m: f"Beyond {m.group('body').strip()}",
            "transition_break",
        ),
        (
            r"^Despite its success,\s*(?P<body>.+)$",
            lambda m: f"That success has limits. {m.group('body').strip()[0].upper() + m.group('body').strip()[1:]}",
            "transition_split",
        ),
        (
            r"^However,\s*(?P<body>.+)$",
            lambda m: f"But {m.group('body').strip()[0].lower() + m.group('body').strip()[1:]}",
            "transition_plain",
        ),
    ]
    for pattern, replacement, operation in replacements:
        match = re.match(pattern, candidate, flags=re.I)
        if not match:
            continue
        updated = replacement(match)
        if updated and updated != candidate:
            candidate = updated
            operations.append(operation)
            break
    phrase_replacements = [
        (r"\bplays? (?:a|an) (?:important|significant|major|crucial) role in\b", "matters in", "formula_verb_reduce"),
        (r"\bhas a significant impact on\b", "affects", "formula_verb_reduce"),
        (r"\bhas a strong influence on\b", "influences", "formula_verb_reduce"),
        (r"\bis one of the most influential\b", "has wide influence", "formula_route_reduce"),
        (r"\ba wide range of\b", "many", "generic_phrase_reduce"),
    ]
    for pattern, replacement, operation in phrase_replacements:
        updated = re.sub(pattern, replacement, candidate, count=1, flags=re.I)
        if updated != candidate:
            candidate = updated
            operations.append(operation)
            break
    if candidate == original and _AI_DENSITY_TRANSITION_RE.search(candidate):
        updated = re.sub(r"^This means that\s+", "That means ", candidate, flags=re.I)
        updated = re.sub(r"^This has led to\s+", "That has left ", updated, flags=re.I)
        updated = re.sub(r"^This shows that\s+", "That shows ", updated, flags=re.I)
        if updated != candidate:
            candidate = updated
            operations.append("this_route_reduce")
    candidate = re.sub(r"\s{2,}", " ", candidate).strip()
    return (candidate, operations) if candidate and candidate != original else (original, [])


def _ai_density_window_targets(sentence_rows: list[dict] | None, *, max_windows: int = 6) -> list[dict]:
    """Rank 3-5 sentence windows by document-level AI-density drag."""
    rows = [row for row in (sentence_rows or []) if isinstance(row, dict)]
    if not rows:
        return []
    raw_windows: list[dict] = []
    for start in range(len(rows)):
        for size in (3, 4, 5):
            end = start + size
            if end > len(rows):
                continue
            window_rows = rows[start:end]
            editable = [
                row for row in window_rows
                if not row.get("canonical_fact_preserved")
                and float(row.get("score") or 0.0) > 0.0
            ]
            if not editable:
                continue
            transition_count = sum(1 for row in editable if row.get("transition_risk"))
            generic_hits = sum(int(row.get("generic_hits") or 0) for row in editable)
            canonical_count = sum(1 for row in window_rows if row.get("canonical_fact_preserved"))
            score = (
                sum(max(0.0, float(row.get("score") or 0.0)) for row in editable)
                + transition_count * 1.25
                + generic_hits * 0.35
                - canonical_count * 1.5
            )
            if score <= 0.0:
                continue
            raw_windows.append({
                "start_sentence": start,
                "end_sentence": end - 1,
                "sentence_count": size,
                "editable_sentence_count": len(editable),
                "score": round(score, 3),
                "editable_sentences": [
                    {
                        "sentence_index": row.get("sentence_index"),
                        "sentence": row.get("sentence"),
                        "score": row.get("score"),
                        "top10_ratio": row.get("top10_ratio"),
                        "transition_risk": row.get("transition_risk"),
                        "generic_hits": row.get("generic_hits"),
                    }
                    for row in editable
                ],
                "preview": " ".join(str(row.get("sentence") or "") for row in window_rows)[:320],
            })
    raw_windows.sort(key=lambda row: (float(row.get("score") or 0.0), int(row.get("editable_sentence_count") or 0)), reverse=True)
    selected: list[dict] = []
    occupied: set[int] = set()
    for row in raw_windows:
        indexes = set(range(int(row["start_sentence"]), int(row["end_sentence"]) + 1))
        if len(indexes & occupied) > 1:
            continue
        selected.append(row)
        occupied.update(indexes)
        if len(selected) >= max_windows:
            break
    return selected
