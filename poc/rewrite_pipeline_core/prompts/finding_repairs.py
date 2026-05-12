"""Prompt and patch helpers for local finding repair."""

from __future__ import annotations

import json
import re


def _sentences_from_excerpt(text: str) -> list[str]:
    text = " ".join(str(text or "").split()).strip()
    if not text:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip().split()) >= 5
    ]


def _exact_blocking_target_from_context(
    item: dict,
    context: dict,
    candidate_text: str,
) -> tuple[str, str]:
    candidate_text = str(candidate_text or "")
    target = (
        context.get("target_sentence")
        or item.get("target_sentence")
        or item.get("sentence")
        or ""
    )
    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        target = target or evidence.get("text") or evidence.get("sentence") or ""
    elif isinstance(evidence, str):
        target = target or evidence
    target = " ".join(str(target or "").split()).strip()
    if target and (not candidate_text or target in candidate_text):
        expanded = _expand_fragment_to_candidate_sentence(candidate_text, target)
        if expanded:
            return expanded, (
                "explicit_target_expanded_to_sentence"
                if expanded != target else "explicit_target"
            )
        return target, "explicit_target"

    excerpt = str(context.get("paragraph_excerpt") or "")
    sentences = _sentences_from_excerpt(excerpt)
    title = str(item.get("title") or "").lower()
    category = str(item.get("category") or "").lower()
    preferred_patterns = []
    if "draft_evolution" in title or "ai_generation" in category:
        preferred_patterns.extend([
            r"\bin this review\b",
            r"\bhas been examined\b",
            r"\bfurthermore\b",
            r"\bwhereas\b",
            r"\bit arises from\b",
        ])
    if "predictability" in title:
        preferred_patterns.extend([
            r"\bwhen people\b",
            r"\bthis does not\b",
            r"\bstill must\b",
            r"\bthere are\b",
        ])

    def exact(sentence: str) -> bool:
        return bool(sentence and (not candidate_text or sentence in candidate_text))

    for pattern in preferred_patterns:
        for sentence in sentences:
            if exact(sentence) and re.search(pattern, sentence, flags=re.I):
                return sentence, "paragraph_excerpt_preferred_sentence"
    for sentence in sentences:
        if exact(sentence):
            return sentence, "paragraph_excerpt_sentence"
    return "", ""


def _expand_fragment_to_candidate_sentence(candidate_text: str, fragment: str) -> str:
    """Return the full candidate sentence that contains a scanner fragment."""
    candidate = " ".join(str(candidate_text or "").split()).strip()
    needle = " ".join(str(fragment or "").split()).strip()
    if not candidate or not needle or needle not in candidate:
        return ""
    start = candidate.find(needle)
    end = start + len(needle)
    sentences = _sentences_from_excerpt(candidate)
    cursor = 0
    overlapping: list[str] = []
    for sentence in sentences:
        sentence_start = candidate.find(sentence, cursor)
        if sentence_start < 0:
            continue
        sentence_end = sentence_start + len(sentence)
        cursor = sentence_end
        if sentence_end > start and sentence_start < end:
            overlapping.append(sentence)
    if overlapping:
        return " ".join(overlapping)
    return needle


def _blocking_finding_targets(
    report_dict: dict | None,
    *,
    limit: int = 3,
    candidate_text: str = "",
) -> list[dict]:
    if not isinstance(report_dict, dict):
        return []
    findings = report_dict.get("findings") or {}
    rows = []
    for tier in ("critical", "high", "medium"):
        for item in findings.get(tier, []) or []:
            if not isinstance(item, dict):
                continue
            context = item.get("rewrite_context") or {}
            target, target_source = _exact_blocking_target_from_context(
                item,
                context,
                candidate_text,
            )
            if not target:
                continue
            row = {
                "tier": tier,
                "finding_id": item.get("finding_id"),
                "title": item.get("title"),
                "category": item.get("category"),
                "detail": item.get("detail"),
                "recommendation": item.get("recommendation"),
                "target_sentence": str(target or "").strip(),
                "target_source": target_source,
                "paragraph_excerpt": str(context.get("paragraph_excerpt") or "")[:700],
                "signals": context.get("signals") or {},
            }
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows[:limit]


def _finding_local_repair_prompt(
    blocked_candidate: str,
    blocked_summary: dict,
    targets: list[dict],
    attempt_index: int,
) -> str:
    return (
        "DraftProof FINDING_LOCAL_BLOCKED_WINNER_REPAIR.\n"
        "A high-Human candidate failed because specific findings became too severe. "
        "Patch only the listed finding targets. Do not rewrite the whole document.\n\n"
        f"Blocked candidate scorecard: {json.dumps(blocked_summary, ensure_ascii=False)[:2200]}\n\n"
        "Blocking findings to repair:\n"
        f"{json.dumps(targets, ensure_ascii=False, indent=2)[:3200]}\n\n"
        "Rules:\n"
        "- Return JSON only.\n"
        "- Each patch must target one exact sentence or short paragraph span from the blocked candidate.\n"
        "- Replacement must narrow or qualify the unsafe claim; do not add evidence.\n"
        "- Preserve the Human Contribution structure and author reasoning where possible.\n"
        "- Do not invent sources, examples, personal experiences, institutions, statistics, or citations.\n"
        "- Do not change unaffected sentences.\n"
        "- If a target is not exact enough to patch safely, omit it.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "patches": [\n'
        '    {"target": "exact text from blocked candidate", "replacement": "safer replacement text"}\n'
        "  ]\n"
        "}\n\n"
        f"Attempt {attempt_index}.\n\n"
        "BLOCKED CANDIDATE:\n"
        f"<BLOCKED_CANDIDATE>\n{blocked_candidate.strip()}\n</BLOCKED_CANDIDATE>"
    )


def _extract_finding_local_patches(output: str) -> list[dict]:
    text = str(output or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        payload = json.loads(text)
    except Exception:
        return []
    patches = payload.get("patches") if isinstance(payload, dict) else payload
    if not isinstance(patches, list):
        return []
    cleaned = []
    polished_patch_patterns = [
        r"\bthere are instances where\b",
        r"\balign with\b",
        r"\bemphasizing aspects such as\b",
        r"\bit is important to\b",
        r"\bplays? a crucial role\b",
        r"\bserves as\b",
        r"\bhighlights? the importance\b",
        r"\bunderscores? the need\b",
    ]
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        target = " ".join(str(patch.get("target") or "").split()).strip()
        replacement = " ".join(str(patch.get("replacement") or "").split()).strip()
        if len(target.split()) < 5 or len(replacement.split()) < 5:
            continue
        if "[[" in replacement or "]]" in replacement:
            continue
        if any(re.search(pattern, replacement, flags=re.I) for pattern in polished_patch_patterns):
            continue
        cleaned.append({"target": target, "replacement": replacement})
    return cleaned[:5]


def _apply_finding_local_patches(text: str, patches: list[dict]) -> tuple[str, list[dict]]:
    updated = str(text or "")
    applied = []
    for patch in patches or []:
        target = str(patch.get("target") or "").strip()
        replacement = str(patch.get("replacement") or "").strip()
        if not target or not replacement or target not in updated:
            continue
        updated = updated.replace(target, replacement, 1)
        applied.append({
            "target": target[:220],
            "replacement": replacement[:220],
        })
    return updated, applied
