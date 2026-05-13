"""Academic rewrite layers for rewrite V2."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from llm.gateway import LLMGateway
from rewrite.guards import detect_protected_spans

from ..strategy import clean_candidate_output


def _json_from_response(raw: str) -> dict[str, Any]:
    text = clean_candidate_output(raw)
    if not text:
        return {}
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _supports_openai_penalties(model: str | None) -> bool:
    name = (model or "").lower()
    if not name:
        return True
    if "gpt-5" in name or "o1" in name or "o3" in name:
        return False
    return True


def _supports_repetition_penalty(model: str | None) -> bool:
    name = (model or "").lower()
    return any(provider in name for provider in ("deepseek", "qwen", "mistral", "llama", "anthropic"))


def _section_heading_pattern() -> re.Pattern[str]:
    return re.compile(
        r"(?m)^(?P<heading>\s*(?:Question|Section|Part|Task|Prompt|Answer)\s+\d+[A-Za-z]?\s*:?\s*)$"
    )


def _split_academic_sections(text: str) -> list[dict[str, Any]]:
    """Split academic prose into conservative heading sections.

    The resolver only rewrites sections it can replace exactly. If no explicit
    headings exist, it falls back to citation-bearing paragraph groups.
    """
    source = str(text or "")
    matches = list(_section_heading_pattern().finditer(source))
    sections: list[dict[str, Any]] = []
    if matches:
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
            section_text = source[start:end].strip()
            if section_text:
                heading = match.group("heading").strip()
                sections.append({
                    "section_id": f"s{index + 1:03d}",
                    "heading": heading.rstrip(":"),
                    "text": section_text,
                    "start": start,
                    "end": end,
                    "word_count": len(re.findall(r"\b[\w'-]+\b", section_text)),
                    "citations": _exact_citation_markers(section_text),
                })
        return sections

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", source.strip())
        if paragraph.strip()
    ]
    cursor = 0
    for index, paragraph in enumerate(paragraphs):
        start = source.find(paragraph, cursor)
        if start < 0:
            start = cursor
        end = start + len(paragraph)
        cursor = end
        citations = _exact_citation_markers(paragraph)
        if citations:
            sections.append({
                "section_id": f"p{index + 1:03d}",
                "heading": f"Paragraph {index + 1}",
                "text": paragraph,
                "start": start,
                "end": end,
                "word_count": len(re.findall(r"\b[\w'-]+\b", paragraph)),
                "citations": citations,
            })
    return sections


def _academic_sections_from_handoff(original_text: str, scan_report: dict[str, Any]) -> list[dict[str, Any]]:
    units = (((scan_report or {}).get("generation_handoff") or {}).get("section_generation_units") or [])
    if not isinstance(units, list):
        return []
    sections: list[dict[str, Any]] = []
    source = str(original_text or "")
    prepared: list[tuple[int, dict[str, Any], int, int]] = []
    for index, unit in enumerate(units, start=1):
        if not isinstance(unit, dict):
            continue
        source_span = unit.get("source_span") if isinstance(unit.get("source_span"), dict) else {}
        start = source_span.get("start_char")
        end = source_span.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            continue
        meaning = unit.get("meaning_inventory") if isinstance(unit.get("meaning_inventory"), list) else []
        first_keywords = []
        if meaning and isinstance(meaning[0], dict):
            first_keywords = [str(item).strip() for item in (meaning[0].get("keywords") or []) if str(item).strip()]
        for size in (4, 3, 2, 1):
            phrase = " ".join(first_keywords[:size])
            if not phrase:
                continue
            window_start = max(0, start - 180)
            window_end = min(len(source), start + 180)
            found = source.find(phrase, window_start, window_end)
            if found >= 0:
                if size == 1:
                    sentence_start = max(
                        source.rfind(". ", window_start, found),
                        source.rfind("? ", window_start, found),
                        source.rfind("! ", window_start, found),
                    )
                    start = sentence_start + 2 if sentence_start >= 0 else found
                else:
                    start = found
                break
        prepared.append((index, unit, start, end))
    prepared.sort(key=lambda row: row[2])
    adjusted: list[tuple[int, dict[str, Any], int, int]] = []
    for row_index, (index, unit, start, end) in enumerate(prepared):
        next_start = prepared[row_index + 1][2] if row_index + 1 < len(prepared) else len(source)
        adjusted.append((index, unit, start, max(start, min(end, next_start))))
    for index, unit, start, end in adjusted:
        target_text = source[start:end].strip()
        section_text = target_text
        if not section_text:
            continue
        citation_keys = [str(key).strip() for key in (unit.get("citation_keys_used") or []) if str(key).strip()]
        citations = _exact_citation_markers(section_text)
        for citation in (((unit.get("must_preserve_anchors") or [])) if isinstance(unit.get("must_preserve_anchors"), list) else []):
            value = str(citation or "").strip()
            if (
                value
                and (value.startswith("(") or re.search(r"\bet al\.\s+\(\d{4}", value))
                and value not in citations
            ):
                citations.append(value)
        if citation_keys and not citations:
            citations = citation_keys
        heading = str(unit.get("heading") or f"Section {index}").strip()
        if heading.lower() == "main body" and index == 1:
            heading = "Question 1"
        if heading and not section_text.startswith(heading):
            section_text = f"{heading}\n{section_text}"
        sections.append({
            "section_id": str(unit.get("section_id") or f"sec_{index:03d}"),
            "heading": heading,
            "text": section_text,
            "target_text": target_text,
            "start": start,
            "end": end,
            "word_count": int(unit.get("current_word_count") or len(re.findall(r"\b[\w'-]+\b", section_text))),
            "citations": citations,
        })
    return sections


def _exact_citation_markers(text: str) -> list[str]:
    markers: list[str] = []
    source = str(text or "")
    patterns = [
        r"\([A-Z][A-Za-z' -]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}[a-z]?(?:,\s*p{1,2}\.?\s*\d+)?\)",
        r"\b[A-Z][A-Za-z' -]+(?:\s+et\s+al\.)?\s+\((?:19|20)\d{2}[a-z]?\)",
        r"\[(?:\d+(?:\s*[,\-]\s*\d+)*)\]",
        r"\b(?:doi|DOI):\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            value = match.group(0).strip()
            if value and value not in markers:
                markers.append(value)
    return markers


def _protected_texts_for_scope(text: str) -> list[str]:
    protected: list[str] = _exact_citation_markers(text)
    for value in _visual_reference_markers(text):
        if value and value not in protected:
            protected.append(value)
    for span in detect_protected_spans(str(text or "")):
        value = str(span.text or "").strip()
        if span.reason == "citation":
            continue
        if span.reason == "numeric" and not re.search(r"\d{2,}", value):
            continue
        if span.reason not in {"direct_quote", "numeric"}:
            continue
        if value and value not in protected:
            protected.append(value)
    return protected


def _visual_reference_markers(text: str) -> list[str]:
    markers: list[str] = []
    pattern = (
        r"\b(?:Figure|Fig\.?|Table|Appendix|Exhibit|Chart)\s+"
        r"\d+[A-Za-z]?(?:\s*(?:to|through|and|,|-|–)\s*\d+[A-Za-z]?)*\b"
    )
    for match in re.finditer(pattern, str(text or ""), flags=re.IGNORECASE):
        value = match.group(0).strip()
        if value and value not in markers:
            markers.append(value)
    return markers


def _academic_section_targets(original_text: str, scan_report: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    raw_sections = _split_academic_sections(original_text)
    if len(raw_sections) <= 1:
        handoff_sections = _academic_sections_from_handoff(original_text, scan_report)
        if handoff_sections:
            raw_sections = handoff_sections
    sections = [
        section for section in raw_sections
        if section.get("word_count", 0) >= 25 and section.get("citations")
    ]
    if not sections:
        return []
    top_clusters = (
        (((scan_report or {}).get("scan_intelligence") or {}).get("ai_detection") or {}).get("top_unsafe_clusters")
        or []
    )
    cluster_text = " ".join(
        _cluster_text_from_gate(original_text, cluster)
        for cluster in top_clusters
        if isinstance(cluster, dict)
    )
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, section in enumerate(sections):
        overlap = 0
        if cluster_text:
            section_sentences = set(_split_sentences(str(section.get("text") or "")))
            cluster_sentences = set(_split_sentences(cluster_text))
            overlap = len(section_sentences & cluster_sentences)
        citation_weight = len(section.get("citations") or [])
        # Prefer sections with citations, enough body text, and unsafe-cluster overlap.
        score = float(overlap * 10 + citation_weight * 2 + min(4, int(section.get("word_count", 0) / 120)))
        if not cluster_text and index == 0 and len(sections) > 1:
            score -= 1.0
        scored.append((score, index, section))
    scored.sort(key=lambda item: (-item[0], item[1]))
    max_sections = max(1, min(4, limit))
    selected = [item[2] for item in scored[:max_sections]]
    selected.sort(key=lambda section: int(section.get("start") or 0))
    for section in selected:
        section["protected_spans"] = _protected_texts_for_scope(str(section.get("text") or ""))
    return selected


def _build_academic_section_prompt(
    *,
    sections: list[dict[str, Any]],
    candidate_count: int,
) -> str:
    payload = {
        "task": "Rewrite only the provided citation-bearing academic sections.",
        "candidate_count": candidate_count,
        "sections": [
            {
                "section_id": section.get("section_id"),
                "heading": section.get("heading"),
                "word_count": section.get("word_count"),
                "required_exact_citations": section.get("citations") or [],
                "required_exact_protected_spans": section.get("protected_spans") or [],
                "source_section": section.get("text"),
            }
            for section in sections
        ],
    }
    return (
        "DraftProof academic cited section density resolver.\n"
        f"Create exactly {candidate_count} candidate section rewrite sets.\n"
        "Rewrite at section level, not sentence level and not full-document essay reconstruction.\n"
        "Each candidate must include exactly one rewritten section for every input section_id.\n"
        "Each rewritten_section must start with the original section heading exactly as given.\n"
        "Every required_exact_citation and required_exact_protected_span must appear verbatim in the same rewritten section.\n"
        "Keep the same claims, source attribution, numbers, figure references, service-model terms, and solution list.\n"
        "Do not add new sources, facts, examples, personal experience, or references.\n"
        "Reduce detector risk by changing section rhythm, sentence routes, and generic academic phrasing.\n"
        "Avoid over-polished textbook phrasing, but keep normal academic student prose.\n"
        "Do not use typos, fake errors, fragments, slang, or bullet points.\n"
        "Return the result with these exact delimiters, not JSON:\n"
        "===CANDIDATE 1===\n"
        "<<<SECTION section_id>>>\n"
        "rewritten section text\n"
        "<<<END SECTION>>>\n"
        "Repeat the section block for each input section. If more candidates are requested, use ===CANDIDATE 2===.\n\n"
        f"ACADEMIC_SECTION_RESOLVER_JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False, default=str)[:36000]}"
    )


def _parse_academic_section_delimited(raw: str) -> list[dict[str, Any]]:
    text = clean_candidate_output(raw)
    if not text:
        return []
    chunks = re.split(r"(?m)^===\s*CANDIDATE\s+(\d+)\s*===\s*$", text)
    candidates: list[dict[str, Any]] = []
    if len(chunks) > 1:
        iterator = zip(chunks[1::2], chunks[2::2])
    else:
        iterator = [("1", text)]
    for number, body in iterator:
        sections: list[dict[str, Any]] = []
        for match in re.finditer(
            r"(?ms)^<<<SECTION\s+([^>\s]+)\s*>>>\s*(.*?)\s*^<<<END SECTION>>>\s*",
            body,
        ):
            section_id = match.group(1).strip()
            rewritten = match.group(2).strip()
            if section_id and rewritten:
                sections.append({
                    "section_id": section_id,
                    "rewritten_section": rewritten,
                    "rationale": "delimiter_section_rewrite",
                })
        if sections:
            candidates.append({
                "candidate_id": f"academic_section_variant_{number}",
                "sections": sections,
            })
    return candidates


def _candidate_section_map(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates") if isinstance(payload, dict) else []
    if isinstance(candidates, list):
        return [candidate for candidate in candidates if isinstance(candidate, dict)]
    sections = payload.get("sections") if isinstance(payload, dict) else None
    if isinstance(sections, list):
        return [{"candidate_id": "academic_section_variant_1", "sections": sections}]
    return []


def _compose_academic_sections(original_text: str, sections: list[dict[str, Any]], patches: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    by_id = {
        str(patch.get("section_id") or ""): str(patch.get("rewritten_section") or "").strip()
        for patch in patches
        if isinstance(patch, dict)
    }
    text = str(original_text or "")
    applied: list[dict[str, Any]] = []
    for section in sorted(sections, key=lambda row: int(row.get("start") or 0), reverse=True):
        section_id = str(section.get("section_id") or "")
        replacement = by_id.get(section_id, "")
        source = str(section.get("target_text") or section.get("text") or "").strip()
        if not section_id or not replacement or not source:
            applied.append({"section_id": section_id, "applied": False, "reason": "missing_section_replacement"})
            continue
        heading = str(section.get("heading") or "").strip()
        if heading and not replacement.startswith(heading):
            replacement = f"{heading}\n{replacement}"
        start = int(section.get("start") or 0)
        end = int(section.get("end") or 0)
        if text[start:end].strip() != source:
            replaced_text, ok = _replace_once_flexible(text, source, replacement)
            text = replaced_text
            applied.append({"section_id": section_id, "applied": ok, "target_section": source, "rewritten_section": replacement})
            continue
        text = text[:start] + replacement + text[end:]
        applied.append({"section_id": section_id, "applied": True, "target_section": source, "rewritten_section": replacement})
    applied.reverse()
    return text, applied


def _academic_section_filter_failures(sections: list[dict[str, Any]], patches: list[dict[str, Any]]) -> list[str]:
    by_id = {
        str(patch.get("section_id") or ""): str(patch.get("rewritten_section") or "").strip()
        for patch in patches
        if isinstance(patch, dict)
    }
    failures: list[str] = []
    for section in sections:
        section_id = str(section.get("section_id") or "")
        rewritten = by_id.get(section_id, "")
        if not rewritten:
            failures.append(f"{section_id}:missing_rewritten_section")
            continue
        heading = str(section.get("heading") or "").strip()
        if heading and not rewritten.startswith(heading):
            failures.append(f"{section_id}:heading_not_preserved")
        for citation in section.get("citations") or []:
            if citation not in rewritten:
                failures.append(f"{section_id}:citation_lost:{citation}")
        for protected in section.get("protected_spans") or []:
            if protected not in rewritten:
                failures.append(f"{section_id}:protected_span_lost:{protected}")
    return failures[:20]


def _academic_assignment_sections(original_text: str, scan_report: dict[str, Any]) -> list[dict[str, Any]]:
    raw_sections = _split_academic_sections(original_text)
    if len(raw_sections) <= 1:
        handoff_sections = _academic_sections_from_handoff(original_text, scan_report)
        if handoff_sections:
            raw_sections = handoff_sections
    return [
        section for section in raw_sections
        if section.get("word_count", 0) >= 20
    ]


def _all_section_compact_allowed(original_text: str, scan_report: dict[str, Any]) -> bool:
    if os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_ALL_SECTION_COMPACT", "1").lower() in {"0", "false", "no"}:
        return False
    sections = _academic_assignment_sections(original_text, scan_report)
    if len(sections) < 2:
        return False
    headings = " ".join(str(section.get("heading") or "") for section in sections).lower()
    if re.search(r"\bquestion\s+\d+|section\s+\d+|part\s+\d+|task\s+\d+", headings):
        return True
    handoff = (scan_report or {}).get("generation_handoff") if isinstance(scan_report, dict) else {}
    profile = (handoff or {}).get("document_profile") if isinstance(handoff, dict) else {}
    document_type = str((profile or {}).get("document_type") or "").lower()
    return "assignment" in document_type or "analytical_submission" in document_type


def _build_academic_all_section_prompt(
    *,
    sections: list[dict[str, Any]],
    candidate_count: int,
) -> str:
    protected_all: list[str] = []
    for section in sections:
        for value in _protected_texts_for_scope(str(section.get("text") or "")):
            if value and value not in protected_all:
                protected_all.append(value)
    payload = {
        "task": "Rewrite all academic assignment sections as compact structured academic prose.",
        "candidate_count": candidate_count,
        "sections": [
            {
                "section_id": section.get("section_id"),
                "heading": section.get("heading"),
                "word_count": section.get("word_count"),
                "required_exact_citations": section.get("citations") or [],
                "required_exact_protected_spans": _protected_texts_for_scope(str(section.get("text") or "")),
                "source_section": section.get("text"),
            }
            for section in sections
        ],
        "global_required_protected_spans": protected_all,
    }
    return (
        "DraftProof academic all-section compact reconstruction.\n"
        f"Create exactly {candidate_count} variants.\n"
        "Use this only for assignment-like academic content with numbered/labelled sections.\n"
        "Rewrite every provided section, including setup/classification sections, because leaving the setup frozen can keep detector predictability high.\n"
        "Each variant must preserve every section heading exactly and in the same order.\n"
        "Every required citation, direct quote, number with two or more digits, and protected span must appear verbatim.\n"
        "Do not convert parenthetical citations into narrative citations, or narrative citations into parenthetical citations. "
        "For example, keep '(Gao et al., 2025)' exactly instead of writing 'Gao et al. (2025)'.\n"
        "Keep the same claims, source attribution, required terms, and solution inventory. Do not add new sources, facts, examples, statistics, or personal experience.\n"
        "Use compact structured academic reconstruction. Numbered subpoints are allowed when the source already has stages, answers, or solution lists.\n"
        "Add a short continuity bridge when a setup/classification section leads into analysis, so the rewrite does not create semantic-shape jumps.\n"
        "Avoid polished textbook phrasing and smooth template transitions. Keep a believable undergraduate analytical voice.\n"
        "Do not use typos, slang, random fragments, or fake mistakes.\n"
        "Return exact delimiters only:\n"
        "===VARIANT 1===\n"
        "Question 1\n"
        "...\n\n"
        "Question 2\n"
        "...\n"
        "===END===\n"
        "Repeat through the requested variant count.\n\n"
        f"ACADEMIC_ALL_SECTION_JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False, default=str)[:42000]}"
    )


def _parse_academic_all_section_variants(raw: str) -> list[dict[str, Any]]:
    text = clean_candidate_output(raw)
    if not text:
        return []
    variants: list[dict[str, Any]] = []
    for number, body in re.findall(r"(?ms)^===\s*VARIANT\s+(\d+)\s*===\s*(.*?)\s*^===\s*END\s*===\s*", text):
        rewritten = body.strip()
        if rewritten:
            variants.append({
                "candidate_id": f"academic_all_section_variant_{number}",
                "text": rewritten,
            })
    if not variants and text:
        variants.append({"candidate_id": "academic_all_section_variant_1", "text": text})
    return variants


def _normalize_academic_all_section_candidate(candidate_text: str, sections: list[dict[str, Any]]) -> str:
    text = clean_candidate_output(candidate_text)
    text = re.sub(r"(?m)^\s*\*\*((?:Question|Section|Part|Task|Prompt|Answer)\s+\d+[A-Za-z]?)\*\*\s*:?\s*$", r"\1", text)
    text = re.sub(r"(?m)^\s*#+\s*((?:Question|Section|Part|Task|Prompt|Answer)\s+\d+[A-Za-z]?)\s*:?\s*$", r"\1", text)
    text = re.sub(r"\*([^*\n]{4,120})\*", r"\1", text)
    for section in sections:
        heading = str(section.get("heading") or "").strip()
        if heading and not re.search(rf"(?m)^\s*{re.escape(heading)}\b", text):
            text = f"{heading}\n{text}"
            break
    for section in sections:
        source_text = str(section.get("text") or "")
        for span in detect_protected_spans(source_text):
            if span.reason != "direct_quote":
                continue
            exact = str(span.text or "").strip()
            if exact in text:
                continue
            content = exact.strip('"“”\'‘’').strip()
            straight = f'"{content}"'
            if straight in text:
                text = text.replace(straight, exact)
    text = _restore_exact_citation_forms(text, sections)
    text = _restore_visual_references(text, sections)
    return text.strip() + "\n"


def _restore_visual_references(candidate_text: str, sections: list[dict[str, Any]]) -> str:
    text = str(candidate_text or "")
    for section in sections:
        source_text = str(section.get("text") or "")
        missing_refs = [ref for ref in _visual_reference_markers(source_text) if ref not in text]
        if not missing_refs:
            continue
        heading = str(section.get("heading") or "").strip()
        sentence = " ".join(f"This section keeps {ref} as an evidence reference." for ref in missing_refs)
        if heading:
            heading_match = re.search(rf"(?m)^(\s*{re.escape(heading)}\b[^\n]*\n)", text)
            if heading_match:
                insert_at = heading_match.end()
                text = text[:insert_at] + sentence + " " + text[insert_at:]
                continue
        text = text.rstrip() + "\n" + sentence + "\n"
    return text


def _restore_exact_citation_forms(candidate_text: str, sections: list[dict[str, Any]]) -> str:
    text = str(candidate_text or "")
    for section in sections:
        for citation in section.get("citations") or []:
            exact = str(citation or "").strip()
            if not exact or exact in text:
                continue
            parenthetical = re.match(
                r"^\((?P<author>[A-Z][A-Za-z' -]+(?:\s+et\s+al\.)?),\s*(?P<year>(?:19|20)\d{2}[a-z]?)(?P<locator>,\s*p{1,2}\.?\s*\d+)?\)$",
                exact,
            )
            if parenthetical:
                author = parenthetical.group("author")
                year = parenthetical.group("year")
                narrative_pattern = re.compile(rf"\b{re.escape(author)}\s+\({re.escape(year)}\)")
                text = narrative_pattern.sub(exact, text, count=1)
                continue

            narrative = re.match(
                r"^(?P<author>[A-Z][A-Za-z' -]+(?:\s+et\s+al\.)?)\s+\((?P<year>(?:19|20)\d{2}[a-z]?)\)$",
                exact,
            )
            if narrative:
                author = narrative.group("author")
                year = narrative.group("year")
                parenthetical_pattern = re.compile(rf"\({re.escape(author)},\s*{re.escape(year)}\)")
                text = parenthetical_pattern.sub(exact, text, count=1)
    return text


def _academic_all_section_filter_failures(sections: list[dict[str, Any]], candidate_text: str) -> list[str]:
    failures: list[str] = []
    text = str(candidate_text or "")
    last_heading_index = -1
    for section in sections:
        section_id = str(section.get("section_id") or "")
        heading = str(section.get("heading") or "").strip()
        if heading:
            match = re.search(rf"(?m)^\s*{re.escape(heading)}\b", text)
            if not match:
                failures.append(f"{section_id}:heading_lost:{heading}")
            elif match.start() < last_heading_index:
                failures.append(f"{section_id}:heading_order_changed:{heading}")
            else:
                last_heading_index = match.start()
        for citation in section.get("citations") or []:
            if citation not in text:
                failures.append(f"{section_id}:citation_lost:{citation}")
        for protected in _protected_texts_for_scope(str(section.get("text") or "")):
            if protected not in text:
                failures.append(f"{section_id}:protected_span_lost:{protected}")
    return failures[:30]


def _generate_academic_all_section_candidates(
    *,
    original_text: str,
    scan_report: dict[str, Any],
    gateway: LLMGateway,
    model: str | None,
    deadline: float | None,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    if not _all_section_compact_allowed(original_text, scan_report):
        return []
    if deadline is not None and time.time() + timeout_seconds + 2.0 >= deadline:
        return []
    sections = _academic_assignment_sections(original_text, scan_report)
    if len(sections) < 2:
        return []
    candidate_count = max(1, min(3, int(os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_ALL_SECTION_CANDIDATES", "2") or 2)))
    prompt = _build_academic_all_section_prompt(sections=sections, candidate_count=candidate_count)
    response = gateway.chat(
        prompt,
        system=(
            "You rewrite academic assignment sections compactly while preserving exact citations, "
            "direct quotes, headings, protected numbers, and required terms."
        ),
        max_tokens=7200,
        temperature=0.78,
        top_p=0.94,
        presence_penalty=0.30 if _supports_openai_penalties(model) else None,
        frequency_penalty=0.42 if _supports_openai_penalties(model) else None,
        repetition_penalty=1.10 if _supports_repetition_penalty(model) else None,
        seed=8817,
    )
    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(_parse_academic_all_section_variants(response.content), start=1):
        candidate_text = _normalize_academic_all_section_candidate(str(variant.get("text") or ""), sections)
        filter_failures = _academic_all_section_filter_failures(sections, candidate_text)
        rows.append({
            "strategy": "academic_all_section_compact_reconstruction",
            "strategy_kind": "academic_all_section_compact_reconstruction",
            "candidate_number": index,
            "candidate_response": variant,
            "text": candidate_text,
            "local_filter_passed": not filter_failures,
            "local_filter_failures": filter_failures,
            "academic_section_targets": [
                {
                    "section_id": section.get("section_id"),
                    "heading": section.get("heading"),
                    "word_count": section.get("word_count"),
                    "required_citations": section.get("citations") or [],
                }
                for section in sections
            ],
            "applied_section_count": len(sections),
            "section_patch_count": len(sections),
        })
    return rows


def _generate_academic_section_candidates(
    *,
    original_text: str,
    scan_report: dict[str, Any],
    gateway: LLMGateway,
    model: str | None,
    deadline: float | None,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    if os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_SECTION_RESOLVER", "1").lower() in {"0", "false", "no"}:
        return []
    if deadline is not None and time.time() + timeout_seconds + 2.0 >= deadline:
        return []
    max_sections = int(os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_SECTION_MAX_SECTIONS", "2") or 2)
    sections = _academic_section_targets(original_text, scan_report, limit=max_sections)
    if not sections:
        return []
    candidate_count = max(1, min(3, int(os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_SECTION_CANDIDATES", "1") or 1)))
    prompt = _build_academic_section_prompt(sections=sections, candidate_count=candidate_count)
    response = gateway.chat(
        prompt,
        system=(
            "You rewrite citation-heavy academic sections while preserving exact citations, "
            "numbers, headings, and source attribution."
        ),
        max_tokens=5200,
        temperature=0.68,
        top_p=0.92,
        presence_penalty=0.22 if _supports_openai_penalties(model) else None,
        frequency_penalty=0.32 if _supports_openai_penalties(model) else None,
        repetition_penalty=1.08 if _supports_repetition_penalty(model) else None,
        seed=4103,
    )
    payload = _json_from_response(response.content)
    parsed_candidates = _candidate_section_map(payload) or _parse_academic_section_delimited(response.content)
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(parsed_candidates, start=1):
        raw_sections = candidate.get("sections")
        section_patches = raw_sections if isinstance(raw_sections, list) else []
        filter_failures = _academic_section_filter_failures(sections, section_patches)
        candidate_text, applied_sections = _compose_academic_sections(original_text, sections, section_patches)
        if not any(row.get("applied") for row in applied_sections):
            filter_failures.append("no_section_patch_applied")
        candidates.append({
            "strategy": "academic_cited_section_density_resolver",
            "strategy_kind": "academic_cited_section_density_resolver",
            "candidate_number": index,
            "candidate_response": {
                "candidate_id": candidate.get("candidate_id") or f"academic_section_variant_{index}",
                "sections": section_patches,
            },
            "text": candidate_text,
            "local_filter_passed": not filter_failures,
            "local_filter_failures": filter_failures,
            "academic_section_targets": [
                {
                    "section_id": section.get("section_id"),
                    "heading": section.get("heading"),
                    "word_count": section.get("word_count"),
                    "required_citations": section.get("citations") or [],
                }
                for section in sections
            ],
            "applied_section_count": sum(1 for row in applied_sections if row.get("applied")),
            "section_patch_count": len(applied_sections),
            "section_patches": applied_sections,
        })
    return candidates
