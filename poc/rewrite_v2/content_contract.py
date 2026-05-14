"""Mode-specific candidate shape contracts for rewrite V2."""

from __future__ import annotations

import re


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text or "")))


def _non_heading_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in str(text or "").splitlines()
        if line.strip()
        and not re.match(r"^(?:Question|Section|Part|Task|Prompt|Answer)\s+\d+[A-Za-z]?\s*:?\s*$", line.strip(), flags=re.I)
    ]


def _quote_spans(text: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"[\"“][^\"”]{8,}[\"”]", str(text or ""))]


def _academic_headings(text: str) -> list[str]:
    return [
        match.group(1).strip().rstrip(":")
        for match in re.finditer(
            r"(?m)^\s*((?:Question|Section|Part|Task|Prompt|Answer)\s+\d+[A-Za-z]?)\s*:?\s*$",
            str(text or ""),
            flags=re.I,
        )
    ]


def _citation_markers(text: str) -> list[str]:
    markers: list[str] = []
    patterns = [
        r"\([A-Z][A-Za-z' -]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}[a-z]?(?:,\s*p{1,2}\.?\s*\d+)?\)",
        r"\b[A-Z][A-Za-z' -]+(?:,\s*[A-Z][A-Za-z' -]+)*(?:,?\s+and\s+[A-Z][A-Za-z' -]+)\s+\((?:19|20)\d{2}[a-z]?\)",
        r"\b[A-Z][A-Za-z' -]+(?:\s+et\s+al\.)?\s+\((?:19|20)\d{2}[a-z]?\)",
        r"\[(?:\d+(?:\s*[,\-]\s*\d+)*)\]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, str(text or ""), flags=re.I):
            marker = match.group(0).strip()
            if marker and marker not in markers:
                markers.append(marker)
    return markers


def _bullet_like_lines(text: str) -> list[str]:
    return [
        line for line in _non_heading_lines(text)
        if re.match(r"^(?:\d+[.)]|[-*•])\s+", line)
        or re.match(r"^\d+\.\s*\*{1,2}[^*]{2,80}\*{1,2}\s*:", line)
    ]


def _markdown_label_lines(text: str) -> list[str]:
    return [
        line for line in _non_heading_lines(text)
        if re.match(r"^(?:\d+[.)]\s*)?\*{1,2}[^*]{2,80}\*{1,2}\s*:", line)
    ]


def _academic_shape_failures(original_text: str, candidate_text: str) -> list[str]:
    failures: list[str] = []
    source = str(original_text or "")
    candidate = str(candidate_text or "")
    original_words = _word_count(source)
    candidate_words = _word_count(candidate)
    bullet_lines = _bullet_like_lines(candidate)
    non_heading = _non_heading_lines(candidate)
    markdown_labels = _markdown_label_lines(candidate)
    source_headings = _academic_headings(source)
    candidate_headings = _academic_headings(candidate)
    if source_headings:
        last_index = -1
        for heading in source_headings:
            match = re.search(rf"(?m)^\s*{re.escape(heading)}\s*:?\s*$", candidate, flags=re.I)
            if not match:
                failures.append(f"content_contract:academic_heading_lost:{heading}")
                continue
            if match.start() < last_index:
                failures.append(f"content_contract:academic_heading_order_changed:{heading}")
                continue
            last_index = match.start()
        if len(candidate_headings) < len(source_headings):
            failures.append("content_contract:academic_heading_count_reduced")
    for marker in _citation_markers(source):
        if marker not in candidate:
            failures.append(f"content_contract:academic_citation_lost:{marker}")
            break
    if bullet_lines and len(bullet_lines) / max(1, len(non_heading)) >= 0.28:
        failures.append("content_contract:academic_outline_numbered_or_bulleted")
    if len(markdown_labels) >= 2:
        failures.append("content_contract:academic_markdown_labelled_subpoints")
    if re.search(r"\*\*[^*\n]{2,80}\*\*", candidate) or re.search(r"(?m)^\s*\*[^*\n]{2,80}\*\s*:", candidate):
        failures.append("content_contract:academic_markdown_emphasis")
    if re.search(r"[.!?][ \t]+(Question|Section|Part|Task|Prompt|Answer)\s+\d+[A-Za-z]?\s*(?:\n|$)", candidate):
        failures.append("content_contract:academic_heading_merged_into_paragraph")
    if re.search(r"\b[a-z]{3,}\.[a-z]{3,}\b", candidate):
        failures.append("content_contract:malformed_sentence_splice_artifact")
    if original_words >= 450 and candidate_words < max(260, int(original_words * 0.55)):
        failures.append("content_contract:academic_over_compressed")
    return failures


def _structured_shape_failures(original_text: str, candidate_text: str) -> list[str]:
    failures: list[str] = []
    original_bullets = len(_bullet_like_lines(original_text))
    candidate_bullets = len(_bullet_like_lines(candidate_text))
    if original_bullets >= 3 and candidate_bullets < max(2, int(original_bullets * 0.6)):
        failures.append("content_contract:structured_list_flattened")
    if "|" in str(original_text or "") and "|" not in str(candidate_text or ""):
        failures.append("content_contract:table_structure_lost")
    return failures


def _quote_shape_failures(original_text: str, candidate_text: str) -> list[str]:
    failures: list[str] = []
    original_quotes = _quote_spans(original_text)
    candidate = str(candidate_text or "")
    if len(original_quotes) >= 2:
        lost = [quote for quote in original_quotes if quote not in candidate]
        if lost:
            failures.append("content_contract:quote_heavy_quote_lost")
    return failures


def _technical_shape_failures(original_text: str, candidate_text: str) -> list[str]:
    failures: list[str] = []
    source = str(original_text or "")
    candidate = str(candidate_text or "")
    if "```" in source and "```" not in candidate:
        failures.append("content_contract:technical_code_fence_lost")
    for token in re.findall(r"\b(?:API|SDK|JSON|YAML|HTTP|SQL|CLI|URL)\b|`[^`]{2,80}`", source):
        if token and token not in candidate:
            failures.append(f"content_contract:technical_term_lost:{token.strip('`')}")
            break
    return failures


def _regulated_shape_failures(original_text: str, candidate_text: str) -> list[str]:
    failures: list[str] = []
    source = str(original_text or "")
    candidate = str(candidate_text or "")
    for token in re.findall(r"\b(?:shall|must|must not|required|prohibited|may not)\b", source, flags=re.I):
        if not re.search(rf"\b{re.escape(token)}\b", candidate, flags=re.I):
            failures.append(f"content_contract:regulated_obligation_lost:{token.lower()}")
            break
    return failures


def _short_text_failures(original_text: str, candidate_text: str) -> list[str]:
    original_words = max(1, _word_count(original_text))
    candidate_words = _word_count(candidate_text)
    if original_words < 40 and candidate_words > max(40, int(original_words * 3.0)):
        return ["content_contract:short_text_over_expanded"]
    if original_words < 120 and candidate_words > max(180, int(original_words * 2.2)):
        return ["content_contract:short_text_over_expanded"]
    return []


def candidate_shape_failures(
    *,
    content_mode: str,
    original_text: str,
    candidate_text: str,
) -> list[str]:
    mode = str(content_mode or "generic_expository")
    failures: list[str] = []
    if mode == "academic_cited_text":
        failures.extend(_academic_shape_failures(original_text, candidate_text))
    if mode == "structured_list_table":
        failures.extend(_structured_shape_failures(original_text, candidate_text))
    if mode == "quote_heavy":
        failures.extend(_quote_shape_failures(original_text, candidate_text))
    if mode == "technical_content":
        failures.extend(_technical_shape_failures(original_text, candidate_text))
    if mode == "regulated_policy_content":
        failures.extend(_regulated_shape_failures(original_text, candidate_text))
    if mode == "short_text":
        failures.extend(_short_text_failures(original_text, candidate_text))
    return failures[:12]
