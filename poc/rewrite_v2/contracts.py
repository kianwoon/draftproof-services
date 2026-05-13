"""Document contract extraction for rewrite V2.

The contract separates exact anchors from normalized and contextual anchors so
generation layers and validation do not treat all preserved text the same way.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from rewrite.guards import _extract_named_entities, detect_protected_spans


class AnchorSeverity(str, Enum):
    HARD_EXACT = "hard_exact"
    HARD_NORMALIZED = "hard_normalized"
    SOFT_REQUIRED = "soft_required"
    TITLE_CONTEXT = "title_context"


@dataclass(frozen=True)
class ContractAnchor:
    text: str
    severity: AnchorSeverity
    kind: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    section_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


@dataclass(frozen=True)
class RewriteContract:
    content_mode: str
    anchors: tuple[ContractAnchor, ...]

    def anchors_by_severity(self, severity: AnchorSeverity) -> list[ContractAnchor]:
        return [anchor for anchor in self.anchors if anchor.severity == severity]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_mode": self.content_mode,
            "anchor_count": len(self.anchors),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
        }


_GENERIC_SINGLE_ENTITY_WORDS = {
    "certificate",
    "centre",
    "combination",
    "communicates",
    "down",
    "educational",
    "environment",
    "evaluation",
    "every",
    "haircut",
    "inclusive",
    "learning",
    "part",
    "points",
    "salon",
    "statistics",
    "structures",
    "team",
    "universal",
    "you",
}


def normalized_anchor_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def anchor_present(anchor: ContractAnchor, text: str) -> bool:
    source = str(text or "")
    if anchor.severity == AnchorSeverity.HARD_EXACT:
        if re.fullmatch(r"\d+(?:\.\d+)?%?", anchor.text):
            return bool(re.search(rf"\b{re.escape(anchor.text)}\b", source))
        return anchor.text in source
    values = (anchor.text, *anchor.aliases)
    source_key = normalized_anchor_key(source)
    return any(normalized_anchor_key(value) in source_key for value in values if normalized_anchor_key(value))


def _citation_markers(text: str) -> list[str]:
    markers: list[str] = []
    source = str(text or "")
    patterns = [
        r"\([A-Z][^)]*(?:19|20)\d{2}[a-z]?(?:\s*;\s*[A-Z][^)]*(?:19|20)\d{2}[a-z]?)+\)",
        r"\([A-Z][A-Za-z' -]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}[a-z]?(?:,\s*p{1,2}\.?\s*\d+)?\)",
        r"\b[A-Z][A-Za-z' -]+(?:,\s*[A-Z][A-Za-z' -]+)*(?:,?\s+and\s+[A-Z][A-Za-z' -]+)\s+\((?:19|20)\d{2}[a-z]?\)",
        r"\b[A-Z][A-Za-z' -]+(?:\s+et\s+al\.)?\s+\((?:19|20)\d{2}[a-z]?\)",
        r"\[(?:\d+(?:\s*[,\-]\s*\d+)*)\]",
        r"\b(?:doi|DOI):\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            value = _normalize_citation_marker(match.group(0).strip())
            if re.match(r"(?i)^and\s+", value):
                continue
            if value and value not in markers:
                markers.append(value)
    return markers


def _normalize_citation_marker(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^(?:According\s+to|As|In|Based\s+on)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:research|a\s+report|the\s+report|the\s+study)\s+from\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^the\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" ,.;:")


def _split_heading_and_body(section_text: str) -> tuple[str, str]:
    lines = str(section_text or "").splitlines()
    if len(lines) <= 1:
        return "", str(section_text or "")
    first = lines[0].strip()
    if first and len(first) <= 120 and not re.search(r"[.!?]\s*$", first):
        return first, "\n".join(lines[1:]).strip()
    return "", str(section_text or "")


def _aliases_for_term(value: str) -> tuple[str, ...]:
    text = str(value or "").strip()
    aliases: list[str] = []
    stripped_lead = re.sub(
        r"^(?:according\s+to|as|at|by|for|from|in|on|to|with)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if stripped_lead and stripped_lead != text:
        aliases.append(stripped_lead)
    if re.search(r"\bTik\s+Tok\b", text, flags=re.IGNORECASE):
        aliases.append(re.sub(r"\bTik\s+Tok\b", "TikTok", text, flags=re.IGNORECASE))
    if re.search(r"\bTikTok\b", text, flags=re.IGNORECASE):
        aliases.append(re.sub(r"\bTikTok\b", "Tik Tok", text, flags=re.IGNORECASE))
    return tuple(alias for alias in aliases if alias and alias != text)


def _clean_entity_anchor(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(
        r"^(?:according\s+to|as|at|by|for|from|in|on|to|with)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


def _single_entity_allowed(entity: str, body: str, multiword_entities: list[str]) -> bool:
    value = str(entity or "").strip()
    if not value or " " in value:
        return False
    lowered = value.casefold()
    if lowered in _GENERIC_SINGLE_ENTITY_WORDS:
        return False
    if any(re.search(rf"\b{re.escape(value)}\b", item) for item in multiword_entities):
        return False
    return len(re.findall(rf"\b{re.escape(value)}\b", str(body or ""))) >= 2


def _generic_entity_anchor(entity: str) -> bool:
    words = [word.casefold() for word in re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", str(entity or ""))]
    if not words:
        return True
    content_words = [word for word in words if word != "the"]
    return bool(content_words) and all(word in _GENERIC_SINGLE_ENTITY_WORDS for word in content_words)


def _heading_terms(heading: str) -> list[str]:
    terms: list[str] = []
    for chunk in re.split(r"\b(?:and|to|of|for|in|with)\b|[:;,-]", str(heading or "")):
        words = re.findall(r"\b[A-Z][A-Za-z'-]*\b", chunk)
        if len(words) < 2:
            continue
        value = " ".join(words).strip()
        if value and value not in terms:
            terms.append(value)
    for entity in sorted(_extract_named_entities(heading), key=lambda item: (-len(item), item)):
        if entity and entity not in terms:
            terms.append(entity)
    return terms


def _add_anchor(
    anchors: list[ContractAnchor],
    seen: set[tuple[str, str, str | None]],
    *,
    text: str,
    severity: AnchorSeverity,
    kind: str,
    section_id: str | None,
    aliases: tuple[str, ...] = (),
) -> None:
    value = str(text or "").strip()
    if not value:
        return
    key = (value, severity.value, section_id)
    if key in seen:
        return
    seen.add(key)
    anchors.append(ContractAnchor(value, severity, kind, aliases=aliases, section_id=section_id))


def build_rewrite_contract(
    text: str,
    *,
    content_mode: str = "generic_expository",
    sections: list[dict[str, Any]] | None = None,
) -> RewriteContract:
    anchors: list[ContractAnchor] = []
    seen: set[tuple[str, str, str | None]] = set()
    scoped_sections = sections or [{"section_id": None, "heading": "", "text": str(text or "")}]
    for section in scoped_sections:
        section_id = str(section.get("section_id") or "") or None
        section_text = str(section.get("text") or "")
        heading, body = _split_heading_and_body(section_text)
        if heading:
            for entity in _heading_terms(heading):
                _add_anchor(
                    anchors,
                    seen,
                    text=entity,
                    severity=AnchorSeverity.TITLE_CONTEXT,
                    kind="heading_term",
                    section_id=section_id,
                    aliases=_aliases_for_term(entity),
                )
        for citation in _citation_markers(section_text):
            _add_anchor(anchors, seen, text=citation, severity=AnchorSeverity.HARD_EXACT, kind="citation", section_id=section_id)
        for span in detect_protected_spans(section_text):
            value = str(span.text or "").strip()
            if span.reason in {"direct_quote", "numeric"}:
                _add_anchor(anchors, seen, text=value, severity=AnchorSeverity.HARD_EXACT, kind=span.reason, section_id=section_id)
        raw_entities = sorted(_extract_named_entities(body), key=lambda item: (-len(item), item))
        cleaned_entities = [_clean_entity_anchor(entity) for entity in raw_entities]
        multiword_entities = [
            entity for entity in cleaned_entities
            if len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", entity)) >= 2
        ]
        for entity in cleaned_entities:
            word_count = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", entity))
            if word_count < 2 and not _single_entity_allowed(entity, body, multiword_entities):
                continue
            if word_count >= 2 and _generic_entity_anchor(entity):
                continue
            _add_anchor(
                anchors,
                seen,
                text=entity,
                severity=AnchorSeverity.SOFT_REQUIRED,
                kind="academic_term",
                section_id=section_id,
                aliases=_aliases_for_term(entity),
            )
    return RewriteContract(content_mode=content_mode, anchors=tuple(anchors))
