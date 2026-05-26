from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class Sentence:
    id: str
    paragraph_id: str
    index: int
    text: str
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Paragraph:
    id: str
    index: int
    text: str
    sentences: list[Sentence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "text": self.text,
            "sentences": [sentence.to_dict() for sentence in self.sentences],
        }


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or "")))


def split_paragraphs(text: str) -> list[Paragraph]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", str(text or "")) if block.strip()]
    paragraphs: list[Paragraph] = []
    for p_index, block in enumerate(blocks):
        paragraph_id = f"p{p_index + 1:03d}"
        sentences = split_sentences(block, paragraph_id=paragraph_id)
        paragraphs.append(Paragraph(id=paragraph_id, index=p_index, text=block, sentences=sentences))
    return paragraphs


def split_sentences(text: str, *, paragraph_id: str) -> list[Sentence]:
    protected = _protect_sentence_periods(str(text or ""))
    sentences: list[Sentence] = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+[\"'”’»)]*|$)", protected, flags=re.M):
        sentence = _restore_sentence_periods(match.group(0).strip())
        if not sentence:
            continue
        sentences.append(
            Sentence(
                id=f"{paragraph_id}_s{len(sentences) + 1:03d}",
                paragraph_id=paragraph_id,
                index=len(sentences),
                text=sentence,
                word_count=word_count(sentence),
            )
        )
    return sentences


def _protect_sentence_periods(text: str) -> str:
    protected = text
    replacements = {
        r"\bet al\.": "et al<prd>",
        r"\be\.g\.": "e<prd>g<prd>",
        r"\bi\.e\.": "i<prd>e<prd>",
        r"\bvs\.": "vs<prd>",
    }
    for pattern, replacement in replacements.items():
        protected = re.sub(pattern, replacement, protected, flags=re.I)
    return protected


def _restore_sentence_periods(text: str) -> str:
    return text.replace("<prd>", ".")


def source_terms(text: str, *, limit: int = 12) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    stop = {
        "about", "above", "after", "again", "against", "being", "between", "could",
        "every", "from", "have", "into", "more", "most", "only", "other", "over",
        "should", "still", "their", "there", "these", "those", "through", "under",
        "where", "which", "while", "would",
        "than", "that", "this", "they", "with", "many", "because", "however", "therefore",
        "does",
    }
    for token in re.findall(r"\b(?:\d+(?:\.\d+)?|[A-Z][A-Z0-9]{1,}|[A-Za-z][A-Za-z'-]{3,})\b", str(text or "")):
        key = token.strip("'").casefold()
        if key in stop or key in seen:
            continue
        terms.append(token.strip("'"))
        seen.add(key)
        if len(terms) >= limit:
            break
    return terms
