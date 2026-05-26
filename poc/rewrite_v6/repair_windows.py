from __future__ import annotations

from dataclasses import dataclass

from .scan import Finding
from .text import Paragraph, word_count
from .write import Variant


@dataclass(frozen=True)
class RepairWindow:
    paragraph_id: str
    start_sentence_index: int
    end_sentence_index: int
    source_text: str
    source_sentence_ids: list[str]
    finding_count: int
    max_severity: float


def select_repair_window(
    paragraph: Paragraph,
    findings: list[Finding],
    *,
    max_sentences: int = 7,
    max_words: int = 190,
) -> RepairWindow | None:
    if not _needs_window(paragraph, findings):
        return None
    sentence_count = len(paragraph.sentences)
    if sentence_count <= 1:
        return None
    severity_by_index = _severity_by_index(paragraph, findings)
    if not severity_by_index:
        return None
    anchor = max(severity_by_index.items(), key=lambda item: (item[1], -item[0]))[0]
    start = end = anchor
    while True:
        candidates = []
        if start > 0:
            candidates.append((start - 1, end))
        if end + 1 < sentence_count:
            candidates.append((start, end + 1))
        viable = [
            row for row in candidates
            if row[1] - row[0] + 1 <= max_sentences
            and _window_word_count(paragraph, row[0], row[1]) <= max_words
        ]
        if not viable:
            break
        next_start, next_end = max(
            viable,
            key=lambda row: (
                _window_severity(severity_by_index, row[0], row[1]),
                _window_finding_count(severity_by_index, row[0], row[1]),
                -(row[1] - row[0]),
            ),
        )
        if (next_start, next_end) == (start, end):
            break
        if _window_severity(severity_by_index, next_start, next_end) <= _window_severity(severity_by_index, start, end):
            break
        start, end = next_start, next_end
    if start == 0 and end == sentence_count - 1:
        return None
    return _make_window(paragraph, findings, start, end)


def compose_window_rewrite(
    paragraphs: list[Paragraph],
    window: RepairWindow,
    selected: Variant | None,
) -> str:
    blocks: list[str] = []
    for paragraph in paragraphs:
        if paragraph.id != window.paragraph_id or selected is None:
            blocks.append(paragraph.text)
            continue
        before = [sentence.text for sentence in paragraph.sentences[:window.start_sentence_index]]
        after = [sentence.text for sentence in paragraph.sentences[window.end_sentence_index + 1:]]
        replacement = str(selected.text or "").strip()
        blocks.append(" ".join(part for part in [*before, replacement, *after] if part).strip())
    return "\n\n".join(blocks)


def _needs_window(paragraph: Paragraph, findings: list[Finding]) -> bool:
    if len(findings) < 6:
        return False
    if len(paragraph.sentences) >= 7:
        return True
    if word_count(paragraph.text) >= 180:
        return True
    return False


def _severity_by_index(paragraph: Paragraph, findings: list[Finding]) -> dict[int, float]:
    sentence_indexes = {sentence.id: sentence.index for sentence in paragraph.sentences}
    scores: dict[int, float] = {}
    for finding in findings:
        index = sentence_indexes.get(finding.sentence_id)
        if index is None:
            continue
        scores[index] = scores.get(index, 0.0) + max(0.0, float(finding.severity or 0.0))
    return scores


def _window_word_count(paragraph: Paragraph, start: int, end: int) -> int:
    return sum(sentence.word_count for sentence in paragraph.sentences[start:end + 1])


def _window_severity(scores: dict[int, float], start: int, end: int) -> float:
    return sum(value for index, value in scores.items() if start <= index <= end)


def _window_finding_count(scores: dict[int, float], start: int, end: int) -> int:
    return sum(1 for index in scores if start <= index <= end)


def _make_window(paragraph: Paragraph, findings: list[Finding], start: int, end: int) -> RepairWindow:
    source_sentence_ids = [sentence.id for sentence in paragraph.sentences[start:end + 1]]
    window_findings = [finding for finding in findings if finding.sentence_id in source_sentence_ids]
    return RepairWindow(
        paragraph_id=paragraph.id,
        start_sentence_index=start,
        end_sentence_index=end,
        source_text=" ".join(sentence.text for sentence in paragraph.sentences[start:end + 1]),
        source_sentence_ids=source_sentence_ids,
        finding_count=len(window_findings),
        max_severity=max((float(finding.severity or 0.0) for finding in window_findings), default=0.0),
    )
