from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from .text import Paragraph, Sentence, split_paragraphs


@dataclass(frozen=True)
class Finding:
    sentence_id: str
    paragraph_id: str
    tags: list[str]
    severity: float
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Scan:
    source_text: str
    paragraphs: list[Paragraph]
    findings: list[Finding]
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
            "findings": [finding.to_dict() for finding in self.findings],
            "scores": dict(self.scores),
        }


def scan_text(text: str) -> Scan:
    paragraphs = split_paragraphs(text)
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph.sentences]
    risks = [_risk(sentence) for sentence in sentences]
    threshold = max(12.0, mean(risks) if risks else 0.0)
    findings: list[Finding] = []
    seen_findings: set[tuple[str, str]] = set()
    for sentence, risk in zip(sentences, risks, strict=False):
        tags = _tags(sentence)
        if risk >= threshold:
            tags.append("paragraph_rhythm")
        tags = _dedupe(tags)
        if tags and risk >= 12.0:
            for tag in tags:
                seen_findings.add((sentence.id, tag))
            findings.append(
                Finding(
                    sentence_id=sentence.id,
                    paragraph_id=sentence.paragraph_id,
                    tags=tags,
                    severity=round(risk, 3),
                    evidence={
                        "text": sentence.text,
                        "word_count": sentence.word_count,
                        "list_pressure": round(_list_pressure(sentence.text), 3),
                        "abstract_pressure": round(_abstract_pressure(sentence.text), 3),
                    },
                )
            )
    for finding in _repeated_frame_findings(paragraphs):
        key = (finding.sentence_id, "repeated_sentence_frame")
        if key in seen_findings:
            continue
        findings.append(finding)
        seen_findings.add(key)
    return Scan(
        source_text=text,
        paragraphs=paragraphs,
        findings=findings,
        scores={
            "finding_count": float(len(findings)),
            "paragraph_count": float(len(paragraphs)),
            "sentence_count": float(len(sentences)),
            "mean_sentence_shape_risk": round(mean(risks), 3) if risks else 0.0,
        },
    )


def _repeated_frame_findings(paragraphs: list[Paragraph]) -> list[Finding]:
    findings: list[Finding] = []
    for paragraph in paragraphs:
        frames: dict[str, list[Sentence]] = {}
        for sentence in paragraph.sentences:
            frame = _sentence_frame(sentence.text)
            if frame:
                frames.setdefault(frame, []).append(sentence)
        for frame, matches in frames.items():
            if len(matches) < 3:
                continue
            for sentence in matches:
                findings.append(
                    Finding(
                        sentence_id=sentence.id,
                        paragraph_id=sentence.paragraph_id,
                        tags=["repeated_sentence_frame", "paragraph_rhythm"],
                        severity=22.0 + len(matches),
                        evidence={
                            "text": sentence.text,
                            "word_count": sentence.word_count,
                            "repeated_frame": frame,
                            "frame_count": len(matches),
                        },
                    )
                )
    return findings


def _sentence_frame(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or "").casefold())
    if len(words) < 4:
        return ""
    if words[0] in {"the", "a", "an"} and len(words) >= 4:
        return " ".join(words[:4])
    return " ".join(words[:3])


def select_target_paragraph(
    scan: Scan,
    excluded_ids: set[str] | None = None,
    priority_ids: set[str] | None = None,
) -> Paragraph:
    if not scan.paragraphs:
        raise ValueError("no paragraphs to rewrite")
    excluded = excluded_ids or set()
    scores: dict[str, float] = {}
    for paragraph_id in priority_ids or set():
        if paragraph_id not in excluded:
            scores[paragraph_id] = scores.get(paragraph_id, 0.0) + 1000.0
    for finding in scan.findings:
        if finding.paragraph_id in excluded:
            continue
        scores[finding.paragraph_id] = scores.get(finding.paragraph_id, 0.0) + finding.severity
    if not scores:
        return next((paragraph for paragraph in scan.paragraphs if paragraph.id not in excluded), scan.paragraphs[0])
    target_id = max(scores.items(), key=lambda item: item[1])[0]
    return next((paragraph for paragraph in scan.paragraphs if paragraph.id == target_id), scan.paragraphs[0])


def findings_for_paragraph(scan: Scan, paragraph_id: str) -> list[Finding]:
    return [finding for finding in scan.findings if finding.paragraph_id == paragraph_id]


def _tags(sentence: Sentence) -> list[str]:
    text = sentence.text
    tags: list[str] = []
    if _list_pressure(text) >= 0.22:
        tags.append("packed_list")
    if _predictable_start_pressure(sentence) >= 12.0:
        tags.append("predictable_start")
    if _abstract_risk_pressure(sentence) >= 12.0:
        tags.append("abstract_density")
    if sentence.word_count >= 22:
        tags.append("sentence_overload")
    if _context_anchor_gap(sentence):
        tags.append("context_anchor_gap")
    if _author_anchor_gap(sentence):
        tags.append("author_anchor_gap")
    if _citation_anchor(text):
        tags.append("citation_anchor")
    if _broad_claim(text):
        tags.append("broad_claim")
    if _transition_stack(text):
        tags.append("transition_stack")
    if _semantic_bridge_gap(sentence):
        tags.append("semantic_bridge_gap")
    if _unsupported_claim_gap(sentence):
        tags.append("unsupported_claim_gap")
    if _paraphrase_smoothing(sentence):
        tags.append("paraphrase_smoothing")
    return tags


def _risk(sentence: Sentence) -> float:
    return (
        _list_pressure(sentence.text) * 36.0
        + _abstract_risk_pressure(sentence)
        + _predictable_start_pressure(sentence)
        + (8.0 if _context_anchor_gap(sentence) else 0.0)
        + (8.0 if _author_anchor_gap(sentence) else 0.0)
        + (6.0 if _citation_anchor(sentence.text) else 0.0)
        + (6.0 if _broad_claim(sentence.text) else 0.0)
        + (6.0 if _transition_stack(sentence.text) else 0.0)
        + (6.0 if _semantic_bridge_gap(sentence) else 0.0)
        + (8.0 if _unsupported_claim_gap(sentence) else 0.0)
        + (6.0 if _paraphrase_smoothing(sentence) else 0.0)
        + min(12.0, max(0, sentence.word_count - 18) * 0.8)
    )


def _abstract_risk_pressure(sentence: Sentence) -> float:
    pressure = _abstract_pressure(sentence.text)
    if sentence.word_count <= 6 and _list_pressure(sentence.text) == 0:
        return pressure * 8.0
    if sentence.word_count <= 12 and _list_pressure(sentence.text) == 0:
        return pressure * 15.0
    return pressure * 34.0


def _predictable_start_pressure(sentence: Sentence) -> float:
    if not _predictable_start(sentence.text):
        return 0.0
    list_pressure = _list_pressure(sentence.text)
    abstract_pressure = _abstract_pressure(sentence.text)
    if sentence.word_count <= 6 and list_pressure == 0 and re.match(
        r"^(it|this|that|they|these|those|he|she)\s+(may|might|can|could|should|would|will)\b",
        sentence.text.strip(),
        flags=re.I,
    ):
        return 6.0
    if sentence.word_count <= 5 and list_pressure == 0 and abstract_pressure <= 0.22:
        return 6.0
    if sentence.word_count <= 9 and list_pressure == 0 and abstract_pressure <= 0.25:
        return 8.0
    if sentence.word_count <= 12 and list_pressure == 0 and abstract_pressure <= 0.18:
        return 10.0
    return 18.0


def _list_pressure(text: str) -> float:
    visible = _without_parentheticals(str(text or ""))
    lowered = visible.casefold()
    comma_count = visible.count(",")
    if comma_count == 1 and _single_preposed_context_comma(visible):
        comma_count = 0
    comma_separators = comma_count if comma_count >= 2 else 0
    separators = comma_separators + visible.count(";") + max(0, lowered.count(" and ") - 1)
    return min(1.0, separators / 3.0)


def _single_preposed_context_comma(text: str) -> bool:
    return bool(
        re.match(r"^(in|during|after|before|through|at|from|to|for|as|when|while)\s+[^,]{1,80},\s+\S+", text.strip(), flags=re.I)
    )


def _abstract_pressure(text: str) -> float:
    words = [word.strip(".,:;!?()[]{}\"'").casefold() for word in str(text or "").split()]
    if not words:
        return 0.0
    signals = sum(
        1
        for word in words
        if len(word) >= 10 or word.endswith(("tion", "ment", "ity", "ness", "ance", "ence"))
    )
    return min(1.0, signals / max(5, len(words)))


def _predictable_start(text: str) -> bool:
    lowered = str(text or "").strip().casefold()
    starts = ("today", "now", "in ", "this ", "that ", "there ", "it ", "they ", "overall")
    return lowered.startswith(starts)


def _context_anchor_gap(sentence: Sentence) -> bool:
    lowered = re.sub(
        r"^(hence|next|lastly|finally|however|therefore|moreover|furthermore|overall),?\s+",
        "",
        sentence.text.strip().casefold(),
    )
    return sentence.word_count >= 8 and lowered.startswith(("this ", "that ", "it ", "they ", "these ", "those "))


def _author_anchor_gap(sentence: Sentence) -> bool:
    lowered = sentence.text.casefold()
    if re.search(r"\b(i|my|we|our)\b", lowered):
        return False
    evaluative_markers = ("important", "challenge", "concern")
    evidence_markers = ("this shows", "that shows", "shows that", "this demonstrates", "that demonstrates", "demonstrates that")
    return sentence.word_count >= 14 and (
        any(marker in lowered for marker in evaluative_markers)
        or any(marker in lowered for marker in evidence_markers)
    )


def _citation_anchor(text: str) -> bool:
    return bool(
        re.search(r"\baccording to\b|\b\w+\s+et al\.\s*\(\d{4}\)\s+(states|indicates|argues|notes|describes)\b", text, flags=re.I)
    )


def _broad_claim(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in ("always", "never", "no longer", "the real", "the main", "the most", "one of the"))


def _transition_stack(text: str) -> bool:
    lowered = text.strip().casefold()
    markers = ("however", "additionally", "therefore", "moreover", "furthermore", "in addition", "as a result")
    return sum(1 for marker in markers if marker in lowered) >= 2 or lowered.startswith(markers)


def _semantic_bridge_gap(sentence: Sentence) -> bool:
    lowered = sentence.text.casefold()
    return sentence.word_count >= 10 and any(marker in lowered for marker in ("therefore", "as a result", "this means", "this shows"))


def _unsupported_claim_gap(sentence: Sentence) -> bool:
    lowered = sentence.text.casefold()
    if _citation_anchor(sentence.text):
        return False
    return sentence.word_count >= 10 and any(marker in lowered for marker in ("should", "need to", "needs to", "important", "serious"))


def _paraphrase_smoothing(sentence: Sentence) -> bool:
    return (
        sentence.word_count >= 16
        and _list_pressure(sentence.text) == 0
        and _abstract_pressure(sentence.text) >= 0.12
        and not _citation_anchor(sentence.text)
        and not _parenthetical_citation(sentence.text)
        and _named_anchor_count(sentence.text) < 2
    )


def _named_anchor_count(text: str) -> int:
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9'-]{2,}\b", str(text or ""))
    return max(0, len(tokens) - 1)


def _without_parentheticals(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", text)


def _parenthetical_citation(text: str) -> bool:
    return bool(re.search(r"\([^)]*\d{4}[^)]*\)", str(text or "")))


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
