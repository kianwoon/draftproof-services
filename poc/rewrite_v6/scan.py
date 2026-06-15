from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from .paragraph_normalizer import normalize_paragraph_blocks
from .text import Paragraph, Sentence, split_paragraphs, split_sentences, word_count



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
    text = normalize_paragraph_blocks(text)
    return _scan_from_paragraphs(text, split_paragraphs(text))


def scan_text_preserve_blocks(text: str) -> Scan:
    """Scan rewritten output without creating virtual paragraphs from oversized blocks."""
    visible = str(text or "")
    return _scan_from_paragraphs(visible, split_paragraphs(visible))


def scan_text_with_report(text: str, report: dict[str, Any] | None) -> Scan:
    """Build a V6 scan from full submitted text plus scanner report identity metadata."""
    base = scan_text(text)
    if not isinstance(report, dict):
        return base
    paragraphs = _report_aligned_paragraphs(base.source_text, report)
    if not paragraphs:
        return base
    scan = _scan_from_paragraphs(base.source_text, paragraphs)
    report_findings = _report_findings(report, paragraphs)
    if not report_findings:
        return scan
    findings = _merge_findings(scan.findings, report_findings)
    return Scan(
        source_text=scan.source_text,
        paragraphs=scan.paragraphs,
        findings=findings,
        scores={
            **scan.scores,
            "finding_count": float(len(findings)),
            "report_finding_count": float(len(report_findings)),
        },
    )


def _scan_from_paragraphs(text: str, paragraphs: list[Paragraph]) -> Scan:
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph.sentences]
    risks = [_risk(sentence) for sentence in sentences]
    threshold = max(12.0, mean(risks) if risks else 0.0)
    findings: list[Finding] = []
    seen_findings: set[tuple[str, str]] = set()
    for sentence, risk in zip(sentences, risks, strict=False):
        tags = _dedupe([*_tags(sentence), *(["paragraph_rhythm"] if risk >= threshold else [])])
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


def _report_aligned_paragraphs(text: str, report: dict[str, Any]) -> list[Paragraph]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", normalize_paragraph_blocks(text)) if block.strip()]
    report_rows = _report_paragraph_rows(report)
    if not blocks:
        return []
    if report_rows and len(report_rows) != len(blocks):
        return []
    sentence_id_map = _report_sentence_id_map(report)
    paragraphs: list[Paragraph] = []
    for index, block in enumerate(blocks):
        paragraph_id = _report_paragraph_id(report_rows, index)
        sentences = _align_sentence_ids(
            split_sentences(block, paragraph_id=paragraph_id),
            paragraph_id=paragraph_id,
            sentence_id_map=sentence_id_map,
        )
        paragraphs.append(Paragraph(id=paragraph_id, index=index, text=block, sentences=sentences))
    return paragraphs


def _report_paragraph_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    document = (report.get("scan_intelligence") or {}).get("document") if isinstance(report.get("scan_intelligence"), dict) else {}
    rows = document.get("paragraphs") if isinstance(document, dict) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _report_paragraph_id(rows: list[dict[str, Any]], index: int) -> str:
    if 0 <= index < len(rows):
        paragraph_id = str(rows[index].get("paragraph_id") or "").strip()
        if paragraph_id:
            return paragraph_id
    return f"p{index + 1:03d}"


def _report_sentence_id_map(report: dict[str, Any]) -> dict[tuple[str, str], str]:
    sentence_map = report.get("sentence_map") if isinstance(report.get("sentence_map"), dict) else {}
    rows: dict[tuple[str, str], str] = {}
    for sentence_id, item in sentence_map.items():
        if not isinstance(item, dict):
            continue
        paragraph_id = str(item.get("paragraph_id") or "").strip()
        text = _text_key(item.get("text"))
        if paragraph_id and text:
            rows[(paragraph_id, text)] = str(sentence_id)
    return rows


def _align_sentence_ids(
    sentences: list[Sentence],
    *,
    paragraph_id: str,
    sentence_id_map: dict[tuple[str, str], str],
) -> list[Sentence]:
    used: set[str] = set()
    aligned: list[Sentence] = []
    for sentence in sentences:
        report_id = sentence_id_map.get((paragraph_id, _text_key(sentence.text)))
        sentence_id = report_id if report_id and report_id not in used else sentence.id
        used.add(sentence_id)
        aligned.append(
            Sentence(
                id=sentence_id,
                paragraph_id=paragraph_id,
                index=sentence.index,
                text=sentence.text,
                word_count=sentence.word_count,
            )
        )
    return aligned


def _report_findings(report: dict[str, Any], paragraphs: list[Paragraph]) -> list[Finding]:
    sentence_lookup = {
        sentence.id: sentence
        for paragraph in paragraphs
        for sentence in paragraph.sentences
    }
    protected_by_finding = _protected_spans_by_finding(report)
    grouped: dict[str, dict[str, Any]] = {}
    for segment in report.get("highlight_segments") or []:
        if not isinstance(segment, dict):
            continue
        sentence_id = str(segment.get("sentence_id") or segment.get("segment_id") or "").strip()
        sentence = sentence_lookup.get(sentence_id)
        if sentence is None:
            continue
        signals = [signal for signal in segment.get("signals") or [] if isinstance(signal, dict)]
        if not signals:
            primary = segment.get("primary_signal")
            signals = [primary] if isinstance(primary, dict) else []
        if not signals:
            continue
        entry = grouped.setdefault(sentence_id, {
            "sentence": sentence,
            "tags": [],
            "severity": 0.0,
            "finding_ids": [],
            "actionability": [],
            "predictability": {},
        })
        # Token-level predictability detail lives in segment["predictability"] (not in the signal):
        # which token spans the graded detector found predictable, and the protected spans the
        # writer must keep. Capturing it here makes the finding actionable downstream.
        predictability = segment.get("predictability")
        if isinstance(predictability, dict) and not entry["predictability"]:
            entry["predictability"] = predictability
        for signal in signals:
            entry["tags"].extend(_report_signal_tags(signal))
            entry["severity"] = max(float(entry["severity"]), _report_signal_severity(signal))
            finding_id = str(signal.get("finding_id") or "").strip()
            if finding_id:
                entry["finding_ids"].append(finding_id)
            actionability = str(signal.get("actionability") or "").strip()
            if actionability:
                entry["actionability"].append(actionability)
    findings: list[Finding] = []
    for sentence_id, entry in grouped.items():
        sentence = entry["sentence"]
        tags = _dedupe([tag for tag in entry["tags"] if tag])
        if not tags:
            continue
        finding_ids = _dedupe(entry["finding_ids"])[:8]
        actionability = _dedupe(entry["actionability"])
        # A segment whose ONLY actionability is review-level (e.g. high/medium predictability the
        # detector marks "review_only") must not be the sole reason a paragraph is rewritten --
        # predictability is un-mitigable by rewrite. It still rides along as annotation when a
        # mitigable signal is present on the same sentence; alone, it does not drive targeting.
        mitigable = [a for a in actionability if a and a != "review_only"]
        if actionability and not mitigable:
            continue
        evidence: dict[str, Any] = {
            "text": sentence.text,
            "source": "scan_report",
            "finding_ids": finding_ids,
            "actionability": actionability[:6],
            "word_count": word_count(sentence.text),
        }
        detail = _predictability_detail(entry["predictability"], finding_ids, protected_by_finding)
        if detail:
            evidence["predictability"] = detail
        findings.append(
            Finding(
                sentence_id=sentence_id,
                paragraph_id=sentence.paragraph_id,
                tags=tags,
                severity=round(max(12.0, float(entry["severity"])), 3),
                evidence=evidence,
            )
        )
    return findings


def _predictability_detail(
    predictability: dict[str, Any],
    finding_ids: list[str],
    protected_by_finding: dict[str, list[str]],
) -> dict[str, Any]:
    """Token-level predictability targets for a finding: which spans to break, which to keep.

    `predictable_token_spans` / `top_predicted_tokens` come from the detect report's per-sentence
    predictability block; `protected_spans` from the matching rewrite_edit_brief. This is the data
    the planner/writer (Stage 2) need to know WHICH tokens to change rather than guessing.
    """
    spans = [str(s) for s in (predictability.get("predictable_token_spans") or []) if str(s).strip()]
    problem_tokens = [
        str(token.get("token"))
        for token in (predictability.get("top_predicted_tokens") or [])
        if isinstance(token, dict) and str(token.get("token") or "").strip()
    ]
    protected: list[str] = []
    for finding_id in finding_ids:
        protected.extend(protected_by_finding.get(finding_id, []))
    detail: dict[str, Any] = {}
    if spans:
        detail["predictable_token_spans"] = spans[:12]
    if problem_tokens:
        detail["problem_tokens"] = _dedupe(problem_tokens)[:16]
    if protected:
        detail["protected_spans"] = _dedupe(protected)[:12]
    for metric in ("score", "risk_label", "top10_ratio", "avg_surprisal"):
        if predictability.get(metric) is not None:
            detail[metric] = predictability.get(metric)
    return detail


def _protected_spans_by_finding(report: dict[str, Any]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for brief in report.get("rewrite_edit_briefs") or []:
        if not isinstance(brief, dict):
            continue
        finding_id = str(brief.get("finding_id") or "").strip()
        spans = [str(s) for s in (brief.get("protected_spans") or []) if str(s).strip()]
        if finding_id and spans:
            rows.setdefault(finding_id, []).extend(spans)
    return rows


def _report_signal_tags(signal: dict[str, Any]) -> list[str]:
    values = [
        signal.get("title"),
        signal.get("scanner"),
        signal.get("category"),
        signal.get("key"),
    ]
    tags: list[str] = []
    for value in values:
        tag = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
        if tag:
            tags.append(tag)
    return tags


def _report_signal_severity(signal: dict[str, Any]) -> float:
    for key in ("score", "adjusted_risk", "severity"):
        try:
            value = float(signal.get(key))
        except (TypeError, ValueError):
            continue
        return value * 100.0 if 0 < value <= 1 else value
    tier = str(signal.get("tier") or "").casefold()
    return {"critical": 85.0, "high": 70.0, "medium": 45.0, "low": 18.0}.get(tier, 20.0)


def _merge_findings(base: list[Finding], report_findings: list[Finding]) -> list[Finding]:
    merged: dict[str, Finding] = {finding.sentence_id: finding for finding in base}
    for finding in report_findings:
        current = merged.get(finding.sentence_id)
        if current is None:
            merged[finding.sentence_id] = finding
            continue
        report_evidence = finding.evidence if isinstance(finding.evidence, dict) else {}
        merged_evidence = {**current.evidence, "report_evidence": report_evidence}
        if report_evidence.get("predictability"):
            merged_evidence["predictability"] = report_evidence["predictability"]
        merged[finding.sentence_id] = Finding(
            sentence_id=current.sentence_id,
            paragraph_id=current.paragraph_id,
            tags=_dedupe([*current.tags, *finding.tags]),
            severity=max(current.severity, finding.severity),
            evidence=merged_evidence,
        )
    return list(merged.values())


def _text_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


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
    if _abstract_risk_pressure(sentence) >= 12.0:
        tags.append("abstract_density")
    if sentence.word_count >= 22:
        tags.append("sentence_overload")
    if _citation_anchor(text):
        tags.append("citation_anchor")
    if _paraphrase_smoothing(sentence):
        tags.append("paraphrase_smoothing")
    return tags


def _risk(sentence: Sentence) -> float:
    return (
        _list_pressure(sentence.text) * 36.0
        + _abstract_risk_pressure(sentence)
        + (6.0 if _citation_anchor(sentence.text) else 0.0)
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



def _citation_anchor(text: str) -> bool:
    # Structural citation FORM only (author + year, or parenthetical year) -- agnostic to the
    # reporting verb / phrasing. No hardcoded verb vocabulary.
    return bool(re.search(r"\b\w+\s+et al\.\s*\(\d{4}\)", text)) or _parenthetical_citation(text)



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
