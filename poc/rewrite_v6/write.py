from __future__ import annotations
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

from .coverage_guard import coverage_ratio, missing_required_source_terms
from .integrity_guard import candidate_integrity_blockers
from .json_io import parse_json
from .paragraph_architecture import apply_architecture_split_text, architecture_split_contract
from .plan import Plan
from .prose_quality import (
    catalogue_sentence_chain,
    fragment_trace_penalty,
    has_fragment_or_trace_sentences,
    repair_generated_prose,
    robotic_sentence_chain,
)
from .review_provenance import annotate_review_items
from .scan import scan_text
from .source_quality import source_quality_blockers
from .text import Paragraph, source_terms, split_paragraphs, word_count
from .writer_brief_prompt import build_writer_brief_prompt

_SENTENCE_EXPANSION_REVIEW_RATIO = 1.5
_SHORT_SENTENCE_CHAIN_RATIO = 0.35
_SHORT_SENTENCE_WORD_LIMIT = 10
_REPEATED_START_REVIEW_RATIO = 0.25
_REPEATED_FRAME_REVIEW_RATIO = 0.20
_TRANSITION_STACK_REVIEW_RATIO = 0.25


class ChatClient(Protocol):
    def chat(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> Any:
        ...


def build_prompt(paragraph: Paragraph, plan: Plan) -> str:
    return build_writer_brief_prompt(paragraph, plan)


@dataclass(frozen=True)
class Variant:
    id: str
    text: str
    source: str
    mode: str | None = None
    author_proxy_provenance: list[dict[str, Any]] | None = None
    author_review_items: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_variants(paragraph: Paragraph, plan: Plan, *, client: ChatClient) -> list[Variant]:
    variants, split_contract = [source_preserved_variant(paragraph)], architecture_split_contract(paragraph, plan)
    rows: list[Variant] = []
    for _ in range(2):
        try:
            rows = _request_variants(paragraph, plan, client)
        except (Exception, ValueError):
            rows = []
        if len(rows) >= _requested_variant_count():
            break
    variants.extend(replace(v, text=repair_generated_prose(apply_architecture_split_text(v.text, split_contract), paragraph.text)) for v in rows)
    return _dedupe_variants(variants)


def _request_variants(paragraph: Paragraph, plan: Plan, client: ChatClient) -> list[Variant]:
    response = client.chat(
        build_writer_brief_prompt(paragraph, plan),
        system=(
            "Return valid JSON only. Rewrite from the curated writer brief, not from hidden assumptions. "
            "Preserve submitted meaning and required terms. Write complete grammatical sentences. "
            "Follow writer_execution_plan in order and obey route_sequence_guards. "
            "Use proxy or neighbor context only when the brief says it resolves a local anchor gap. "
            "Return every requested variant id. Reject your own variant if it has fragments, repeated and-chains, "
            "keyword dumps, malformed connectors, premature assessment consequences, duplicated consequences, or generic added claims."
        ),
        temperature=0.12,
        top_p=0.75,
        max_tokens=None,
        response_format={"type": "json_object"},
        app_label=_writer_app_label(plan),
    )
    return parse_variants(parse_json(getattr(response, "raw_content", "") or response.content))


def _requested_variant_count() -> int:
    try:
        count = int(__import__("os").environ.get("DRAFTPROOF_V6_WRITER_VARIANTS", "3"))
    except ValueError:
        count = 3
    return max(1, min(3, count))


def _writer_app_label(plan: Plan) -> str:
    return "Writer"


def source_preserved_variant(paragraph: Paragraph) -> Variant:
    return Variant(id="source_preserved", text=paragraph.text, source="source_preserved")


def parse_variants(payload: Any) -> list[Variant]:
    rows = payload.get("variants") if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    variants: list[Variant] = []
    seen_texts: set[str] = set()
    for index, row in enumerate(rows if isinstance(rows, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        text_key = _normalize_variant_text(text)
        if text_key and text_key in seen_texts:
            continue
        if text:
            seen_texts.add(text_key)
            variants.append(
                Variant(
                    id=str(row.get("id") or row.get("variant_id") or f"v{index}"),
                    text=text,
                    source="llm",
                    mode=str(row.get("mode") or "") or None,
                    author_proxy_provenance=_review_rows(row.get("author_proxy_provenance"), "inferred_from_draft"),
                    author_review_items=_review_rows(row.get("author_review_items"), "needs_author_confirmation"),
                )
            )
    return variants


def _normalize_variant_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def _dedupe_variants(variants: list[Variant]) -> list[Variant]:
    rows: list[Variant] = []
    seen: set[str] = set()
    for variant in variants:
        key = _normalize_variant_text(variant.text)
        if key in seen:
            continue
        rows.append(variant)
        seen.add(key)
    return rows


def _review_rows(value: Any, default_provenance: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            rows.append(item)
        elif isinstance(item, str) and item.strip():
            rows.append({
                "item_id": f"r{index:03d}",
                "provenance": default_provenance,
                "target_text": "",
                "generated_text": item.strip(),
                "user_input_needed": "Author review required for this generated bridge or provenance note.",
                "author_task": "Verify, replace, or remove before final submission.",
            })
    return rows
def choose_variant(variants: list[Variant], paragraph: Paragraph) -> Variant | None:
    if not variants:
        return None
    source_words = max(1, word_count(paragraph.text))
    source_variant = next((variant for variant in variants if variant.source == "source_preserved"), None)
    if source_variant:
        improved = [
            variant for variant in variants
            if variant.source != "source_preserved"
            and _has_meaningful_movement(variant, source_variant, paragraph)
        ]
        if improved:
            return _annotate_selected_variant(
                max(improved, key=lambda variant: _variant_rank(variant, paragraph, source_words)),
                paragraph,
            )
        return source_variant
    return _annotate_selected_variant(max(variants, key=lambda variant: _variant_rank(variant, paragraph, source_words)), paragraph)


def _has_meaningful_movement(candidate: Variant, source_variant: Variant, paragraph: Paragraph) -> bool:
    if _hard_integrity_blockers(candidate.text):
        return False
    if source_quality_blockers(candidate.text, paragraph):
        return False
    if has_fragment_or_trace_sentences(candidate.text):
        return False
    before = scan_text(source_variant.text)
    after = scan_text(candidate.text)
    finding_drop = before.scores["finding_count"] - after.scores["finding_count"]
    risk_drop = before.scores["mean_sentence_shape_risk"] - after.scores["mean_sentence_shape_risk"]
    if word_count(candidate.text) < max(8, int(word_count(paragraph.text) * 0.35)):
        return False
    if finding_drop >= 1 and risk_drop >= -5.0 and not _severe_route_quality_penalty(candidate.text):
        return True
    if finding_drop >= 2 and risk_drop >= -2.0:
        return True
    if finding_drop >= 1 and risk_drop >= 0.0:
        return True
    if finding_drop >= 0 and risk_drop >= 10.0 and not _severe_route_quality_penalty(candidate.text):
        return True
    if before.scores["finding_count"] <= 1 and finding_drop >= 0 and risk_drop >= 5.0:
        return not _over_decomposition_review_reasons(candidate.text, paragraph)
    return False


def _annotate_selected_variant(variant: Variant, paragraph: Paragraph) -> Variant:
    annotated = annotate_review_items(variant, paragraph.text)
    review_items = list(annotated.author_review_items or [])
    review_reasons: list[str] = []
    if _polarity_violation(annotated.text, paragraph):
        review_reasons.append("polarity_or_contrast_changed")
    if missing_required_source_terms(annotated.text, paragraph):
        review_reasons.append("source_terms_missing")
    review_reasons.extend(_over_decomposition_review_reasons(annotated.text, paragraph))
    if _hard_candidate_contract_violation(annotated.text, paragraph):
        review_reasons.append("hard_contract_warning")
    if has_fragment_or_trace_sentences(annotated.text):
        review_reasons.append("sentence_quality_warning")
    review_reasons.extend(candidate_integrity_blockers(annotated.text))
    review_reasons.extend(source_quality_blockers(annotated.text, paragraph))
    if not review_reasons:
        return annotated
    review_items.append({
        "item_id": "auto_review_risk_mitigation_001",
        "provenance": "needs_author_confirmation",
        "target_text": " ".join(sorted(set(review_reasons))),
        "generated_text": "The rewrite reduced scanner risk but changed or weakened a guarded source constraint.",
        "user_input_needed": "Review the rewritten paragraph and keep, edit, or reject this mitigation before final use.",
        "author_task": "Confirm that meaning, contrast, source terms, and sentence quality still match the submitted draft.",
    })
    return replace(annotated, author_review_items=review_items)
def _variant_rank(variant: Variant, paragraph: Paragraph, source_words: int) -> tuple[float, float, float, float, bool, int]:
    scan = scan_text(variant.text)
    words = word_count(variant.text)
    quality_penalty = _mechanical_quality_penalty(variant.text, paragraph)
    virtual_findings = _virtual_quality_findings(quality_penalty)
    source_drift_penalty = _source_drift_penalty(variant, paragraph)
    compression_penalty = 2.0 if _compresses_list_repair(variant.text, paragraph) else 0.0
    extra_beat_penalty = 2.0 if _adds_extra_conclusion_beat(variant.text, paragraph) else 0.0
    final_beat_penalty = 2.0 if _replaces_final_source_beat_with_conclusion(variant.text, paragraph) else 0.0
    catalogue_penalty = 4.0 if catalogue_sentence_chain(variant.text) else 0.0
    polarity_penalty = 1.0 if _polarity_violation(variant.text, paragraph) else 0.0
    bridge_penalty = 4.0 if _unreviewed_bridge_violation(variant, paragraph) else 0.0
    contract_penalty = 4.0 if _candidate_contract_violation(variant.text, paragraph) else 0.0
    hard_integrity_penalty = 12.0 if _hard_integrity_blockers(variant.text) else 0.0
    route_quality_penalty = _route_quality_penalty(variant.text)
    mean_risk = scan.scores["mean_sentence_shape_risk"]
    return (
        -(scan.scores["finding_count"] + virtual_findings),
        -mean_risk,
        -(quality_penalty + fragment_trace_penalty(variant.text) + source_drift_penalty * 0.25 + compression_penalty + extra_beat_penalty + final_beat_penalty + catalogue_penalty + polarity_penalty + bridge_penalty + contract_penalty + hard_integrity_penalty + route_quality_penalty),
        coverage_ratio(variant.text, paragraph),
        words >= source_words * 0.9,
        -abs(words - source_words),
    )


def _virtual_quality_findings(quality_penalty: float) -> int:
    if quality_penalty >= 9.0:
        return 3
    if quality_penalty >= 6.0:
        return 2
    if quality_penalty >= 3.0:
        return 1
    return 0


def _mechanical_quality_penalty(text: str, source_paragraph: Paragraph) -> float:
    stats = _candidate_sentence_stats(text, source_paragraph)
    if stats["sentence_count"] <= 0:
        return 8.0
    penalty = 0.0
    if stats["expansion_ratio"] >= _SENTENCE_EXPANSION_REVIEW_RATIO:
        penalty += min(6.0, (stats["expansion_ratio"] - 1.0) * 3.0)
    if stats["short_ratio"] >= _SHORT_SENTENCE_CHAIN_RATIO and stats["avg_words"] <= _SHORT_SENTENCE_WORD_LIMIT:
        penalty += stats["short_ratio"] * 5.0
    if stats["repeated_first_ratio"] >= _REPEATED_START_REVIEW_RATIO:
        penalty += stats["repeated_first_ratio"] * 4.0
    if stats["repeated_frame_ratio"] >= _REPEATED_FRAME_REVIEW_RATIO:
        penalty += stats["repeated_frame_ratio"] * 5.0
    if stats["transition_start_ratio"] >= _TRANSITION_STACK_REVIEW_RATIO:
        penalty += stats["transition_start_ratio"] * 3.0
    if catalogue_sentence_chain(text):
        penalty += 9.0
    penalty += _route_quality_penalty(text)
    return round(penalty, 3)


def _route_quality_penalty(text: str) -> float:
    value = str(text or "")
    penalty = 0.0
    forced_connectors = re.findall(r"(?:^|[.!?]\s+)(?:Moreover|Thus|Consequently|Furthermore|Additionally)\b", value)
    penalty += len(forced_connectors) * 4.0
    patterns = {
        r"\bSuch\s+a\s+mix\b": 3.0,
        r"\bblend\s+of\s+[a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,2}\s+and\s+[a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,2}\s+sources\b": 5.0,
        r"\bdifficulty\s+shifts\s+toward\s+assessing\b": 5.0,
        r"\bjudging\s+information\s+quality\b": 3.0,
        r"\bassessing\s+information\s+quality\b": 3.0,
    }
    for pattern, weight in patterns.items():
        if re.search(pattern, value, flags=re.I):
            penalty += weight
    return penalty


def _severe_route_quality_penalty(text: str) -> bool:
    return _route_quality_penalty(text) >= 8.0


def _over_decomposition_review_reasons(text: str, source_paragraph: Paragraph) -> list[str]:
    stats = _candidate_sentence_stats(text, source_paragraph)
    if stats["sentence_count"] <= 0:
        return []
    reasons: list[str] = []
    if (
        stats["expansion_ratio"] >= _SENTENCE_EXPANSION_REVIEW_RATIO
        and stats["sentence_count"] - stats["source_sentence_count"] >= 3
    ):
        reasons.append("sentence_count_expansion")
    if stats["short_ratio"] >= _SHORT_SENTENCE_CHAIN_RATIO and stats["avg_words"] <= _SHORT_SENTENCE_WORD_LIMIT:
        reasons.append("short_sentence_chain")
    if (
        stats["repeated_first_ratio"] >= _REPEATED_START_REVIEW_RATIO
        or stats["repeated_frame_ratio"] >= _REPEATED_FRAME_REVIEW_RATIO
    ):
        reasons.append("repeated_sentence_start")
    if stats["transition_start_ratio"] >= _TRANSITION_STACK_REVIEW_RATIO:
        reasons.append("mechanical_transition_stack")
    if catalogue_sentence_chain(text):
        reasons.append("catalogue_sentence_chain")
    return reasons


def _candidate_sentence_stats(text: str, source_paragraph: Paragraph) -> dict[str, float]:
    paragraphs = split_paragraphs(text)
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph.sentences]
    sentence_count = len(sentences)
    source_sentence_count = max(1, len(source_paragraph.sentences))
    if not sentences:
        return {
            "sentence_count": 0.0,
            "source_sentence_count": float(source_sentence_count),
            "expansion_ratio": 0.0,
            "avg_words": 0.0,
            "short_ratio": 0.0,
            "repeated_first_ratio": 0.0,
            "repeated_frame_ratio": 0.0,
            "transition_start_ratio": 0.0,
        }
    first_words: dict[str, int] = {}
    first_frames: dict[str, int] = {}
    transition_starts = 0
    transition_tokens = {
        "also", "additionally", "again", "another", "finally", "further", "furthermore",
        "however", "instead", "moreover", "therefore", "these", "this", "those", "it", "they",
    }
    for sentence in sentences:
        parts = [
            part.strip(".,:;!?\"'“”’").casefold()
            for part in sentence.text.split()
            if part.strip(".,:;!?\"'“”’")
        ]
        first = parts[0] if parts else ""
        if first in {"a", "an", "the"} and len(parts) > 1:
            first = parts[1]
        if first:
            first_words[first] = first_words.get(first, 0) + 1
            if first in transition_tokens:
                transition_starts += 1
        if len(parts) >= 3:
            frame = " ".join(parts[:3])
            first_frames[frame] = first_frames.get(frame, 0) + 1
    return {
        "sentence_count": float(sentence_count),
        "source_sentence_count": float(source_sentence_count),
        "expansion_ratio": sentence_count / source_sentence_count,
        "avg_words": sum(sentence.word_count for sentence in sentences) / max(1, sentence_count),
        "short_ratio": sum(1 for sentence in sentences if sentence.word_count <= _SHORT_SENTENCE_WORD_LIMIT) / max(1, sentence_count),
        "repeated_first_ratio": max(first_words.values(), default=0) / max(1, sentence_count) if sentence_count >= 4 else 0.0,
        "repeated_frame_ratio": max(first_frames.values(), default=0) / max(1, sentence_count) if sentence_count >= 4 else 0.0,
        "transition_start_ratio": transition_starts / max(1, sentence_count),
    }


def _compresses_list_repair(text: str, source_paragraph: Paragraph) -> bool:
    source_sentences = source_paragraph.sentences
    if not any(_has_list_shape(sentence.text) for sentence in source_sentences):
        return False
    candidate_sentences = [
        sentence
        for paragraph in split_paragraphs(text)
        for sentence in paragraph.sentences
    ]
    return len(candidate_sentences) < len(source_sentences)


def _candidate_contract_violation(text: str, source_paragraph: Paragraph) -> bool:
    return (
        _repeats_sentence_intent(text)
        or robotic_sentence_chain(text)
        or catalogue_sentence_chain(text)
        or bool(_hard_integrity_blockers(text))
        or _hard_candidate_contract_violation(text, source_paragraph)
    )


def _hard_candidate_contract_violation(text: str, source_paragraph: Paragraph) -> bool:
    return (
        _keeps_forbidden_list_contract(text, source_paragraph)
        or _adds_unsubmitted_success_close(text, source_paragraph)
        or _adds_citation_report_sentence(text, source_paragraph)
    )


def _adds_citation_report_sentence(text: str, source_paragraph: Paragraph) -> bool:
    source = str(source_paragraph.text or "")
    if re.search(r"\b(?:reports?|reported|observed|documented|noted|confirms?|proves?|supports?|supported|highlights?|highlighted)\b", source, flags=re.I):
        return False
    value = str(text or "")
    active_report = re.search(
        r"\b[A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?\s*\((?:19|20)\d{2}\)\s+(?:reports?|reported|observed|documented|noted|confirms?|proves?|supports?|supported|highlights?|highlighted)\b",
        value,
        flags=re.I,
    )
    passive_report = re.search(
        r"\b(?:result|finding|findings|evidence|study|source|claim|point)\s+(?:was|were|is|are)\s+(?:reported|observed|documented|noted|confirmed|proved|supported|highlighted)\s*\([^)]*(?:19|20)\d{2}[^)]*\)",
        value,
        flags=re.I,
    )
    return bool(active_report or passive_report)


def _repeats_sentence_intent(text: str) -> bool:
    sentence_words = [
        _content_word_set(sentence.text)
        for paragraph in split_paragraphs(text)
        for sentence in paragraph.sentences
    ]
    rows = [words for words in sentence_words if words]
    for left, right in zip(rows, rows[1:]):
        overlap = len(left & right) / max(1, min(len(left), len(right)))
        if overlap >= 0.7:
            return True
    return False


def _keeps_forbidden_list_contract(text: str, source_paragraph: Paragraph) -> bool:
    if not any(_has_list_shape(sentence.text) for sentence in source_paragraph.sentences):
        return False
    if _natural_compact_list_allowed(text, source_paragraph):
        return False
    for paragraph in split_paragraphs(text):
        for sentence in paragraph.sentences:
            if _without_parentheticals(sentence.text).count(",") >= 3:
                return True
    return False


def _natural_compact_list_allowed(text: str, source_paragraph: Paragraph) -> bool:
    candidate_sentences = [
        sentence
        for paragraph in split_paragraphs(text)
        for sentence in paragraph.sentences
    ]
    if not candidate_sentences or len(candidate_sentences) > len(source_paragraph.sentences):
        return False
    comma_list_sentences = [
        sentence for sentence in candidate_sentences
        if _without_parentheticals(sentence.text).count(",") >= 3
    ]
    if len(comma_list_sentences) != 1:
        return False
    return not catalogue_sentence_chain(text) and not robotic_sentence_chain(text)


def _without_parentheticals(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", str(text or ""))


def _adds_unsubmitted_success_close(text: str, source_paragraph: Paragraph) -> bool:
    source = str(source_paragraph.text or "").casefold()
    outcome_terms = _final_outcome_terms()
    if _has_any_terms(source, outcome_terms):
        return False
    candidate_sentences = [
        sentence.text.casefold()
        for paragraph in split_paragraphs(text)
        for sentence in paragraph.sentences
    ]
    if not candidate_sentences:
        return False
    final = candidate_sentences[-1]
    return _has_any_terms(final, outcome_terms)


def _final_outcome_terms() -> tuple[str, ...]:
    return ("success", "essential", "readiness", "ready", "complex", "real-world")


def _has_any_terms(text: str, terms: tuple[str, ...]) -> bool:
    pattern = "|".join(re.escape(term) for term in terms)
    return bool(re.search(rf"\b(?:{pattern})\b", str(text or ""), flags=re.I))


def _adds_extra_conclusion_beat(text: str, source_paragraph: Paragraph) -> bool:
    source_sentences = source_paragraph.sentences
    candidate_sentences = [
        sentence
        for paragraph in split_paragraphs(text)
        for sentence in paragraph.sentences
    ]
    if len(candidate_sentences) <= len(source_sentences):
        return False
    final_text = candidate_sentences[-1].text
    final_words = _content_word_set(final_text)
    source_words = _content_word_set(source_paragraph.text)
    if not final_words:
        return False
    source_overlap = len(final_words & source_words) / max(1, len(final_words))
    if _conclusion_like_start(final_text):
        return source_overlap < 0.75
    return len(candidate_sentences) > len(source_sentences) + 2 and source_overlap < 0.55


def _replaces_final_source_beat_with_conclusion(text: str, source_paragraph: Paragraph) -> bool:
    source_sentences = source_paragraph.sentences
    candidate_sentences = [
        sentence
        for paragraph in split_paragraphs(text)
        for sentence in paragraph.sentences
    ]
    if not source_sentences or not candidate_sentences:
        return False
    final_candidate = candidate_sentences[-1].text
    if not _conclusion_like_start(final_candidate):
        return False
    final_source = source_sentences[-1].text
    if _conclusion_like_start(final_source):
        return False
    source_words = _content_word_set(final_source)
    candidate_window = " ".join(sentence.text for sentence in candidate_sentences[-3:])
    candidate_words = _content_word_set(candidate_window)
    if not source_words or not candidate_words:
        return False
    return len(source_words & candidate_words) / max(1, len(source_words)) < 0.45


def _polarity_violation(text: str, source_paragraph: Paragraph) -> bool:
    source = str(source_paragraph.text or "").casefold()
    candidate = str(text or "").casefold()
    if "not less" in source:
        reversal_terms = ("reduce", "reduced", "reducing", "limit", "limits", "limited", "limiting", "diminish", "diminished", "diminishing", "smaller", "weaker")
        negated_reduction = re.search(r"\b(?:does\s+not|do\s+not|not|no)\s+(?:reduce|reduced|reducing|limit|limits|limited|limiting|diminish|diminished|diminishing)\b", candidate)
        if not negated_reduction and any(re.search(rf"\b{term}\b", candidate) for term in reversal_terms):
            return True
        if re.search(r"(?<!not\s)\bless\s+[a-z]+", candidate):
            return True
        if not (negated_reduction or any(shape in candidate for shape in ("not less", "remains important", "still important", "more important"))):
            return True
    if "no longer" in source and "no longer" not in candidate and not re.search(r"\bnot\b|\bdoes not\b|\bdo not\b", candidate):
        return True
    if "not only" in source and "not only" not in candidate and not _not_only_relation_preserved(source, candidate):
        return True
    if "not only" in source and not (re.search(r"\b(?:also|as well as|too)\b", candidate) or ("but also" in source and all(term.casefold() in candidate for term in source_terms(source.split("but also", 1)[-1], limit=3)[:2]))):
        return True
    if "not only" in source and _malformed_not_only(candidate):
        return True
    if _modal_risk_hardened(source, candidate):
        return True
    if _reverses_rather_than(source, candidate):
        return True
    if _moves_not_always_to_positive_side(source, candidate):
        return True
    if _not_always_scope_inverted(source, candidate):
        return True
    return False


def _modal_risk_hardened(source: str, candidate: str) -> bool:
    modal_shapes = (
        ("may become", "become"),
        ("might become", "become"),
        ("could become", "become"),
        ("may submit", "submit"),
        ("might submit", "submit"),
        ("could submit", "submit"),
    )
    for modal, bare in modal_shapes:
        if modal in source and modal not in candidate:
            if re.search(rf"\b(?:they|people|users|[A-Za-z][A-Za-z'’-]*(?:ers|ents|ants|ists|ors))\s+{bare}\b", candidate):
                return True
    return False


def _not_only_relation_preserved(source: str, candidate: str) -> bool:
    if not re.search(r"\b(?:also|as well|too|both)\b", candidate):
        return False
    first_side, second_side = _not_only_source_sides(source)
    if not first_side or not second_side:
        return True
    return _side_specific_term_present(first_side, second_side, candidate) and _side_specific_term_present(second_side, first_side, candidate)


def _not_only_source_sides(source: str) -> tuple[str, str]:
    tail = source.split("not only", 1)[-1]
    if "but also" in tail:
        first, second = tail.split("but also", 1)
        return first, second
    if "also" in tail:
        first, second = tail.split("also", 1)
        return first, second
    return tail, ""


def _side_term_present(side_text: str, candidate: str) -> bool:
    terms = _side_terms(side_text)
    if not terms:
        return True
    return any(term in candidate for term in terms[:5])


def _side_specific_term_present(side_text: str, other_side_text: str, candidate: str) -> bool:
    other_keys = {_side_term_key(term) for term in _side_terms(other_side_text)}
    terms = [
        term for term in _side_terms(side_text)
        if _side_term_key(term) not in other_keys
    ]
    if not terms:
        terms = _side_terms(side_text)
    if not terms:
        return True
    return any(term in candidate for term in terms[:5])


def _side_terms(side_text: str) -> list[str]:
    return [
        term.casefold()
        for term in source_terms(side_text, limit=8)
        if _contrast_side_term(term)
    ]


def _contrast_side_term(term: str) -> bool:
    value = str(term or "").casefold()
    return len(value) > 3 and value not in {
        "only", "also", "about", "that", "with", "into", "from", "people", "person", "group", "groups", "users", "user", "those", "these"
    }


def _side_term_key(term: str) -> str:
    value = str(term or "").casefold()
    return value[:-1] if len(value) > 4 and value.endswith("s") else value


def _malformed_not_only(candidate: str) -> bool:
    return bool(re.search(r"\bnot\s+only\s+(?:a\s+|an\s+|the\s+)?(?:serious\s+)?(?:concern|issue|problem|challenge)\b", candidate))


def _reverses_rather_than(source: str, candidate: str) -> bool:
    for match in re.finditer(r"\b(rather than|instead of)\b", source):
        left_terms = source_terms(source[max(0, match.start() - 80):match.start()], limit=6)[-3:]
        right_terms = source_terms(source[match.end():match.end() + 80], limit=6)[:3]
        if not left_terms or not right_terms:
            continue
        left = "|".join(re.escape(term.casefold()) for term in left_terms)
        right = "|".join(re.escape(term.casefold()) for term in right_terms)
        if re.search(rf"\b(?:{right})\b[\s\S]{{0,80}}\b(?:rather than|instead of)\b[\s\S]{{0,80}}\b(?:{left})\b", candidate):
            return True
    return False


def _moves_not_always_to_positive_side(source: str, candidate: str) -> bool:
    if "not always" not in source:
        return False
    for match in re.finditer(r"\bnot always\b", source):
        positive_terms = source_terms(source[max(0, match.start() - 80):match.start()], limit=6)
        scoped_terms = _scoped_positive_terms(positive_terms)
        for term in scoped_terms:
            if re.search(rf"\b(?:do\s+not\s+always|does\s+not\s+always|not\s+always)\b(?:\W+\w+){{0,4}}\W+{re.escape(term)}\b", candidate):
                return True
    return False


def _scoped_positive_terms(terms: list[str]) -> set[str]:
    return {
        term.casefold()
        for term in terms
        if len(term) >= 4
        and re.search(r"[a-z]", term, flags=re.I)
        and not term.casefold().endswith(("tion", "sion", "ment", "ness", "ity"))
    }


def _not_always_scope_inverted(source: str, candidate: str) -> bool:
    if "not always" not in source:
        return False
    scoped_terms: set[str] = set()
    for match in re.finditer(r"\bnot always\b", source):
        scoped_terms.update(_scoped_positive_terms(source_terms(source[max(0, match.start() - 100):match.start()], limit=8)))
        scoped_terms.update(_scoped_positive_terms(source_terms(source[match.end():match.end() + 100], limit=8)))
    if not scoped_terms:
        return False
    pattern = "|".join(re.escape(term) for term in sorted(scoped_terms, key=len, reverse=True))
    return bool(re.search(rf"(?:\b(?:may|might|could)\s+always|(?<!not\s)\balways)\s+(?:{pattern})\b", candidate, flags=re.I))


def _conclusion_like_start(text: str) -> bool:
    lowered = str(text or "").strip().casefold()
    return lowered.startswith((
        "the shift", "the gap", "this shows", "this means", "this highlights",
        "this demonstrates", "overall", "therefore", "in conclusion", "the goal",
        "the result",
    ))


def _has_list_shape(text: str) -> bool:
    visible = str(text or "")
    lowered = visible.casefold()
    return visible.count(",") >= 2 or ";" in visible or lowered.count(" and ") >= 2


def _source_drift_penalty(variant: Variant, paragraph: Paragraph) -> float:
    if variant.source == "source_preserved":
        return 0.0
    source_words = _content_word_set(paragraph.text)
    candidate_words = _content_word_set(variant.text)
    if not source_words or not candidate_words:
        return 0.0
    new_words = candidate_words - source_words
    flagged = {
        word
        for word in new_words
        if len(word) >= 8 or word.endswith(("tion", "ment", "ity", "ness", "ance", "ence"))
    }
    reviewed_words = _reviewed_word_set(variant)
    unsupported = [word for word in flagged if word not in reviewed_words]
    unsupported_ratio = len(unsupported) / max(1, len(candidate_words))
    adjusted = max(0.0, unsupported_ratio - 0.04)
    return round(adjusted * 8.0, 3)


def _unreviewed_bridge_violation(variant: Variant, paragraph: Paragraph) -> bool:
    if variant.source == "source_preserved":
        return False
    reviewed_words = _reviewed_word_set(variant)
    if reviewed_words:
        return False
    source_words = _content_word_set(paragraph.text)
    candidate_words = _content_word_set(variant.text)
    new_words = candidate_words - source_words
    flagged = {
        word
        for word in new_words
        if _word_base(word) not in source_words
        if len(word) >= 8 or word.endswith(("tion", "ment", "ity", "ness", "ance", "ence", "form"))
    }
    return bool(flagged)


def _hard_integrity_blockers(text: str) -> list[str]:
    return [
        blocker for blocker in candidate_integrity_blockers(text)
        if blocker in {
            "planner_language_leakage",
            "external_narrator_reporting_chain",
            "malformed_negation_order",
            "missing_verb_after_negation_scope",
            "malformed_serial_verb_chain",
            "malformed_nominal_stack",
            "malformed_nonhuman_activity_predicate",
            "malformed_telegraphic_predicate",
            "unnatural_completion_phrase",
            "dangling_consequence_tail",
            "dangling_additive_tail",
            "standalone_additive_fragment",
            "malformed_parallel_connector_list",
            "malformed_parallel_verb_tail",
            "redundant_trust_phrase",
            "keyword_dump_sequence",
            "lost_serial_punctuation",
            "capitalized_common_noun_mid_sentence",
            "repeated_platform_catalogue",
            "repeated_subject_start",
            "vague_unintroduced_reliance",
            "malformed_tool_actor_relation",
            "malformed_with_finite_clause",
            "malformed_tool_skill_predicate",
            "malformed_contrast_pair",
            "malformed_additive_predicate",
            "proxy_context_adjective_stack",
            "generic_role_inflation",
            "unsupported_evidence_tail",
            "awkward_modal_double_hedge",
            "duplicated_assessment_consequence",
            "premature_assessment_consequence",
            "transition_label_final_consequence",
            "compressed_final_consequence_list",
        }
    ]


def _word_base(word: str) -> str:
    return str(word or "").removesuffix("'s").rstrip("s")


def _reviewed_word_set(variant: Variant) -> set[str]:
    chunks: list[str] = []
    for row in [*(variant.author_review_items or []), *(variant.author_proxy_provenance or [])]:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if isinstance(value, str):
                chunks.append(value)
    return _content_word_set(" ".join(chunks))


def _content_word_set(text: str) -> set[str]:
    stop = {
        "about", "above", "after", "again", "against", "also", "because", "before",
        "being", "between", "could", "every", "from", "have", "into", "more",
        "most", "only", "other", "over", "should", "still", "their", "there",
        "these", "those", "through", "under", "where", "which", "while", "would",
    }
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", str(text or ""))
        if token.casefold() not in stop
    }
