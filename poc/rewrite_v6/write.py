from __future__ import annotations
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

from .coverage_guard import coverage_ratio, missing_required_source_terms
from .integrity_guard import candidate_integrity_blockers
from .json_io import parse_json
from .paragraph_architecture import apply_architecture_split_text, architecture_split_contract
from .plan import Plan
from .prose_quality import fragment_trace_penalty, has_fragment_or_trace_sentences, repair_generated_prose, robotic_sentence_chain
from .review_provenance import annotate_review_items
from .scan import scan_text
from .text import Paragraph, source_terms, split_paragraphs, word_count
from .writer_prompt import build_prompt


class ChatClient(Protocol):
    def chat(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> Any:
        ...

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
    try:
        response = client.chat(
            build_prompt(paragraph, plan),
            system=(
                "Return valid JSON only with a variants array matching the requested variant ids. "
                "If list_contract_active is true, no final text sentence may contain two or more commas. "
                "When the source uses not only / but also inside an overloaded sentence, preserve both sides in separate ordinary sentences; do not keep one long not-only sentence. "
                "Do not use pronoun-led also wrappers such as It also or This also for concrete source work; name the subject and split the consequence. "
                "Avoid repeated sentence starts; do not use summary-noun wrappers such as The example or The result as the route. "
                "Use source_units as the only source text; writer_execution_contract rows reference source_sentence_id and must drive the final text. "
                "Write complete grammatical sentences with normal articles, prepositions, subjects, and objects; never split an action from its object into adjacent fragments. Preserve paired alternatives when the source uses either/or or not-yet wording. Preserve submitted meaning, coverage, and first-person voice when present, but do not preserve submitted wording, order, "
                "list rhythm, opener, or closure shape. Never return a final sentence that starts with And, But, Or, Which, Where, In, Through, During, From, This, That, These, Those, As, or Thereby. Keep citations parenthetical; do not create citation report sentences."
            ),
            temperature=0.12,
            top_p=0.75,
            max_tokens=None,
            response_format={"type": "json_object"},
            app_label="writer",
        )
        variants.extend(replace(v, text=repair_generated_prose(apply_architecture_split_text(v.text, split_contract), paragraph.text)) for v in parse_variants(parse_json(getattr(response, "raw_content", "") or response.content)))
    except (Exception, ValueError):
        pass
    return _dedupe_variants(variants)


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
    before = scan_text(source_variant.text)
    after = scan_text(candidate.text)
    finding_drop = before.scores["finding_count"] - after.scores["finding_count"]
    risk_drop = before.scores["mean_sentence_shape_risk"] - after.scores["mean_sentence_shape_risk"]
    if word_count(candidate.text) < max(8, int(word_count(paragraph.text) * 0.35)):
        return False
    if finding_drop >= 2 and risk_drop >= -2.0:
        return True
    if finding_drop >= 1 and risk_drop >= 0.0:
        return True
    if finding_drop >= 0 and risk_drop >= 8.0:
        return True
    return False


def _annotate_selected_variant(variant: Variant, paragraph: Paragraph) -> Variant:
    annotated = annotate_review_items(variant, paragraph.text)
    review_items = list(annotated.author_review_items or [])
    review_reasons: list[str] = []
    if _polarity_violation(annotated.text, paragraph):
        review_reasons.append("polarity_or_contrast_changed")
    if missing_required_source_terms(annotated.text, paragraph):
        review_reasons.append("source_terms_missing")
    if _hard_candidate_contract_violation(annotated.text, paragraph):
        review_reasons.append("hard_contract_warning")
    if has_fragment_or_trace_sentences(annotated.text):
        review_reasons.append("sentence_quality_warning")
    review_reasons.extend(candidate_integrity_blockers(annotated.text))
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
    source_drift_penalty = _source_drift_penalty(variant, paragraph)
    compression_penalty = 2.0 if _compresses_list_repair(variant.text, paragraph) else 0.0
    extra_beat_penalty = 2.0 if _adds_extra_conclusion_beat(variant.text, paragraph) else 0.0
    final_beat_penalty = 2.0 if _replaces_final_source_beat_with_conclusion(variant.text, paragraph) else 0.0
    polarity_penalty = 1.0 if _polarity_violation(variant.text, paragraph) else 0.0
    bridge_penalty = 4.0 if _unreviewed_bridge_violation(variant, paragraph) else 0.0
    contract_penalty = 4.0 if _candidate_contract_violation(variant.text, paragraph) else 0.0
    mean_risk = scan.scores["mean_sentence_shape_risk"]
    return (
        -scan.scores["finding_count"],
        -mean_risk,
        -(quality_penalty + fragment_trace_penalty(variant.text) + source_drift_penalty * 0.25 + compression_penalty + extra_beat_penalty + final_beat_penalty + polarity_penalty + bridge_penalty + contract_penalty),
        coverage_ratio(variant.text, paragraph),
        words >= source_words * 0.9,
        -abs(words - source_words),
    )


def _mechanical_quality_penalty(text: str, source_paragraph: Paragraph) -> float:
    paragraphs = split_paragraphs(text)
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph.sentences]
    if not sentences:
        return 8.0
    source_sentence_count = max(1, len(source_paragraph.sentences))
    sentence_count = len(sentences)
    avg_words = sum(sentence.word_count for sentence in sentences) / max(1, sentence_count)
    short_ratio = sum(1 for sentence in sentences if sentence.word_count <= 7) / max(1, sentence_count)
    first_words: dict[str, int] = {}
    first_frames: dict[str, int] = {}
    for sentence in sentences:
        parts = [
            part.strip(".,:;!?\"'“”’").casefold()
            for part in sentence.text.split()
            if part.strip(".,:;!?\"'“”’")
        ]
        first = parts[0] if parts else ""
        if first:
            first_words[first] = first_words.get(first, 0) + 1
        if len(parts) >= 3:
            frame = " ".join(parts[:3])
            first_frames[frame] = first_frames.get(frame, 0) + 1
    repeated_first_ratio = max(first_words.values(), default=0) / max(1, sentence_count)
    repeated_frame_ratio = max(first_frames.values(), default=0) / max(1, sentence_count)
    penalty = 0.0
    if sentence_count > source_sentence_count * 1.8:
        penalty += min(4.0, (sentence_count / source_sentence_count) - 1.0)
    if short_ratio >= 0.45 and avg_words <= 9.0:
        penalty += short_ratio * 4.0
    if repeated_first_ratio >= 0.35:
        penalty += repeated_first_ratio * 3.0
    if repeated_frame_ratio >= 0.30:
        penalty += repeated_frame_ratio * 4.0
    return round(penalty, 3)


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
    for paragraph in split_paragraphs(text):
        for sentence in paragraph.sentences:
            if _without_parentheticals(sentence.text).count(",") >= 2:
                return True
    return False


def _without_parentheticals(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", str(text or ""))


def _adds_unsubmitted_success_close(text: str, source_paragraph: Paragraph) -> bool:
    source = str(source_paragraph.text or "").casefold()
    if re.search(r"\b(success|essential|readiness|ready|complex|real-world)\b", source):
        return False
    candidate_sentences = [
        sentence.text.casefold()
        for paragraph in split_paragraphs(text)
        for sentence in paragraph.sentences
    ]
    if not candidate_sentences:
        return False
    final = candidate_sentences[-1]
    return bool(re.search(r"\b(success|essential|readiness|ready|complex|real-world)\b", final))


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
    if _reverses_rather_than(source, candidate):
        return True
    if _moves_not_always_to_positive_side(source, candidate):
        return True
    return False


def _not_only_relation_preserved(source: str, candidate: str) -> bool:
    if not re.search(r"\b(?:also|as well|too|both)\b", candidate):
        return False
    first_side, second_side = _not_only_source_sides(source)
    if not first_side or not second_side:
        return True
    return _side_term_present(first_side, candidate) and _side_term_present(second_side, candidate)


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
    terms = [
        term.casefold()
        for term in source_terms(side_text, limit=8)
        if len(term) > 3 and term.casefold() not in {"only", "also", "about", "that", "with", "into", "from"}
    ]
    if not terms:
        return True
    return any(term in candidate for term in terms[:5])


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
        scoped_terms = {term.casefold() for term in positive_terms if term.casefold() in {"learn", "pass", "progress"}}
        for term in scoped_terms:
            if re.search(rf"\b(?:do\s+not\s+always|does\s+not\s+always|not\s+always)\b(?:\W+\w+){{0,4}}\W+{re.escape(term)}\b", candidate):
                return True
    return False


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
