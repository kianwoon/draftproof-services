from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .json_io import parse_json
from .plan import Plan
from .scan import scan_text
from .text import Paragraph, source_terms, split_paragraphs, word_count


class ChatClient(Protocol):
    def chat(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class Variant:
    id: str
    text: str
    source: str
    author_proxy_provenance: list[dict[str, Any]] | None = None
    author_review_items: list[dict[str, Any]] | None = None
    coverage_map: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_variants(paragraph: Paragraph, plan: Plan, *, client: ChatClient) -> list[Variant]:
    variants = [source_preserved_variant(paragraph)]
    try:
        response = client.chat(
            build_prompt(paragraph, plan),
            system=(
                "Return valid JSON only. Preserve submitted meaning and coverage, "
                "but do not preserve submitted wording, order, list rhythm, opener, or closure shape. "
                "Generate from coverage beats and mark inferred bridge material for author review."
            ),
            temperature=0.12,
            top_p=0.75,
            max_tokens=None,
            response_format={"type": "json_object"},
        )
    except Exception:
        return variants
    try:
        parsed = parse_variants(parse_json(getattr(response, "raw_content", "") or response.content))
    except ValueError:
        return variants
    variants.extend(parsed)
    return variants


def build_prompt(paragraph: Paragraph, plan: Plan) -> str:
    golden_route = plan.ai_safe_route.get("golden_route", {})
    payload = {
        "task": "coverage_beat_paragraph_generation",
        "golden_question": golden_route.get(
            "question_rule",
            "What did the writer see, read, compare, struggle with, or decide, and why does it matter?",
        ),
        "golden_route": {
            "compact_formula": golden_route.get("compact_formula", "Scope it. Anchor it. Show it. Explain it. Close it."),
            "technical_formula": golden_route.get(
                "technical_formula",
                "narrow claim -> human/context anchor -> concrete evidence -> decompressed reasoning -> non-generic close",
            ),
        },
        "context_anchors": {
            "context_terms": _prompt_context_terms(plan)[:24],
            "named_references": plan.author_proxy_context.get("named_references", []),
            "years": plan.author_proxy_context.get("years", []),
            "citation_spans": plan.author_proxy_context.get("citation_spans", []),
            "quoted_terms": plan.author_proxy_context.get("quoted_terms", []),
        },
        "coverage_beats_must_all_appear": _prompt_coverage_beats(plan),
        "construction_recipes": _prompt_construction_recipes(plan),
        "planner_decision": plan.ai_safe_route.get("llm_planner_decision", {}),
        "variant_requirements": [
            {
                "id": "v1",
                "route": "plain coverage-beat route: one short source-relation sentence per beat, no added benefit or intensity claim",
            },
            {
                "id": "v2",
                "route": "golden-question route: source basis, concrete detail, narrow interpretation, and careful close",
            },
            {
                "id": "v3",
                "route": "context-anchor route: use available anchors first, then carry coverage beats without polished expansion",
            },
        ],
        "generation_rules": [
            "Answer the golden_question through the paragraph route, not as visible notes.",
            "Treat planner_decision.finding_contracts as the primary build contract.",
            "Every scanner finding must be visibly resolved by a final sentence mapped in coverage_map.",
            "For each finding_contract, execute writer_must_do and avoid writer_must_not_do.",
            "Use safe_rebuild_shape as the sentence construction pattern; do not copy unsafe_original_shape.",
            "If planner_decision.contract_gaps is not empty, repair those gaps in the final paragraph instead of copying the weak planner shape.",
            "If a contract_gap says a risky source phrase was copied, that quoted phrase must not appear in the final text.",
            "Before returning, compare final text against planner_decision.contract_gaps and rewrite any sentence that still contains a copied risky phrase or planning label.",
            "If safe_rebuild_shape contains placeholder brackets, instantiate it using the contract coverage_terms before writing.",
            "Do not copy planning labels from safe_rebuild_shape into the paragraph, including relation, beat, anchor, route, contract, source term, or writer.",
            "For context_anchor_gap or broad_claim contracts, the final sentence must start from the named antecedent, not this, that, model, result, or a generic claim.",
            "For broad_claim contracts, use a scoped partial-relation sentence such as only part of, one part of, limited to, or explains only part of when supported by the contract.",
            "For closure contracts, split continuity and limitation into separate sentences instead of comma-but, as/which, or broad-summary closure.",
            "Use construction_recipes as the positive build plan. Do not treat avoid rules as the plan.",
            "If planner_decision.status is ok, follow planner_decision.paragraph_blueprint in order before using fallback recipes.",
            "Each final sentence must map to a planner_decision.paragraph_blueprint step or a coverage beat; do not invent extra route filler.",
            "For each blueprint step, use must_include terms and safe_sentence_shape, while avoiding must_avoid_shape.",
            "For every coverage beat, apply its construction_recipe before writing the sentence.",
            "Before final text, make coverage_map show the construction_recipe_id used for each sentence.",
            "Create a coverage_map before the final text. Every coverage beat must appear in the candidate meaning.",
            "Every source-side contrast must survive. If a beat contains terms from both sides of a contrast, preserve both sides instead of keeping only the first side.",
            "Write one ordinary sentence for each coverage beat first; merge only when the merge does not create a list, repeated frame, or overloaded sentence.",
            "Preserve submitted meaning, factual scope, and coverage. Do not preserve exact source wording, source order, source opener, source list rhythm, or source closure shape.",
            "Each sentence should carry one coverage beat unless two beats connect naturally without a comma tail, which clause, where clause, semicolon, dash, or list.",
            "Preserve every beat's polarity_markers. Do not invert not, no longer, not only, not always, rather than, instead of, or without.",
            "Do not add abstract consequences, institutional labels, or polished explanation after a coverage beat unless that idea is present in coverage_beats_must_all_appear or context_anchors.",
            "Do not add unsubmitted intensity, speed, frequency, importance, ease, or benefit claims. If the draft does not say how fast, common, essential, easy, immediate, abundant, or effective something is, do not add that quality.",
            "A coverage sentence should state the source relation only. Do not append why it matters unless that reasoning is a separate coverage beat or an author-review bridge.",
            "Do not upgrade plain source nouns into academic labels. Use everyday wording and the concrete coverage terms.",
            "Every sentence must start with a concrete submitted anchor: source noun, named reference, cited source, quoted term, setting, actor, object, condition, or comparison.",
            "For each coverage beat, start the mapped sentence with one of its starter_terms when possible.",
            "When starter_terms are present, do not start that beat's sentence with a different coverage term just because it appears earlier in the capsule.",
            "If a beat has no starter_terms, it is probably a vague reference beat; bridge it to a concrete earlier coverage term instead of starting with model, result, this, that, it, or the same vague noun.",
            "Do not start sentences with Today, Now, In the past, This, That, It, They, These, Those, Overall, Therefore, or However.",
            "Before returning, rewrite any sentence that starts with a pronoun, forbidden opener, vague reference noun, or a term that is not allowed by the mapped beat's starter_terms.",
            "The final coverage beat must not start with But, This, That, It, or Model; start from a concrete actor, source term, or young/people/learn when those starter_terms are available.",
            "If a coverage capsule contains a forbidden opener, choose another concrete term from the same beat.",
            "Do not add comma-plus-explanation tails to coverage sentences. Put the reasoning in a separate sentence only when needed.",
            "Do not use semicolons, em dashes, parenthetical asides, or colon-led lists.",
            "Do not put three or more examples, nouns, actions, qualities, or source anchors in one sentence.",
            "Do not turn grouped coverage terms into repeated one-item sentences. For a grouped beat, write one ordinary relation sentence carrying the group.",
            "When coverage beats are split from the same source sentence, do not recombine them into the same final list sentence.",
            "When submitted context has many anchors, use at most two in one sentence or split them across connected sentences.",
            "Use plain submitted source terms. Avoid elevated labels, metaphor labels, abstract engagement phrases, polished role labels, and broad capture/reflect phrases unless present in the source.",
            "Preserve meaning and coverage. Expand when needed for grounding, concrete detail, or reasoning.",
            "Use author_proxy_provenance or author_review_items for inferred bridge wording.",
            "Return JSON only.",
        ],
        "plain_route_contract_for_v1": {
            "purpose": "force the first variant to be a low-polish source-coverage candidate before any higher-author-proxy version",
            "lexical_rule": "Use coverage capsule words, context anchor words, function words, and plain relation verbs. Avoid descriptive adjectives/adverbs that are not in the submitted anchors.",
            "allowed_relation_moves": [
                "A remains part of the route.",
                "A and B add another route.",
                "A sits beside B.",
                "A gives another place to compare or check.",
                "A creates or changes the setting.",
                "A is not the scarce/main part anymore.",
                "The difficult part is deciding or judging the remaining source terms.",
            ],
            "invalid_pattern_contrasts": [
                {
                    "invalid_shape": "<anchor> has become a common/essential/dynamic place where <actor> benefits.",
                    "reason": "adds intensity and benefit claims not carried by the coverage beat",
                },
                {
                    "invalid_shape": "<role/issue/goal> has become more important/challenging/serious.",
                    "reason": "starts from an evaluative label before showing the source relation",
                },
                {
                    "invalid_shape": "<actor> become/becomes <descriptor A>, <descriptor B>, and <descriptor C>.",
                    "reason": "recombines a descriptor list that should be carried as separate source relations",
                },
                {
                    "invalid_shape": "<anchor> serves as an essential tool for <broad outcome>.",
                    "reason": "turns a source anchor into polished explanatory wrapping",
                },
                {
                    "invalid_shape": "<anchor>; it is widely/easily/instantly available.",
                    "reason": "uses punctuation and unsubmitted intensity to smooth the claim",
                },
            ],
        },
        "author_proxy_policy": {
            "enabled": True,
            "non_interrupting": True,
            "allowed_provenance": ["source_preserved", "inferred_from_draft", "needs_author_confirmation", "must_replace"],
            "review_required": True,
            "responsibility_boundary": "DraftProof drafts and labels provenance; the user must review and owns facts, citations, anchors, and author-proxy bridges before submission.",
            "rule": "Do not stop for questions. When a needed anchor is not directly supported by submitted text, continue the rewrite and include it in author_review_items.",
        },
        "required_shape": {
            "paragraph_count": 1,
            "sentence_count": "uncapped during golden-route discovery; use as many ordinary prose sentences as needed to preserve coverage and answer the route questions",
            "word_count": "uncapped during golden-route discovery; preserve source coverage and expand only when needed for grounding, concrete detail, or reasoning",
        },
        "output_schema": {
            "variants": [
                {
                    "id": "v1",
                    "mode": "coverage_beat_generation",
                    "route_fragments": {},
                    "coverage_map": [
                        {
                            "coverage_beat_id": "coverage beat id",
                            "construction_recipe_id": "recipe id used for this beat",
                            "finding_contract_id": "finding contract id resolved, empty if none",
                            "sentence": "candidate sentence carrying this beat",
                        }
                    ],
                    "text": "replacement paragraph only",
                    "author_proxy_provenance": [],
                    "author_review_items": [],
                },
                {
                    "id": "v2",
                    "mode": "golden_question_generation",
                    "route_fragments": {},
                    "coverage_map": [],
                    "text": "replacement paragraph only",
                    "author_proxy_provenance": [],
                    "author_review_items": [],
                },
                {
                    "id": "v3",
                    "mode": "context_anchor_generation",
                    "route_fragments": {},
                    "coverage_map": [],
                    "text": "replacement paragraph only",
                    "author_proxy_provenance": [],
                    "author_review_items": [],
                },
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _prompt_coverage_beats(plan: Plan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recent_concrete_terms: list[str] = []
    for beat in plan.ai_safe_route.get("coverage_beats", []):
        if not isinstance(beat, dict):
            continue
        terms = [str(term) for term in beat.get("coverage_terms", []) if str(term).strip()]
        finding_tags = [str(tag) for tag in beat.get("finding_tags", []) if str(tag).strip()]
        capsule_terms = _resolved_capsule_terms(terms, recent_concrete_terms)
        polarity_markers = [str(marker) for marker in beat.get("polarity_markers", []) if str(marker).strip()]
        capsule_terms = _capsule_with_polarity(capsule_terms, polarity_markers)
        intent = _coverage_intent(terms, recent_concrete_terms)
        rows.append({
            "beat_id": beat.get("beat_id"),
            "source_sentence_id": beat.get("source_sentence_id"),
            "finding_tags": finding_tags,
            "construction_recipe_id": beat.get("construction_recipe_id"),
            "construction_recipe": beat.get("construction_recipe", {}),
            "coverage_capsule": " | ".join(capsule_terms),
            "coverage_terms": terms,
            "polarity_markers": polarity_markers,
            "polarity_instruction": _polarity_instruction(polarity_markers),
            "starter_terms": beat.get("starter_terms", []),
            "coverage_intent": intent,
            "finding_instruction": _finding_instruction(finding_tags),
            "generation_duty": beat.get("generation_duty"),
            "merge_rule": beat.get("merge_rule"),
            "route_rule": "carry this meaning without reusing the original phrase order",
        })
        recent_concrete_terms = _remember_concrete_terms(recent_concrete_terms, terms)
    return rows


def _prompt_construction_recipes(plan: Plan) -> list[dict[str, Any]]:
    rows = plan.ai_safe_route.get("construction_recipes", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _finding_instruction(tags: list[str]) -> str:
    tag_set = set(tags)
    instructions: list[str] = []
    if "packed_list" in tag_set:
        instructions.append("Keep grouped anchors grouped; do not recombine them into a comma list and do not split into repeated one-item sentences.")
    if "context_anchor_gap" in tag_set:
        instructions.append("Name the concrete antecedent; do not start from this, that, it, they, these, or those.")
    if tag_set & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"}:
        instructions.append("Put the concrete source relation before any evaluative word; avoid broad labels like important, challenge, or concern unless a submitted anchor appears in the same sentence.")
        instructions.append("When the source beat uses an evaluative label, convert it into an observable role, pressure, decision, support, or contrast relation instead of repeating the label as the main claim.")
    if "paraphrase_smoothing" in tag_set:
        instructions.append("Use shorter source-level pressure; do not turn the beat into smoother academic explanation.")
    if "sentence_overload" in tag_set:
        instructions.append("Split at source relationships; do not summarize the beat into one longer polished sentence.")
    return " ".join(instructions)


def _capsule_with_polarity(capsule_terms: list[str], polarity_markers: list[str]) -> list[str]:
    if not polarity_markers:
        return capsule_terms
    rows: list[str] = []
    seen: set[str] = set()
    marker_words = {
        word.casefold()
        for marker in polarity_markers
        for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", marker)
    }
    for value in [*polarity_markers, *capsule_terms]:
        key = value.casefold()
        if key in seen or key in marker_words and value not in polarity_markers:
            continue
        rows.append(value)
        seen.add(key)
    return rows


def _polarity_instruction(polarity_markers: list[str]) -> str:
    lowered = " ".join(polarity_markers).casefold()
    if "not only" in lowered:
        return "Do not state the marked idea as a positive claim by itself; write it as not enough, not the only basis, or part of a wider contrast."
    if "not always" in lowered:
        return "Keep the limitation; do not turn it into an absolute claim."
    if "rather than" in lowered or "instead of" in lowered:
        return "Keep the contrast direction; do not reverse which side is preferred or criticized."
    if "no longer" in lowered:
        return "Keep the time contrast; do not say the old condition still fully applies."
    if "without" in lowered or "not" in lowered:
        return "Keep the negation; do not turn the marked idea into an affirmative claim."
    return ""


def _resolved_capsule_terms(terms: list[str], recent_concrete_terms: list[str]) -> list[str]:
    lowered = {term.casefold() for term in terms}
    if lowered and lowered <= {"model", "exists"} and recent_concrete_terms:
        return [*recent_concrete_terms[-4:], "still exists"]
    if {"fully", "reflects"} & lowered:
        anchor = recent_concrete_terms[-3:] if recent_concrete_terms else ["previous route"]
        return [*anchor, "does not fully fit", "young people learn"]
    return terms


def _coverage_intent(terms: list[str], recent_concrete_terms: list[str]) -> str:
    lowered = {term.casefold() for term in terms}
    if lowered and lowered <= {"model", "exists"}:
        return "continuity beat: name the previous concrete route instead of writing a vague model sentence"
    if {"fully", "reflects"} & lowered:
        return "contrast beat: explain that the previous concrete route does not fully fit how young people learn now"
    return "source coverage beat"


def _remember_concrete_terms(existing: list[str], terms: list[str]) -> list[str]:
    vague = {"model", "exists", "longer", "fully", "reflects"}
    rows = [*existing]
    seen = {term.casefold() for term in rows}
    for term in terms:
        key = term.casefold()
        if key in vague or key in seen:
            continue
        rows.append(term)
        seen.add(key)
    return rows[-12:]


def _prompt_context_terms(plan: Plan) -> list[str]:
    coverage_terms = {
        str(term).casefold()
        for beat in plan.ai_safe_route.get("coverage_beats", [])
        if isinstance(beat, dict)
        for term in beat.get("coverage_terms", [])
    }
    forbidden = {
        "today", "now", "overall", "therefore", "however", "this", "that",
        "they", "these", "those", "past", "model",
    }
    rows: list[str] = []
    for term in plan.author_proxy_context.get("context_terms", []):
        key = str(term).casefold()
        if key in forbidden or key in coverage_terms:
            continue
        rows.append(term)
    return rows


def source_preserved_variant(paragraph: Paragraph) -> Variant:
    return Variant(id="source_preserved", text=paragraph.text, source="source_preserved")


def parse_variants(payload: Any) -> list[Variant]:
    rows = payload.get("variants") if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    variants: list[Variant] = []
    for index, row in enumerate(rows if isinstance(rows, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if text:
            variants.append(
                Variant(
                    id=str(row.get("id") or row.get("variant_id") or f"v{index}"),
                    text=text,
                    source="llm",
                    author_proxy_provenance=_review_rows(row.get("author_proxy_provenance"), "inferred_from_draft"),
                    author_review_items=_review_rows(row.get("author_review_items"), "needs_author_confirmation"),
                    coverage_map=_coverage_rows(row.get("coverage_map")),
                )
            )
    return variants


def _coverage_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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
            return max(improved, key=lambda variant: _variant_rank(variant, paragraph, source_words))
        return source_variant
    return max(variants, key=lambda variant: _variant_rank(variant, paragraph, source_words))


def _has_meaningful_movement(candidate: Variant, source_variant: Variant, paragraph: Paragraph) -> bool:
    before = scan_text(source_variant.text)
    after = scan_text(candidate.text)
    finding_drop = before.scores["finding_count"] - after.scores["finding_count"]
    risk_drop = before.scores["mean_sentence_shape_risk"] - after.scores["mean_sentence_shape_risk"]
    drift_penalty = _source_drift_penalty(candidate, paragraph)
    if _compresses_list_repair(candidate.text, paragraph):
        return False
    if _replaces_final_source_beat_with_conclusion(candidate.text, paragraph):
        return False
    if finding_drop >= 2 and risk_drop >= -2.0:
        return True
    if finding_drop >= 1 and risk_drop >= 5.0:
        return True
    return False


def _variant_rank(variant: Variant, paragraph: Paragraph, source_words: int) -> tuple[float, float, float, float, bool, int]:
    scan = scan_text(variant.text)
    words = word_count(variant.text)
    quality_penalty = _mechanical_quality_penalty(variant.text, paragraph)
    source_drift_penalty = _source_drift_penalty(variant, paragraph)
    compression_penalty = 2.0 if _compresses_list_repair(variant.text, paragraph) else 0.0
    extra_beat_penalty = 2.0 if _adds_extra_conclusion_beat(variant.text, paragraph) else 0.0
    final_beat_penalty = 2.0 if _replaces_final_source_beat_with_conclusion(variant.text, paragraph) else 0.0
    mean_risk = scan.scores["mean_sentence_shape_risk"]
    return (
        -scan.scores["finding_count"],
        -mean_risk,
        -(quality_penalty + source_drift_penalty * 0.25 + compression_penalty + extra_beat_penalty + final_beat_penalty),
        _coverage(variant.text, paragraph),
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
    candidate_words = _content_word_set(final_candidate)
    if not source_words or not candidate_words:
        return False
    return len(source_words & candidate_words) / max(1, len(source_words)) < 0.45


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
        if len(word) >= 9 or word.endswith(("tion", "ment", "ity", "ness", "ance", "ence"))
    }
    reviewed_words = _reviewed_word_set(variant)
    unsupported = [word for word in flagged if word not in reviewed_words]
    unsupported_ratio = len(unsupported) / max(1, len(candidate_words))
    adjusted = max(0.0, unsupported_ratio - 0.04)
    return round(adjusted * 8.0, 3)


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


def _coverage(text: str, paragraph: Paragraph) -> float:
    anchors = source_terms(paragraph.text, limit=24)
    lowered = str(text or "").casefold()
    return sum(1 for anchor in anchors if anchor.casefold() in lowered) / len(anchors) if anchors else 1.0
