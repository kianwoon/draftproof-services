from __future__ import annotations
import json, os, re
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

from .coverage_guard import coverage_ratio, missing_required_source_terms
from .json_io import parse_json
from .paragraph_architecture import apply_architecture_split_text, architecture_split_contract
from .plan import Plan
from .prompt_shape import coverage_loss_contract, paragraph_sentence_plan
from .prose_quality import fragment_trace_penalty, has_fragment_or_trace_sentences, repair_generated_prose
from .review_provenance import annotate_review_items
from .scan import scan_text
from .sentence_rows import compile_or_fallback_text
from .text import Paragraph, source_terms, split_paragraphs, strip_leading_heading, word_count


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
    coverage_map: list[dict[str, Any]] | None = None
    route_answer_cards: list[dict[str, Any]] | None = None

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
                "If paragraph_sentence_plan has required_sentence_groups, cover every group in coverage_map; adjacent groups may share a natural sentence when grammar, coverage, and source order stay clear. "
                "Write complete grammatical sentences with normal articles, prepositions, subjects, and objects; never split an action from its object into adjacent fragments. Preserve paired alternatives when the source uses either/or or not-yet wording. Preserve submitted meaning, coverage, and first-person voice when present, but do not preserve submitted wording, order, "
                "list rhythm, opener, or closure shape."
            ),
            temperature=0.12,
            top_p=0.75,
            max_tokens=None,
            response_format={"type": "json_object"},
        )
        variants.extend(replace(v, text=repair_generated_prose(apply_architecture_split_text(v.text, split_contract), paragraph.text)) for v in parse_variants(parse_json(getattr(response, "raw_content", "") or response.content)))
    except (Exception, ValueError):
        pass
    return _dedupe_variants(variants)


def build_prompt(paragraph: Paragraph, plan: Plan, *, variant_focus: dict[str, str] | None = None) -> str:
    golden_route = plan.ai_safe_route.get("golden_route", {})
    variant_requirements = [variant_focus] if variant_focus else _requested_variant_requirements()
    coverage_beats = _prompt_coverage_beats(plan)
    sentence_plan, split_contract = paragraph_sentence_plan(paragraph, coverage_beats), architecture_split_contract(paragraph, plan)
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
            "context_terms": _prompt_context_terms(plan)[:5],
            "named_references": _prompt_named_references(plan),
            "years": plan.author_proxy_context.get("years", []),
            "citation_spans": plan.author_proxy_context.get("citation_spans", []),
            "quoted_terms": plan.author_proxy_context.get("quoted_terms", []),
        },
        "content_word_boundary": {
            "allowed_content_terms": _allowed_content_terms(paragraph, plan),
            "rule": (
                "Use these terms and ordinary function words as the content boundary. "
                "Any new content noun, adjective, or abstract label must be necessary bridge wording and listed in author_review_items."
            ),
        },
        "polarity_constraints": _polarity_constraints(paragraph),
        "coverage_beats_must_all_appear": coverage_beats,
        "paragraph_sentence_plan": sentence_plan,
        "architecture_split_contract": split_contract,
        "coverage_loss_contract": coverage_loss_contract(sentence_plan),
        "author_route_questions": _prompt_author_route_questions(plan),
        "construction_recipes": _prompt_construction_recipes(plan),
        "planner_decision": plan.ai_safe_route.get("llm_planner_decision", {}),
        "document_signal_contracts": plan.ai_safe_route.get("document_signal_contracts", []),
        "hard_generation_requirements": {
            "required_variant_ids": [row["id"] for row in variant_requirements],
            "exact_variant_count": len(variant_requirements),
            "list_contract_active": _list_contract_active(plan),
            "list_contract_rule": (
                "No final text sentence may contain two or more commas when list_contract_active is true. "
                "Use paired relation sentences instead of comma lists."
            ),
            "source_preserved_shape_is_failure": "Do not return the original sentence route with synonyms. Change clause route and sentence grouping.",
        },
        "active_variant": variant_focus or {},
        "variant_requirements": variant_requirements,
        "generation_rules": [
            "Produce only the active_variant when active_variant is present.",
            "When active_variant is empty, return exactly the ids in hard_generation_requirements.required_variant_ids.",
            "If hard_generation_requirements.list_contract_active is true, no final text sentence may contain two or more commas.",
            "If active_variant is empty, produce materially different variants. If two variants would have the same text, return only the stronger one.",
            "Do not create variant difference by simply reversing paragraph order. Keep readable source logic unless a slot explicitly permits movement.",
            "If architecture_split_contract.active is true, final text may contain paragraph breaks that divide source/support, author/context evidence, and reasoning/consequence; every architecture_split_contract.functional_groups must_survive_terms anchor and every coverage_loss_contract source_terms_to_carry item must still survive.",
            "A variant mode is not decorative: its distinctive_obligation must change the paragraph route.",
            "Answer the golden_question through the paragraph route, not as visible notes.",
            "Before writing text, fill route_answer_cards for every author_route_question.",
            "Each route_answer_cards row must include the exact question_id, answer_basis_terms used, bridge_provenance, and draft_sentence.",
            "Each route_answer_cards.draft_sentence is a route sketch, not a coverage limit; final text must carry all slot source_terms_to_carry as meaning.",
            "A route card sentence must not use which, where, especially, because, rather than, vast, complex, critical, digital platform, or landscape.",
            "A route card sentence must not introduce content words outside answer_basis_terms_used unless bridge_provenance is needs_author_confirmation and author_review_items explains them.",
            "Do not limit final text to route cards; expand each slot until exact anchors survive and revoiceable source terms are carried as plain meaning.",
            "Do not write a polished paragraph first and backfill cards afterward.",
            "Build the paragraph by answering author_route_questions inside the final prose.",
            "Do not write meta-prose such as the writer sees, the writer reads, the writer compares, the writer struggles, the writer decides, or the writer began.",
            "Each author_route_question must change the construction of the mapped source beat; do not answer it by copying the original sentence.",
            "For author/context/support gaps, use the question answer as Author-Proxy bridge material and label any inferred bridge in author_review_items.",
            "For packed-list and sentence-overload questions, decompose the relationship before writing; do not preserve submitted facts as written if that carries the unsafe list route forward.",
            "Stay inside content_word_boundary.allowed_content_terms unless a new term is necessary as reviewable Author-Proxy bridge wording.",
            "Do not add new abstract content words such as critical, overwhelming, central, landscape, nuanced, complex, abundant, unverified, accessibility, importance, or reliability unless that exact idea appears in content_word_boundary.",
            "If a new content word is not in content_word_boundary, either replace it with an allowed source term or include an author_review_items row explaining the bridge.",
            "Use neighboring context only to name the bridge behind a vague source beat. Do not import neighboring examples, platforms, names, or lists into the replacement unless the selected paragraph's own coverage beats contain them.",
            "Treat planner_decision.finding_contracts and document_signal_contracts as the primary build contract.",
            "Every scanner finding and document signal contract must be visibly resolved by final sentences mapped in coverage_map.",
            "For each finding_contract, execute writer_must_do and avoid writer_must_not_do; for each document signal contract, execute writer_obligation.",
            "Use safe_rebuild_shape as the sentence construction pattern; do not copy unsafe_original_shape.",
            "If planner_decision.contract_gaps is not empty, repair those gaps in the final paragraph instead of copying the weak planner shape.",
            "If a contract_gap says a risky source phrase was copied, that quoted phrase must not appear in the final text.",
            "Before returning, compare final text against planner_decision.contract_gaps and rewrite any sentence that still contains a copied risky phrase or planning label.",
            "If safe_rebuild_shape contains placeholder brackets, instantiate it using the contract coverage_terms before writing.",
            "Do not copy planning labels from safe_rebuild_shape into the paragraph, including relation, beat, anchor, route, contract, source term, or writer.",
            "Do not write repair-trace labels such as is the context, same point, same limit, keeps both sides visible, or the other side.",
            "Treat paragraph_sentence_plan.revoiceable_source_terms as meaning-only. Do not copy polished or evaluative revoiceable wording when a simpler source relation can carry the same meaning.",
            "If a source relation needs both an action and a condition, split them into adjacent sentences instead of joining them with while, that, which, or because.",
            "For context_anchor_gap or broad_claim contracts, the final sentence must start from the named antecedent, not this, that, model, result, or a generic claim.",
            "For broad_claim contracts, use a scoped partial-relation sentence such as only part of, one part of, limited to, or explains only part of when supported by the contract.",
            "For closure contracts, split continuity and limitation into separate sentences instead of comma-but, as/which, or broad-summary closure.",
            "Use construction_recipes.repair_sequence in order as the positive build plan. Do not treat avoid rules as the plan.",
            "If planner_decision.status is ok, follow planner_decision.paragraph_blueprint in order before using fallback recipes.",
            "Each final sentence must map to a planner_decision.paragraph_blueprint step or a coverage beat; do not invent extra route filler.",
            "For each blueprint step, use must_include terms and safe_sentence_shape, while avoiding must_avoid_shape.",
            "For every coverage beat, apply its construction_recipe before writing the paragraph.",
            "Before final text, make coverage_map show construction_recipe_id and sentence_row_id for each covered group.",
            "Build from paragraph_sentence_plan. If a slot has required_sentence_groups, cover every group in coverage_map; adjacent groups may share a natural sentence when coverage stays explicit.",
            "The executor compiles final paragraph text from sentence_rows first; use coverage_map to prove group coverage, not as a forced sentence split.",
            "When required_sentence_groups exist, coverage_map needs one row per group, but sentence_rows may merge adjacent groups into ordinary prose when coverage stays explicit.",
            "Every final sentence must map to a sentence_slot_id or explain which coverage beat required an extra sentence.",
            "Every source-side contrast must survive. If a beat contains terms from both sides of a contrast, preserve both sides instead of keeping only the first side.",
            "Do not invert source polarity. If the source says not less, no longer, not only, not always, without, or does not, preserve that direction instead of rewriting it as reduction, limitation, or a positive claim.",
            "Satisfy every polarity_constraints row before optimizing wording or rhythm.",
            "For a not only polarity constraint, write the first side as not enough by itself, then write the second side with also matters or also carries weight.",
            "For a not always polarity constraint, do not repeat not always across a sentence chain. Use one contrast sentence for the positive side and limited side, then one follow-up sentence only if another limited item remains.",
            "Do not write a repair trace. Group related coverage beats into ordinary paragraph sentences when meaning and contrast stay intact.",
            "Do not compress coverage to satisfy preferred_sentence_count; use more sentences when source coverage needs them.",
            "Preserve submitted meaning, factual scope, and coverage. Do not preserve exact source wording, source order, source opener, source list rhythm, or source closure shape.",
            "Preserve every beat's polarity_markers. Do not invert not, no longer, not only, not always, rather than, instead of, or without.",
            "Do not add unsubmitted intensity, speed, frequency, importance, ease, readiness, success, complexity, or benefit claims. If the draft does not say how fast, common, essential, easy, immediate, abundant, effective, successful, ready, or complex something is, do not add that quality.",
            "Do not upgrade plain source nouns into academic labels. Use everyday wording and the concrete coverage terms.",
            "Each sentence should start from a concrete submitted anchor when possible: source noun, named reference, cited source, quoted term, setting, actor, object, condition, or comparison.",
            "When starter_terms are present, do not start that beat's sentence with a different coverage term just because it appears earlier in the capsule.",
            "If a beat has no starter_terms, it is probably a vague reference beat; bridge it to a concrete earlier coverage term instead of starting with model, result, this, that, it, or the same vague noun.",
            "Do not start sentences with Today, Now, In the past, This, That, It, They, These, Those, Overall, Therefore, However, or Not always.",
            "Before returning, rewrite any sentence that starts with a pronoun, forbidden opener, or vague reference noun.",
            "Do not use semicolons, em dashes, parenthetical asides, or colon-led lists.",
            "Do not put three or more examples, actions, qualities, or source anchors in one sentence unless the original meaning needs that grouped phrase and no safer split is possible.",
            "For a final four-item ability, reward, need, or skill list, pair the first two items and the last two items in separate relation clauses or sentences; do not repeat the whole four-item list and do not use a comma list.",
            "When a not only reward beat is followed by ability or skill beats, use this route: the context does not only reward the first side. It also rewards the first pair. The second pair carries the next part. Do not add a success, readiness, or essential-skills ending.",
            "Do not turn grouped coverage terms into repeated one-item sentences. For a grouped beat, write one ordinary relation sentence carrying the group.",
            "Use plain submitted source terms and keep first-person when the source uses I or my. Avoid elevated labels, metaphor labels, abstract engagement phrases, polished role labels, and broad capture/reflect phrases unless present in the source.",
            "Preserve meaning and coverage. Expand when needed for grounding, concrete detail, or reasoning.",
            "Use author_proxy_provenance or author_review_items for inferred bridge wording.",
            "Return JSON only.",
        ],
        "plain_route_contract_for_v1": "Low-polish source-coverage candidate: use source terms and plain relation verbs; avoid unsubmitted intensity, benefit claims, semicolon lists, and descriptor lists.",
        "author_proxy_policy": {
            "enabled": True,
            "non_interrupting": True,
            "allowed_provenance": ["source_preserved", "inferred_from_draft", "needs_author_confirmation", "must_replace"],
            "review_required": True,
            "responsibility_boundary": "DraftProof drafts and labels provenance; the user must review and owns facts, citations, anchors, and author-proxy bridges before submission.",
            "rule": "Do not stop for questions. When a needed anchor is not directly supported by submitted text, continue the rewrite and include it in author_review_items.",
        },
        "required_shape": {
            "paragraph_count": split_contract["paragraph_count"] if split_contract["active"] else 1,
            "sentence_count": "uncapped during golden-route discovery; use as many ordinary prose sentences as needed to preserve coverage and answer the route questions",
            "preferred_sentence_count": _preferred_sentence_count(paragraph),
            "word_count": "uncapped during golden-route discovery; preserve source coverage and expand only when needed for grounding, concrete detail, or reasoning",
        },
        "output_schema": {"variants": [_variant_schema(requirement) for requirement in variant_requirements]},
    }
    prefix = (
        "Return valid JSON only. The JSON must contain a variants array with exactly "
        + ", ".join(row["id"] for row in variant_requirements)
        + ".\n"
    )
    return prefix + json.dumps(payload, ensure_ascii=False, indent=2)


def _variant_requirements() -> list[dict[str, str]]:
    return [
        {
            "id": "v1",
            "mode": "coverage_beat_generation",
            "route": "plain coverage-beat route: group related source relations without trace-like one-beat sentences",
            "distinctive_obligation": "Use paragraph_sentence_plan slot order as the spine; merge grouped beats inside each slot before moving to the next slot. Change the opener route inside each affected slot.",
        },
        {
            "id": "v2",
            "mode": "golden_question_generation",
            "route": "golden-question route: source basis, concrete detail, narrow interpretation, and careful close",
            "distinctive_obligation": "Answer author_route_questions as the spine; use the question answer to change clause route while keeping readable source logic.",
        },
        {
            "id": "v3",
            "mode": "context_anchor_generation",
            "route": "context-anchor route: use available anchors first, then carry coverage beats without polished expansion",
            "distinctive_obligation": "Start each affected slot from the strongest available context/source antecedent; keep paragraph order unless the antecedent is inside the same source slot.",
        },
    ]


def _requested_variant_requirements() -> list[dict[str, str]]:
    return _variant_requirements()[:_requested_variant_count()]
def _requested_variant_count() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_WRITER_VARIANTS", "1")
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(1, min(3, value))


def _variant_schema(requirement: dict[str, str]) -> dict[str, Any]:
    return {
        "id": requirement["id"],
        "mode": requirement["mode"],
        "route_fragments": {},
        "route_answer_cards": [{
            "question_id": "author_route_question id",
            "source_sentence_id": "source sentence id",
            "answer_basis_terms_used": ["terms from answer_basis_terms or coverage beats"],
            "bridge_provenance": "source_preserved | inferred_from_draft | needs_author_confirmation",
            "draft_sentence": "plain route sketch; final text may expand it to preserve coverage",
        }],
        "coverage_map": [{
            "sentence_slot_id": "paragraph_sentence_plan slot id covered by this row",
            "sentence_row_id": "sentence_rows id where this coverage appears",
            "coverage_beat_ids": ["one or more coverage beat ids covered by this row"],
            "coverage_beat_id": "coverage beat id; include required_sentence_group_id when paragraph_sentence_plan requires a group row",
            "construction_recipe_id": "recipe id used for this beat",
            "finding_contract_id": "finding contract id resolved, empty if none",
        }],
        "sentence_rows": [{"sentence_row_id": "srow_001", "sentence_slot_id": "same as coverage_map sentence_slot_id", "coverage_beat_ids": ["coverage beats carried by this sentence"], "sentence": "the exact sentence to compile into final paragraph text", "paragraph_break_after": False}],
        "text": "replacement text only; may contain paragraph breaks only when architecture_split_contract.active is true",
        "author_proxy_provenance": [],
        "author_review_items": [],
    }


def _preferred_sentence_count(paragraph: Paragraph) -> str:
    source_count = max(1, len(paragraph.sentences))
    low = max(1, source_count)
    high = min(source_count + 2, max(source_count, int(source_count * 1.35) + 1))
    return f"{low} to {max(low, high)} ordinary sentences"


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
            "grouped_relation": bool(beat.get("grouped_relation")),
            "coverage_intent": intent,
            "finding_instruction": _finding_instruction(finding_tags),
            "generation_duty": beat.get("generation_duty"),
            "merge_rule": beat.get("merge_rule"),
            "route_rule": "carry this meaning without reusing the original phrase order",
        })
        recent_concrete_terms = _remember_concrete_terms(recent_concrete_terms, terms)
    return rows


def _list_contract_active(plan: Plan) -> bool:
    return any(
        "packed_list" in [str(tag) for tag in beat.get("finding_tags", [])]
        for beat in plan.ai_safe_route.get("coverage_beats", [])
        if isinstance(beat, dict)
    )


def _polarity_constraints(paragraph: Paragraph) -> list[dict[str, Any]]:
    text = str(paragraph.text or "")
    lowered = text.casefold()
    rows: list[dict[str, Any]] = []
    if "not less" in lowered:
        rows.append({
            "source_marker": "not less",
            "required_direction": "preserve that the role, value, or condition is not reduced",
            "required_shapes": ["not less", "remains important", "still important", "more important"],
            "forbidden_shapes": ["reduced", "reducing", "limited", "limits", "limiting", "diminished", "less than"],
        })
    if "no longer" in lowered:
        rows.append({
            "source_marker": "no longer",
            "required_direction": "preserve that the old condition does not fully apply now",
            "required_shapes": ["no longer", "does not", "not fully", "not the same"],
            "forbidden_shapes": ["still fully", "continues to fully"],
        })
    if "not only" in lowered:
        rows.append({
            "source_marker": "not only",
            "required_direction": "preserve that the first side is insufficient by itself and the second side also matters",
            "required_shapes": ["does not only reward", "not enough by itself", "not the only", "also matters", "also carries"],
            "forbidden_shapes": ["only", "rather than", "rewards people who remember facts, not only", "not only a concern", "not only a serious concern"],
        })
    if "rather than" in lowered or "instead of" in lowered:
        rows.append({
            "source_marker": "rather than",
            "required_direction": "preserve which side the source treats as the pressure or preference",
            "required_shapes": ["rather than", "instead of", "over"],
            "forbidden_shapes": ["reversed contrast direction"],
        })
    if "not always" in lowered:
        rows.append({
            "source_marker": "not always",
            "required_direction": "preserve the limited side of the contrast without moving the limitation onto the positive side",
            "required_shapes": ["not always", "does not always", "do not always"],
            "forbidden_shapes": ["not always learn to pass", "not always pass"],
        })
    return rows


def _allowed_content_terms(paragraph: Paragraph, plan: Plan) -> list[str]:
    terms: list[str] = []
    terms.extend(source_terms(strip_leading_heading(paragraph.text), limit=80))
    for value in _prompt_context_terms(plan)[:5]:
        terms.append(str(value))
    for value in _prompt_named_references(plan)[:16]:
        terms.extend(source_terms(str(value), limit=8) or [str(value)])
    for key in ("years", "citation_spans", "quoted_terms"):
        for value in plan.author_proxy_context.get(key, [])[:16]:
            terms.extend(source_terms(str(value), limit=8) or [str(value)])
    for beat in plan.ai_safe_route.get("coverage_beats", []):
        if isinstance(beat, dict):
            terms.extend(str(term) for term in beat.get("coverage_terms", []) if str(term).strip())
    return _dedupe_terms(terms)[:120]


def _dedupe_terms(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        rows.append(text)
        seen.add(key)
    return rows


def _prompt_construction_recipes(plan: Plan) -> list[dict[str, Any]]:
    rows = plan.ai_safe_route.get("construction_recipes", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _prompt_author_route_questions(plan: Plan) -> list[dict[str, Any]]:
    rows = plan.ai_safe_route.get("author_route_questions", [])
    if not isinstance(rows, list):
        return []
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        compact.append({
            "question_id": row.get("question_id"),
            "source_sentence_id": row.get("source_sentence_id"),
            "finding_tags": row.get("finding_tags", []),
            "question": row.get("question"),
            "answer_basis_terms": row.get("answer_basis_terms", []),
            "writer_duty": row.get("writer_duty"),
            "author_proxy_policy": row.get("author_proxy_policy"),
        })
    return compact


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
    if any(str(marker).casefold() == "not always" for marker in polarity_markers):
        return [*capsule_terms, "not always limit"]
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
        return "Do not state the first side as a positive claim by itself. If this is a reward relation, write the context does not only reward the first side, then pair the next source items without repeating the whole list."
    if "not always" in lowered:
        return "Keep the limitation on the limited side only. Use one contrast sentence, not repeated not-always sentences."
    if re.search(r"\bmore\s+\w+\s+than\b", lowered):
        return "Keep the more-than comparison direction; the first side is stronger and the than-side is limited, not endorsed."
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
    selected_terms = {str(term).casefold() for term in plan.author_proxy_context.get("source_terms", [])}
    neighbor_named_terms = {
        term.casefold()
        for value in plan.author_proxy_context.get("named_references", [])
        for term in source_terms(str(value), limit=8)
        if term.casefold() not in selected_terms
    }
    neighbor_list_terms = _neighbor_list_terms(plan) - selected_terms
    forbidden = {
        "today", "now", "overall", "therefore", "however", "this", "that",
        "they", "these", "those", "past", "model",
    }
    rows: list[str] = []
    for term in plan.author_proxy_context.get("context_terms", []):
        key = str(term).casefold()
        if key in forbidden or key in coverage_terms:
            continue
        if key in neighbor_named_terms or key in neighbor_list_terms:
            continue
        rows.append(term)
    return rows


def _neighbor_list_terms(plan: Plan) -> set[str]:
    context = plan.author_proxy_context
    terms: set[str] = set()
    neighbor_text = "\n".join([
        str(context.get("before_context") or ""),
        str(context.get("after_context") or ""),
    ])
    for sentence in re.split(r"(?<=[.!?])\s+", neighbor_text):
        if sentence.count(",") < 2 and len(re.findall(r"\b(?:and|or|also|but)\b", sentence, flags=re.I)) < 2:
            continue
        for term in source_terms(sentence, limit=32):
            terms.add(term.casefold())
    return terms


def _prompt_named_references(plan: Plan) -> list[str]:
    selected_terms = {str(term).casefold() for term in plan.author_proxy_context.get("source_terms", [])}
    rows: list[str] = []
    for value in plan.author_proxy_context.get("named_references", []):
        terms = source_terms(str(value), limit=8)
        if any(term.casefold() in selected_terms for term in terms) or not _reference_in_neighbor_list(plan, str(value)):
            rows.append(str(value))
    return rows


def _reference_in_neighbor_list(plan: Plan, value: str) -> bool:
    needle = str(value or "").strip()
    if not needle:
        return False
    context = plan.author_proxy_context
    neighbor_text = "\n".join([
        str(context.get("before_context") or ""),
        str(context.get("after_context") or ""),
    ])
    for sentence in re.split(r"(?<=[.!?])\s+", neighbor_text):
        if needle not in sentence:
            continue
        if re.search(r"\b(?:19|20)\d{2}[a-z]?\b", sentence):
            return False
        named_count = len(re.findall(r"\b(?:[A-Z][A-Za-z0-9'’-]{2,}|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'’-]{2,}|[A-Z]{2,}))*\b", sentence))
        if named_count >= 2:
            return True
        if sentence.count(",") >= 2 or len(re.findall(r"\b(?:and|or)\b", sentence, flags=re.I)) >= 2:
            return True
    return False


def source_preserved_variant(paragraph: Paragraph) -> Variant:
    return Variant(id="source_preserved", text=paragraph.text, source="source_preserved")


def parse_variants(payload: Any) -> list[Variant]:
    rows = payload.get("variants") if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    variants: list[Variant] = []
    seen_texts: set[str] = set()
    for index, row in enumerate(rows if isinstance(rows, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        text = compile_or_fallback_text(row)
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
                    coverage_map=_coverage_rows(row.get("coverage_map")),
                    route_answer_cards=_coverage_rows(row.get("route_answer_cards")),
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
            return annotate_review_items(max(improved, key=lambda variant: _variant_rank(variant, paragraph, source_words)), paragraph.text)
        return source_variant
    return annotate_review_items(max(variants, key=lambda variant: _variant_rank(variant, paragraph, source_words)), paragraph.text)


def _has_meaningful_movement(candidate: Variant, source_variant: Variant, paragraph: Paragraph) -> bool:
    before = scan_text(source_variant.text)
    after = scan_text(candidate.text)
    finding_drop = before.scores["finding_count"] - after.scores["finding_count"]
    risk_drop = before.scores["mean_sentence_shape_risk"] - after.scores["mean_sentence_shape_risk"]
    if _compresses_list_repair(candidate.text, paragraph):
        return False
    if _replaces_final_source_beat_with_conclusion(candidate.text, paragraph):
        return False
    if _polarity_violation(candidate.text, paragraph):
        return False
    if missing_required_source_terms(candidate.text, paragraph):
        return False
    if _candidate_contract_violation(candidate.text, paragraph):
        return False
    if has_fragment_or_trace_sentences(candidate.text):
        return False
    if finding_drop >= 2 and risk_drop >= -2.0:
        return True
    if finding_drop >= 1 and risk_drop >= 0.0:
        return True
    if finding_drop >= 0 and risk_drop >= 8.0:
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
    polarity_penalty = 4.0 if _polarity_violation(variant.text, paragraph) else 0.0
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
        or _keeps_forbidden_list_contract(text, source_paragraph)
        or _adds_unsubmitted_success_close(text, source_paragraph)
    )


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
            if sentence.text.count(",") >= 2:
                return True
    return False


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
    candidate_words = _content_word_set(final_candidate)
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
    if "not only" in source and "not only" not in candidate:
        has_not_enough = "not enough" in candidate or "not the only" in candidate
        has_also_side = "also" in candidate or ("but also" in source and all(term.casefold() in candidate for term in source_terms(source.split("but also", 1)[-1], limit=3)[:2]))
        if not (has_not_enough and has_also_side):
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
