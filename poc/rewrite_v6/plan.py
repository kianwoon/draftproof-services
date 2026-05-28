from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Any

from .scan import Finding, Scan, findings_for_paragraph, select_target_paragraph
from .text import Paragraph, source_anchor_terms, source_terms, strip_leading_heading as _strip_leading_heading


@dataclass(frozen=True)
class PlanAction:
    sentence_id: str
    source_text: str
    tags: list[str]
    operation: str
    method: str
    preserve_terms: list[str]
    do_not: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Plan:
    paragraph_id: str
    route_goal: str
    opening_terms: list[str]
    actions: list[PlanAction]
    paragraph_strategy: dict[str, Any]
    author_proxy_context: dict[str, Any]
    ai_safe_route: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
            "route_goal": self.route_goal,
            "opening_terms": list(self.opening_terms),
            "actions": [action.to_dict() for action in self.actions],
            "paragraph_strategy": dict(self.paragraph_strategy),
            "author_proxy_context": dict(self.author_proxy_context),
            "ai_safe_route": dict(self.ai_safe_route),
            "handoff_validation": _plan_handoff_validation(self.actions),
        }


def build_plan(
    scan: Scan,
    excluded_paragraph_ids: set[str] | None = None,
    priority_paragraph_ids: set[str] | None = None,
) -> tuple[Paragraph, Plan]:
    paragraph = select_target_paragraph(scan, excluded_paragraph_ids, priority_paragraph_ids)
    findings = {finding.sentence_id: finding for finding in findings_for_paragraph(scan, paragraph.id)}
    actions: list[PlanAction] = []
    for sentence in paragraph.sentences:
        finding = findings.get(sentence.id)
        tags = _planning_tags(finding.tags if finding else [], sentence.text)
        actions.append(
            PlanAction(
                sentence_id=sentence.id,
                source_text=sentence.text,
                tags=tags,
                operation=_operation(tags),
                method=_method(tags),
                preserve_terms=_source_contract_terms(sentence.text, term_limit=24),
                do_not=_do_not(tags),
            )
        )
    author_proxy_context = _author_proxy_context(scan, paragraph)
    paragraph_strategy = _paragraph_strategy(paragraph, actions, author_proxy_context)
    plan = Plan(
        paragraph_id=paragraph.id,
        route_goal="change sentence route while preserving source meaning and reducing packed or predictable shape",
        opening_terms=_source_contract_terms(
            paragraph.sentences[0].text if paragraph.sentences else paragraph.text,
            term_limit=12,
        )[:6],
        actions=actions,
        paragraph_strategy=paragraph_strategy,
        author_proxy_context=author_proxy_context,
        ai_safe_route=_ai_safe_route(paragraph, actions, author_proxy_context, paragraph_strategy),
    )
    return paragraph, plan


def _source_contract_terms(text: str, *, term_limit: int) -> list[str]:
    cleaned = _strip_leading_heading(text)
    return _dedupe([*_scope_contract_markers(cleaned), *source_anchor_terms(cleaned, term_limit=term_limit, phrase_limit=4)])[:term_limit]


def _scope_contract_markers(text: str) -> list[str]:
    value = str(text or "")
    lowered = value.casefold()
    markers: list[str] = []
    for phrase in (
        "not always",
        "not only",
        "no longer",
        "rather than",
        "instead of",
        "without",
    ):
        if phrase in lowered:
            markers.append(phrase)
    for match in re.finditer(r"\b(?:may|might|could|can)\s+(?:become|submit|depend|rely|learn|encourage|create|support|raise|make)\b", value, flags=re.I):
        markers.append(match.group(0).lower())
    return _dedupe(markers)


def _plan_handoff_validation(actions: list[PlanAction]) -> dict[str, Any]:
    rows = []
    for action in actions:
        markers = _scope_contract_markers(action.source_text)
        missing = [marker for marker in markers if marker not in [term.casefold() for term in action.preserve_terms]]
        if markers or missing:
            rows.append({
                "sentence_id": action.sentence_id,
                "source_scope_markers": markers,
                "missing_from_preserve_terms": missing,
                "status": "passed" if not missing else "failed",
            })
    return {
        "planner_to_writer_contract": "passed" if not any(row["missing_from_preserve_terms"] for row in rows) else "failed",
        "scope_rows": rows,
    }


def _author_proxy_context(scan: Scan, paragraph: Paragraph) -> dict[str, Any]:
    previous_paragraph = scan.paragraphs[paragraph.index - 1].text if paragraph.index > 0 else ""
    next_paragraph = scan.paragraphs[paragraph.index + 1].text if paragraph.index + 1 < len(scan.paragraphs) else ""
    context_text = "\n\n".join(part for part in [previous_paragraph, paragraph.text, next_paragraph] if part)
    paragraph_text = _strip_leading_heading(paragraph.text)
    return {
        "schema_version": "v6.author_proxy_context.v1",
        "scope": "selected_paragraph_plus_neighbors",
        "selected_paragraph_id": paragraph.id,
        "before_context": previous_paragraph[:600],
        "after_context": next_paragraph[:600],
        "source_terms": source_anchor_terms(paragraph_text, term_limit=32),
        "context_terms": source_anchor_terms(paragraph_text, term_limit=40),
        "neighbor_context_terms": source_anchor_terms(previous_paragraph + "\n" + next_paragraph, term_limit=40),
        "named_references": _named_references(context_text),
        "years": _years(context_text),
        "citation_spans": _citation_spans(context_text),
        "quoted_terms": _quoted_terms(context_text),
        "neighbor_bridge_anchors": _neighbor_bridge_anchors(previous_paragraph, next_paragraph),
        "author_proxy_instruction": (
            "Use these submitted context anchors to ground missing author/context bridges. "
            "Do not invent external names, years, citations, statistics, events, or institutions. "
            "If a bridge goes beyond these anchors, include it in author_review_items."
        ),
    }


def _named_references(text: str) -> list[str]:
    references: list[str] = []
    seen: set[str] = set()
    pattern = r"\b(?:[A-Z][A-Za-z0-9'’-]{2,}|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'’-]{2,}|[A-Z]{2,}))*\b"
    for match in re.finditer(pattern, str(text or "")):
        value = match.group(0).strip()
        if _is_sentence_start_common_word(text, match.start(), value):
            continue
        key = value.casefold()
        if key not in seen:
            references.append(value)
            seen.add(key)
        if len(references) >= 24:
            break
    return references


def _is_sentence_start_common_word(text: str, start: int, value: str) -> bool:
    previous = str(text or "")[:start].rstrip()
    at_sentence_start = not previous or previous[-1] in ".!?\n"
    words = value.split()
    if at_sentence_start and len(words) == 1 and not value.isupper():
        return True
    common_start_words = {
        "a", "an", "and", "another", "as", "because", "before", "but", "during",
        "for", "however", "if", "in", "many", "now", "overall", "so", "some",
        "that", "the", "these", "they", "this", "those", "today", "when", "while",
    }
    return at_sentence_start and len(words) == 1 and value.casefold() in common_start_words


def _years(text: str) -> list[str]:
    return _dedupe(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", str(text or "")))[:24]


def _citation_spans(text: str) -> list[str]:
    spans: list[str] = []
    for pattern in [
        r"\([A-Z][^()]{0,120}?(?:19|20)\d{2}[a-z]?[^()]{0,80}?\)",
        r"\bAccording to\s+[A-Z][^.]{0,120}?(?:19|20)\d{2}[a-z]?",
        r"\b[A-Z][A-Za-z'’-]+(?:\s+et al\.)?\s*\((?:19|20)\d{2}[a-z]?\)",
    ]:
        spans.extend(match.group(0).strip() for match in re.finditer(pattern, str(text or ""), flags=re.I))
    return _dedupe(spans)[:24]


def _quoted_terms(text: str) -> list[str]:
    values: list[str] = []
    patterns = [r"“([^”]{1,120})”", r"\"([^\"]{1,120})\"", r"‘([^’]{1,120})’", r"'([^']{1,120})'"]
    for pattern in patterns:
        values.extend(match.group(1).strip() for match in re.finditer(pattern, str(text or "")))
    return _dedupe(values)[:24]


def _neighbor_bridge_anchors(previous_paragraph: str, next_paragraph: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, text in [("previous_paragraph", previous_paragraph), ("next_paragraph", next_paragraph)]:
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or "").strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            terms = source_anchor_terms(sentence, term_limit=16, phrase_limit=2)
            if len(terms) < 3:
                continue
            if not _bridge_sentence_worth_carrying(sentence):
                continue
            rows.append({
                "source": source,
                "text": sentence[:260],
                "bridge_terms": terms[:12],
                "use": "ground an abstract paragraph claim with submitted neighboring context",
            })
            if len(rows) >= 6:
                return rows
    return rows


def _bridge_sentence_worth_carrying(sentence: str) -> bool:
    visible = str(sentence or "")
    if visible.count(",") >= 2:
        return True
    if len(re.findall(r"\b(?:also|but|because|although|while|whereas|therefore|so)\b", visible, flags=re.I)) >= 2:
        return True
    if len(re.findall(r"\b(?:and|or)\b", visible, flags=re.I)) >= 2:
        return True
    return False


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value).casefold()
        if value and key not in seen:
            out.append(value)
            seen.add(key)
    return out


def _planning_tags(tags: list[str], source_text: str) -> list[str]:
    rows = list(tags)
    stripped = _strip_leading_heading(source_text).strip()
    if re.match(r"^(this|that|it|they|these|those)\b", stripped, flags=re.I):
        if "predictable_start" not in rows:
            rows.append("predictable_start")
        if "context_anchor_gap" not in rows:
            rows.append("context_anchor_gap")
    return rows


def _ai_safe_route(
    paragraph: Paragraph,
    actions: list[PlanAction],
    context: dict[str, Any],
    paragraph_strategy: dict[str, Any],
) -> dict[str, Any]:
    dense_repair = paragraph_strategy.get("repair_unit") == "paragraph"
    return {
        "route_name": "grounded_source_beat_author_proxy_route",
        "objective": (
            "Produce grounded authorship by moving through submitted source beats, "
            "context anchors, and scoped claims instead of polished summary or generic explanation."
        ),
        "golden_route": {
            "compact_formula": "Scope it. Anchor it. Show it. Explain it. Close it.",
            "technical_formula": "narrow claim -> human/context anchor -> concrete evidence -> decompressed reasoning -> non-generic close",
            "core_rule": "A paragraph should show where the idea came from, what the writer noticed or used, how they interpreted it, and why it matters.",
            "question_rule": "What did the writer see, read, compare, struggle with, or decide, and why does that matter?",
        },
        "golden_route_questions": [
            {
                "question_id": "q_source_basis",
                "question": "What submitted material did the writer read, use, compare, notice, struggle with, or decide from?",
                "answer_source": "source_paragraph, neighboring context, citations, named references, quoted terms, and source vocabulary only",
                "writer_use": "Use the answer as the paragraph anchor before making a broad claim.",
            },
            {
                "question_id": "q_concrete_detail",
                "question": "Which concrete detail, source term, process step, example, or domain mechanic shows that basis?",
                "answer_source": "preserve_terms and available_author_proxy_anchors",
                "writer_use": "Use the answer to show evidence instead of abstract explanation.",
            },
            {
                "question_id": "q_reasoning_move",
                "question": "What did the writer interpret, compare, limit, or correct from that detail?",
                "answer_source": "finding_resolution_targets and local source beat relationship",
                "writer_use": "Use the answer to create a visible thinking path.",
            },
            {
                "question_id": "q_matter",
                "question": "Why does that matter for this paragraph's purpose?",
                "answer_source": "same paragraph purpose and submitted context only",
                "writer_use": "Use the answer for a careful close without a generic takeaway.",
            },
        ],
        "human_authorship_route": [
            {
                "step": "scope_it",
                "instruction": "Start from a narrowed claim, condition, setting, source relation, or paragraph-specific pressure instead of a broad universal opening.",
            },
            {
                "step": "anchor_it",
                "instruction": "Attach the claim to context already present in the paragraph, neighboring paragraphs, citations, named references, quoted terms, or source vocabulary.",
            },
            {
                "step": "show_it",
                "instruction": "Carry concrete source material, observed process detail, example terms, or domain mechanics from the submitted content.",
            },
            {
                "step": "explain_it",
                "instruction": "Explain the source relationship one idea at a time, including contrast, correction, limitation, or cause when the source supports it.",
            },
            {
                "step": "close_it",
                "instruction": "Close by tying the beat back to the paragraph purpose without adding a generic takeaway or stronger claim than the submitted material supports.",
            },
        ],
        "safe_route_formula": "scope it -> anchor it -> show it -> explain it -> close it",
        "origin_paragraph_context": {
            "paragraph_id": paragraph.id,
            "source_beats": [
                {
                    "sentence_id": action.sentence_id,
                    "local_anchor_terms": action.preserve_terms,
                    "findings": action.tags,
                }
                for action in actions
            ],
            "rule": "Every rewritten beat must remain traceable to one origin source beat and its local_anchor_terms without copying the original sentence wording.",
        },
        "paragraph_repair_unit": paragraph_strategy.get("repair_unit", "sentence_cluster"),
        "dense_paragraph_plan": paragraph_strategy.get("dense_paragraph_plan", {}),
        "coverage_beats": _coverage_beats(actions, dense_repair=dense_repair),
        "construction_recipes": [_construction_recipe(action, context) for action in actions if action.tags],
        "author_route_questions": _author_route_questions(actions, context),
        "route_order": [
            "start from the paragraph's concrete source pressure or condition",
            "repair dense findings as paragraph flow when many adjacent sentences are flagged",
            "repair affected source beats in original order without making one sentence per finding",
            "split packed lists only when the source relationship needs separation",
            "scope evaluative claims with submitted context anchors",
            "end on the original source beat, not a new takeaway",
        ],
        "available_author_proxy_anchors": {
            "named_references": context.get("named_references", []),
            "years": context.get("years", []),
            "citation_spans": context.get("citation_spans", []),
            "quoted_terms": context.get("quoted_terms", []),
            "nearby_context_terms": context.get("context_terms", [])[:20],
            "neighbor_bridge_anchors": context.get("neighbor_bridge_anchors", [])[:4],
        },
        "sentence_route_steps": [_route_step(action, context) for action in actions],
        "forbidden_ai_shapes": [
            "generic opening claim followed by smooth explanation",
            "same route with elevated synonyms",
            "compressed list sentence",
            "one-item-per-sentence mechanical list",
            "one short sentence for every scanner finding",
            "paragraph rebuilt as repeated subject-verb fragments",
            "unsupported final takeaway",
            "new external fact, citation, statistic, date, named event, or institution",
            "polished abstract labels that replace source wording",
        ],
        "author_proxy_boundary": (
            "Author-Proxy may add bridge wording only from available_author_proxy_anchors or narrow inference from the submitted paragraph. "
            "Any bridge beyond those anchors must appear in author_review_items."
        ),
    }


def _route_step(action: PlanAction, context: dict[str, Any]) -> dict[str, Any]:
    if action.method == "list_rhythm_rebuild":
        ai_safe_move = "split packed meaning anchors into two short connected prose beats without copying the listed wording shape"
    elif action.method == "claim_scope_repair":
        ai_safe_move = "narrow the claim inside this beat using submitted anchors and context, without adding a new conclusion"
    elif action.method == "context_anchor_bridge":
        ai_safe_move = "replace the vague reference with a nearby source antecedent, then keep the sentence's original role"
    elif action.method == "author_proxy_bridge":
        ai_safe_move = "add reviewable author-proxy bridge wording inside this source beat using submitted context anchors"
    elif action.method == "route_rebuild":
        ai_safe_move = "change opener or clause order while preserving the source meaning"
    else:
        ai_safe_move = "carry this meaning beat and avoid polishing"
    return {
        "sentence_id": action.sentence_id,
        "findings": action.tags,
        "ai_safe_move": ai_safe_move,
        "construction_recipe_id": f"{action.sentence_id}_recipe",
        "finding_resolution_targets": _finding_resolution_targets(action.tags),
        "preserve_terms": action.preserve_terms,
        "allowed_context_anchors": _allowed_context_anchors(action, context),
        "do_not": action.do_not,
    }


def _coverage_beats(actions: list[PlanAction], *, dense_repair: bool = False) -> list[dict[str, Any]]:
    beats: list[dict[str, Any]] = []
    for action in actions:
        source_text = _strip_leading_heading(action.source_text)
        split_coverage = _should_split_coverage(action, dense_repair=dense_repair)
        phrases = _coverage_phrases(source_text) if split_coverage else [_safe_coverage_phrase(source_text, action.tags)]
        if split_coverage:
            for index, phrase in enumerate(phrases or [source_text], start=1):
                beats.append({
                    "beat_id": f"{action.sentence_id}_b{index:02d}",
                    "source_sentence_id": action.sentence_id,
                    "finding_tags": action.tags,
                    "construction_recipe_id": f"{action.sentence_id}_recipe",
                    "construction_recipe": _beat_construction_recipe(action.tags),
                    "coverage_terms": source_anchor_terms(phrase, term_limit=24),
                    "grouped_relation": " | " in phrase,
                    "polarity_markers": _polarity_markers(phrase),
                    "starter_terms": _starter_terms(phrase, action.tags),
                    "generation_duty": "carry this meaning anchor in its own ordinary prose beat without preserving original phrase order",
                    "merge_rule": "merge only if it does not create a list or overload the sentence",
                })
        else:
            phrase = phrases[0] if phrases else action.source_text
            beats.append({
                "beat_id": f"{action.sentence_id}_b01",
                "source_sentence_id": action.sentence_id,
                "finding_tags": action.tags,
                "construction_recipe_id": f"{action.sentence_id}_recipe",
                "construction_recipe": _beat_construction_recipe(action.tags),
                "coverage_terms": source_anchor_terms(phrase, term_limit=24),
                "grouped_relation": " | " in phrase,
                "polarity_markers": _polarity_markers(phrase),
                "starter_terms": _starter_terms(phrase, action.tags),
                "generation_duty": (
                    "carry this source-supported idea inside the paragraph flow without preserving original wording"
                    if dense_repair
                    else "carry this source-supported idea with a scoped route and clear anchor without preserving original wording"
                ),
                "merge_rule": (
                    "merge with neighboring beats when the ideas are semantically dependent or repeated-shape fragments"
                    if dense_repair
                    else "merge only if the source idea stays visible and grounded"
                ),
            })
    return beats


def _construction_recipe(action: PlanAction, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "recipe_id": f"{action.sentence_id}_recipe",
        "source_sentence_id": action.sentence_id,
        "finding_tags": action.tags,
        "repair_sequence": _repair_sequence(action.tags),
        "coverage_terms": action.preserve_terms,
        "build_route": _recipe_route(action.tags),
        "build_steps": _recipe_steps(action.tags),
        "positive_pattern": _positive_pattern(action.tags),
        "author_proxy_move": _author_proxy_move(action.tags, context),
    }


def _repair_sequence(tags: list[str]) -> list[dict[str, Any]]:
    priority = [
        ("predictable_start", "OPEN_WITH_SPECIFIC_OBSERVATION", "change the paragraph route before polishing wording"),
        ("context_author_anchor_gap", "ADD_CONTEXT_ANCHOR", "ground the beat in submitted author, course, setting, source, or task anchors"),
        ("sentence_overload", "ONE_FUNCTION_PER_SENTENCE", "split by source function such as claim, example, condition, or consequence"),
        ("packed_list", "GROUP_LIST_ITEMS", "turn packed lists into grouped meaning or action-based prose"),
        ("paragraph_rhythm", "SENTENCE_WEIGHT_BALANCING", "vary sentence job and weight after route and coverage are stable"),
    ]
    tag_set = set(tags)
    rows: list[dict[str, Any]] = []
    for repair_class, operator, duty in priority:
        if repair_class == "context_author_anchor_gap":
            active = bool(tag_set & {"context_anchor_gap", "author_anchor_gap", "unsupported_claim_gap", "broad_claim"})
        else:
            active = repair_class in tag_set
        if active:
            rows.append({"repair_class": repair_class, "operator": operator, "writer_duty": duty})
    return rows


def _author_route_questions(actions: list[PlanAction], context: dict[str, Any]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for action in actions:
        if not action.tags:
            continue
        question = _route_question(action.tags)
        questions.append({
            "question_id": f"{action.sentence_id}_q01",
            "source_sentence_id": action.sentence_id,
            "finding_tags": action.tags,
            "question": question,
            "answer_basis_terms": _question_basis_terms(action, context),
            "writer_duty": _question_writer_duty(action.tags),
            "author_proxy_policy": _question_author_proxy_policy(action.tags),
        })
    return questions


def _route_question(tags: list[str]) -> str:
    tag_set = set(tags)
    if "packed_list" in tag_set or "sentence_overload" in tag_set:
        return "Which source items belong together, and what plain relationship connects each group to the paragraph purpose?"
    if "context_anchor_gap" in tag_set and tag_set & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"}:
        return "Which earlier actor, object, process, condition, or contrast anchors this sentence, and what submitted or reviewable bridge narrows the claim?"
    if "context_anchor_gap" in tag_set:
        return "Which earlier actor, object, process, condition, or contrast does this sentence point back to?"
    if tag_set & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"}:
        return "What submitted anchor, scope limit, or reviewable author-proxy bridge makes this claim belong to the writer's paragraph?"
    if "semantic_bridge_gap" in tag_set:
        return "What source-to-claim link is missing between this beat and the paragraph's next idea?"
    if "transition_stack" in tag_set:
        return "Which source actor, object, or condition can carry the transition without a discourse marker?"
    if "predictable_start" in tag_set or "paragraph_rhythm" in tag_set:
        return "Which concrete source noun, condition, or comparison can open this beat instead of a predictable opener?"
    if "paraphrase_smoothing" in tag_set or "abstract_density" in tag_set:
        return "Which submitted source terms can replace the polished or abstract wrapper?"
    return "What source-supported relationship does this beat need to show, and why does it matter here?"


def _question_basis_terms(action: PlanAction, context: dict[str, Any]) -> list[str]:
    selected_terms = {term.casefold() for term in context.get("source_terms", [])}
    anchors = [
        *action.preserve_terms,
        *context.get("quoted_terms", [])[:3],
        *context.get("citation_spans", [])[:3],
        *_selected_named_references(context, selected_terms)[:3],
    ]
    if set(action.tags) & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim", "context_anchor_gap"}:
        anchors.extend(_neighbor_bridge_terms(context, selected_terms)[:5])
    return _dedupe(anchors)[:18]


def _selected_named_references(context: dict[str, Any], selected_terms: set[str]) -> list[str]:
    rows: list[str] = []
    for value in context.get("named_references", []):
        terms = source_terms(str(value), limit=8)
        if any(term.casefold() in selected_terms for term in terms) or not _reference_in_neighbor_list(context, str(value)):
            rows.append(str(value))
    return rows


def _neighbor_bridge_terms(context: dict[str, Any], selected_terms: set[str]) -> list[str]:
    named_terms = {
        term.casefold()
        for value in context.get("named_references", [])
        for term in source_terms(str(value), limit=8)
        if term.casefold() not in selected_terms
    }
    list_terms = _neighbor_list_terms(context) - selected_terms
    rows: list[str] = []
    for value in context.get("context_terms", []):
        key = str(value).casefold()
        if key in named_terms or key in list_terms:
            continue
        rows.append(str(value))
    return rows


def _neighbor_list_terms(context: dict[str, Any]) -> set[str]:
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


def _reference_in_neighbor_list(context: dict[str, Any], value: str) -> bool:
    needle = str(value or "").strip()
    if not needle:
        return False
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


def _question_writer_duty(tags: list[str]) -> str:
    tag_set = set(tags)
    if "packed_list" in tag_set or "sentence_overload" in tag_set:
        return "Answer with grouped prose beats, not a comma list and not repeated one-item sentences."
    if "context_anchor_gap" in tag_set and tag_set & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"}:
        return "Answer by naming the antecedent, narrowing the claim, and marking any inferred bridge inside author_review_items."
    if "context_anchor_gap" in tag_set:
        return "Answer by naming the antecedent before the claim continues."
    if tag_set & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"}:
        return "Answer by narrowing the claim or adding reviewable bridge wording inside the same beat."
    if "predictable_start" in tag_set or "paragraph_rhythm" in tag_set:
        return "Answer by changing the sentence entry point and neighboring rhythm."
    return "Answer inside the replacement sentence, not in notes."


def _question_author_proxy_policy(tags: list[str]) -> str:
    if set(tags) & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim", "context_anchor_gap"}:
        return "Author-Proxy may add a narrow bridge from answer_basis_terms; bridges beyond those terms must be listed in author_review_items."
    return "Use submitted source terms only unless a narrow bridge is needed for readability and marked for review."


def _beat_construction_recipe(tags: list[str]) -> dict[str, Any]:
    return {
        "build_route": _recipe_route(tags),
        "build_steps": _recipe_steps(tags),
        "positive_pattern": _positive_pattern(tags),
    }


def _recipe_route(tags: list[str]) -> str:
    tag_set = set(tags)
    if "packed_list" in tag_set:
        return "turn packed items into separate source-relation beats before any interpretation"
    if tag_set & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"}:
        return "write the concrete source relation first, then narrow the claim or mark reviewable author-proxy bridge"
    if "context_anchor_gap" in tag_set:
        return "replace the vague pointer with the concrete antecedent before continuing the sentence role"
    if "paraphrase_smoothing" in tag_set:
        return "write a plain source-level relation instead of polished explanation"
    if "citation_anchor" in tag_set:
        return "keep the source attribution phrase or submitted parenthetical citation span next to the supported claim; do not convert citations into new reporting verbs"
    if "sentence_overload" in tag_set:
        return "split the source relationship into short ordered beats"
    if "predictable_start" in tag_set:
        return "start from a concrete source term or condition, not a timeline or pronoun opener"
    return "carry the source meaning through a plain sentence route"


def _recipe_steps(tags: list[str]) -> list[str]:
    tag_set = set(tags)
    steps: list[str] = ["choose a concrete starter from starter_terms or coverage_terms"]
    if "packed_list" in tag_set:
        steps.extend([
            "write one sentence for this grouped beat",
            "keep later grouped beats visible in the final text",
            "connect grouped beats through ordinary source relations, not a comma list",
        ])
    if "context_anchor_gap" in tag_set:
        steps.append("name the previous concrete actor, object, process, or condition instead of this/that/it/they")
    if tag_set & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"}:
        steps.extend([
            "state the observable source relation before the evaluative word",
            "narrow certainty with submitted scope or add author_review_items for inferred bridge wording",
        ])
    if "paraphrase_smoothing" in tag_set:
        steps.append("remove polished adjectives and keep source-level pressure")
    if "citation_anchor" in tag_set:
        steps.append("keep the source attribution or citation parenthetical with the supported claim and do not write reports, observed, documented, confirms, or proves")
    if "sentence_overload" in tag_set:
        steps.append("split overload into adjacent sentences with different starters")
    return steps


def _positive_pattern(tags: list[str]) -> str:
    tag_set = set(tags)
    if "packed_list" in tag_set:
        return "Pair related anchors in one relation sentence. Put the next related pair in a separate relation sentence."
    if tag_set & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"}:
        return "Start with the submitted source anchor, then state only the specific relation it supports. Mark any wider bridge for author review."
    if "context_anchor_gap" in tag_set:
        return "Name the concrete antecedent first, then continue the local relation."
    if "paraphrase_smoothing" in tag_set:
        return "Use the submitted actor or object with a plain local action."
    return "Use a submitted source anchor, a plain relation, and a scoped continuation."


def _author_proxy_move(tags: list[str], context: dict[str, Any]) -> str:
    if not (set(tags) & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"}):
        return "not required unless a source relation is missing"
    anchors = _dedupe([
        *context.get("quoted_terms", [])[:2],
        *context.get("citation_spans", [])[:2],
        *context.get("named_references", [])[:2],
        *context.get("context_terms", [])[:4],
    ])
    if anchors:
        return "use submitted anchors first: " + ", ".join(anchors)
    return "use a narrow inferred bridge only when recorded in author_review_items"


def _should_split_coverage(action: PlanAction, *, dense_repair: bool = False) -> bool:
    if dense_repair:
        return False
    return (
        action.method == "list_rhythm_rebuild"
        or "broad_claim" in action.tags
        or "context_anchor_gap" in action.tags
        or "sentence_overload" in action.tags
    )


def _coverage_phrases(text: str) -> list[str]:
    visible = re.sub(r"\s+", " ", str(text or "")).strip()
    visible = re.sub(r"^(in|during|after|before|through|at|from|to|for|when|while)\s+[^,]{1,80},\s+", "", visible, flags=re.I)
    visible = _safe_coverage_phrase(visible, ["predictable_start", "context_anchor_gap"])
    visible = visible.rstrip(".!?;:")
    if not visible:
        return []
    if re.search(r",\s+not\b", visible, flags=re.I):
        return [visible]
    because_parts = _split_because_parts(visible)
    if len(because_parts) > 1:
        return because_parts
    relation_parts = _semantic_relation_phrases(visible)
    if len(relation_parts) > 1:
        return relation_parts
    visible = re.sub(r",?\s+\bbut\b\s+", ". ", visible, flags=re.I)
    normalized = re.sub(r",\s+(and|or)\s+", ", ", visible, flags=re.I)
    parts = [part.strip(" ,;:") for part in re.split(r"\.\s+|,\s+|;\s+", normalized) if part.strip(" ,;:")]
    if len(parts) <= 1:
        parts = [part.strip(" ,;:") for part in re.split(r"\s+\band\b\s+", visible, flags=re.I) if part.strip(" ,;:")]
    parts = [_safe_coverage_phrase(part, ["predictable_start", "context_anchor_gap"]) for part in parts]
    parts = [part for part in parts if part and source_anchor_terms(part, term_limit=4)]
    parts = _merge_singleton_relation_parts(parts)
    parts = _propagate_not_always(parts)
    parts = _group_dense_list_parts(parts)
    return parts if len(parts) > 1 else [visible]


def _semantic_relation_phrases(text: str) -> list[str]:
    visible = str(text or "").strip(" ,;:")
    if not visible:
        return []
    parts = _split_before_following_comparative_gerund(visible)
    if len(parts) == 1:
        parts = [visible]
    expanded: list[str] = []
    for part in parts:
        expanded.extend(_split_condition_or_consequence(part))
    return [_safe_coverage_phrase(part, []) for part in expanded if part.strip(" ,;:")]


def _split_before_following_comparative_gerund(text: str) -> list[str]:
    if not re.search(r"\bis\s+(?:far\s+)?more\s+\w+\s+than\b", text, flags=re.I):
        return [text]
    parts = re.split(
        r"\s+(?=[A-Za-z]+ing\b[^.]{0,140}?\bis\s+(?:far\s+)?more\s+\w+\s+than\b)",
        text,
        maxsplit=1,
        flags=re.I,
    )
    return [part.strip(" ,;:") for part in parts if part.strip(" ,;:")]


def _split_condition_or_consequence(text: str) -> list[str]:
    parts = re.split(r",\s+(?=(?:only\s+by|when|rather\s+than|instead\s+of)\b)", text, maxsplit=1, flags=re.I)
    return [part.strip(" ,;:") for part in parts if part.strip(" ,;:")]


def _split_because_parts(text: str) -> list[str]:
    parts = [part.strip(" ,;:") for part in re.split(r"\s+\bbecause\b\s+", str(text or ""), maxsplit=1, flags=re.I)]
    if len(parts) != 2 or not all(parts):
        return [str(text or "").strip()]
    return [_safe_coverage_phrase(part, ["predictable_start", "context_anchor_gap"]) for part in parts]


def _group_dense_list_parts(parts: list[str]) -> list[str]:
    if len(parts) < 4:
        return parts
    if any(re.search(r"\b(?:not always|not only|rather than|instead of|without)\b", part, flags=re.I) for part in parts):
        return parts
    groups: list[str] = []
    index = 0
    while index < len(parts):
        group = parts[index:index + 2]
        groups.append(" | ".join(group))
        index += 2
    return groups


def _merge_singleton_relation_parts(parts: list[str]) -> list[str]:
    rows: list[str] = []
    for part in parts:
        terms = source_terms(part, limit=4)
        if (
            rows
            and len(terms) == 1
            and (re.fullmatch(r"[A-Z][A-Z0-9]{1,}", terms[0]) or re.fullmatch(r"\d+(?:\.\d+)?", terms[0]))
        ):
            rows[-1] = f"{rows[-1]} | {part}"
            continue
        rows.append(part)
    return rows


def _propagate_not_always(parts: list[str]) -> list[str]:
    rows: list[str] = []
    limited: list[str] = []
    for part in parts:
        text = str(part).strip()
        if re.search(r"\bnot always\b", text, flags=re.I):
            limited.append(text)
            continue
        if limited:
            limited.append(text)
            continue
        rows.append(text)
    if limited:
        rows.append(" | ".join(limited))
    return rows


def _polarity_markers(text: str) -> list[str]:
    markers: list[str] = []
    patterns = [
        r"\bno longer\b",
        r"\bdoes not only\b",
        r"\bdo not only\b",
        r"\bnot only\b",
        r"\bnot always\b",
        r"\bmore\s+\w+\s+than\b",
        r"\brather than\b",
        r"\binstead of\b",
        r"\bwithout\b",
        r"\bnot\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, str(text or ""), flags=re.I):
            marker = match.group(0)
            if any(marker.casefold() in existing.casefold() for existing in markers):
                continue
            markers.append(marker)
    return _dedupe(markers)


def _starter_terms(phrase: str, tags: list[str]) -> list[str]:
    anchor_terms = source_anchor_terms(phrase, term_limit=8)
    if anchor_terms and " " in anchor_terms[0]:
        return [anchor_terms[0]]
    terms = source_terms(phrase, limit=8)
    contrast_left_terms = _left_side_contrast_terms(phrase)
    risky_route = bool({"predictable_start", "broad_claim", "context_anchor_gap", "paragraph_rhythm"} & set(tags))
    head_terms = {term.casefold() for term in terms[:2]} if risky_route and len(terms) >= 4 else set()
    weak_starters = {
        "today", "that", "this", "they", "these", "those", "overall", "however", "therefore",
        "model", "result", "thing", "issue", "longer", "fully", "reflects", "exists",
        "mostly", "around", "faster", "comfortably", "rather", "because", "does",
        "also", "beginning", "call", "during", "each", "know", "learn", "one", "quite",
        "some", "what", "will",
    }
    candidates = [
        term for term in terms
        if term.casefold() not in weak_starters
        and not term.casefold().endswith(("ed", "ly"))
    ]
    if contrast_left_terms:
        candidates = [
            term for term in contrast_left_terms
            if term.casefold() not in weak_starters
            and not term.casefold().endswith(("ed", "ly"))
        ] or contrast_left_terms
    if risky_route:
        later_candidates = [term for term in candidates if term.casefold() not in head_terms]
        if later_candidates:
            candidates = later_candidates
    non_ing = [term for term in candidates if not term.casefold().endswith("ing")]
    if non_ing:
        candidates = non_ing
    return candidates[:4]


def _left_side_contrast_terms(phrase: str) -> list[str]:
    match = re.search(r"\b(?:rather than|instead of)\b", str(phrase or ""), flags=re.I)
    if not match:
        return []
    left = source_terms(str(phrase)[:match.start()], limit=6)
    return left[-3:]


def _safe_coverage_phrase(text: str, tags: list[str]) -> str:
    visible = re.sub(r"\s+", " ", str(text or "")).strip().rstrip(".!?;:")
    if not visible:
        return ""
    if "predictable_start" in tags or "context_anchor_gap" in tags:
        visible = re.sub(r"^today[’']?s\s+", "", visible, flags=re.I)
        visible = re.sub(r"^(today|now|overall),?\s+", "", visible, flags=re.I)
        visible = re.sub(r"^in\s+the\s+past,?\s+", "", visible, flags=re.I)
        visible = re.sub(r"^(this|that|it|they|these|those)\s+", "", visible, flags=re.I)
    return visible[:1].upper() + visible[1:] if visible else ""


def _finding_resolution_targets(tags: list[str]) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for tag in tags:
        resolution = _finding_resolution_target(tag)
        if resolution:
            targets.append({"finding_tag": tag, **resolution})
    return targets


def _finding_resolution_target(tag: str) -> dict[str, str] | None:
    targets = {
        "predictable_start": {
            "safe_route": "Open from a preserved source noun, named reference, quoted term, or local condition instead of a generic demonstrative or timeline opener.",
            "unsafe_shape": "same opener route with synonyms, or starts like this/that/it/they/today/now/in the past.",
            "proof_of_execution": "The replacement first words identify source material, not a generic transition.",
        },
        "paragraph_rhythm": {
            "safe_route": "Vary sentence starts and clause weight across neighboring beats while keeping source order readable.",
            "unsafe_shape": "several adjacent sentences with the same subject-verb frame or equal short-item rhythm.",
            "proof_of_execution": "Neighboring replacements do not share the same opening frame or one-beat cadence.",
        },
        "packed_list": {
            "safe_route": "Turn a three-or-more item list into two connected prose beats that keep the items in source context.",
            "unsafe_shape": "one long comma list, a semicolon list, or repeated one-item sentences.",
            "proof_of_execution": "List items are redistributed across connected prose without creating a mechanical item chain.",
        },
        "sentence_overload": {
            "safe_route": "Split the overloaded beat at a real source relationship, preserving all source ideas.",
            "unsafe_shape": "shorter summary that drops ideas, or a longer sentence with the same packed structure.",
            "proof_of_execution": "Every original idea remains present in a less packed clause route.",
        },
        "context_anchor_gap": {
            "safe_route": "Name the local antecedent from the paragraph or nearby context before making the claim.",
            "unsafe_shape": "this/that/it/they points to an unclear or generic idea.",
            "proof_of_execution": "The sentence contains a concrete source antecedent, not only a pronoun.",
        },
        "author_anchor_gap": {
            "safe_route": "Add author-proxy bridge wording from submitted context inside the same beat, with review provenance when inferred.",
            "unsafe_shape": "detached universal conclusion or unsupported evaluator voice.",
            "proof_of_execution": "The claim is tied to submitted anchors or marked for author review.",
        },
        "unsupported_claim_gap": {
            "safe_route": "Reduce certainty, narrow the claim scope, or add reviewable support from submitted context inside the beat.",
            "unsafe_shape": "new verified fact, citation, statistic, event, institution, or stronger claim than source support.",
            "proof_of_execution": "The claim has a visible scope limit or reviewable bridge provenance.",
        },
        "broad_claim": {
            "safe_route": "Limit the claim with a source condition, contrast, actor, or setting already present in the paragraph/context.",
            "unsafe_shape": "broad universal statement that reads like a polished essay takeaway.",
            "proof_of_execution": "The replacement states the boundary of the claim.",
        },
        "abstract_density": {
            "safe_route": "Replace abstract labels with submitted source terms and concrete relationships between them.",
            "unsafe_shape": "new polished abstraction that sounds broader than the source.",
            "proof_of_execution": "The sentence carries source terms rather than abstract labels.",
        },
        "semantic_bridge_gap": {
            "safe_route": "Make the missing source-to-claim link explicit using nearby source/context anchors.",
            "unsafe_shape": "therefore/this means/this shows without the bridge inside the sentence.",
            "proof_of_execution": "The relationship between source beat and claim is stated directly.",
        },
        "transition_stack": {
            "safe_route": "Remove stacked transition wording and connect through the source actor, object, or condition.",
            "unsafe_shape": "however/additionally/therefore-style scaffolding.",
            "proof_of_execution": "The transition is carried by source content rather than discourse markers.",
        },
        "citation_anchor": {
            "safe_route": "Keep submitted parenthetical citations close to the exact claim while avoiding citation-led wrapping.",
            "unsafe_shape": "citation opener or standalone citation sentence with new reporting verbs.",
            "proof_of_execution": "The citation remains parenthetical and attached to a specific supported source relation.",
        },
        "paraphrase_smoothing": {
            "safe_route": "Use source-level phrasing with uneven sentence pressure rather than cleaner academic paraphrase.",
            "unsafe_shape": "same meaning rewritten as smoother, more formal prose.",
            "proof_of_execution": "The route changes through source beats, not vocabulary polish.",
        },
    }
    return targets.get(tag)


def _allowed_context_anchors(action: PlanAction, context: dict[str, Any]) -> list[str]:
    anchors = [term for term in action.preserve_terms]
    anchors.extend(context.get("quoted_terms", [])[:4])
    anchors.extend(context.get("citation_spans", [])[:4])
    anchors.extend(context.get("named_references", [])[:6])
    return _dedupe(anchors)[:16]


def _paragraph_strategy(paragraph: Paragraph, actions: list[PlanAction], context: dict[str, Any]) -> dict[str, Any]:
    affected = [action for action in actions if action.tags]
    tag_counts = Counter(tag for action in affected for tag in action.tags)
    methods = {action.method for action in affected}
    dense_plan = _dense_paragraph_plan(paragraph, actions, affected)
    grounding_plan = _author_proxy_grounding_plan(paragraph, actions, context)
    failed_parts: list[str] = []
    if "list_rhythm_rebuild" in methods:
        failed_parts.append("packed list beats create dense, predictable rhythm")
    if "claim_scope_repair" in methods:
        failed_parts.append("evaluative or broad claims need scoped support inside their own source beat")
    if "route_rebuild" in methods:
        failed_parts.append("predictable openers and clause routes repeat across the paragraph")
    if "context_anchor_bridge" in methods:
        failed_parts.append("this/that/it references need a clearer source-grounded antecedent")
    if "author_proxy_bridge" in methods:
        failed_parts.append("missing author or context anchor may need reviewable author-proxy wording")
    if grounding_plan["required"]:
        failed_parts.append("abstract high-anchor paragraph needs neighbor-context grounding instead of synonym rotation")
    failed_route = "; ".join(failed_parts) if failed_parts else "paragraph needs light route variation while preserving source meaning"
    return {
        "dominant_findings": [tag for tag, _count in tag_counts.most_common()],
        "affected_sentence_ids": [action.sentence_id for action in affected],
        "repair_unit": dense_plan["repair_unit"],
        "finding_density": dense_plan["finding_density"],
        "target_sentence_range": dense_plan["target_sentence_range"],
        "dense_paragraph_plan": dense_plan,
        "author_proxy_grounding": grounding_plan,
        "failed_route": failed_route,
        "replacement_route": (
            (
                "Repair this as one paragraph route because many adjacent source sentences are flagged. "
                "Treat sentence findings as symptoms of paragraph flow, preserve core source terms, "
                "merge semantically dependent short beats, and avoid one sentence per finding."
            )
            if dense_plan["repair_unit"] == "paragraph"
            else (
                "Repair affected sentences as one paragraph route in source order. "
                "Keep every source beat distinct, preserve core terms, redistribute packed lists without compression, "
                "and scope evaluative claims inside the same source beat."
            )
        ),
        "writer_instruction": (
            (
                "Use affected sentence actions as diagnostic evidence for a paragraph-level rewrite. "
                "Do not create a short-sentence chain, do not make every finding a separate sentence, and do not append a new takeaway. "
                "If author-proxy wording is needed, place it inside the relevant source beat and mark it for review."
                + (" Use neighbor-context grounding because the source paragraph lacks concrete anchors." if grounding_plan["required"] else "")
            )
            if dense_plan["repair_unit"] == "paragraph"
            else (
                "Use the affected sentence actions as a mapped source-beat plan, not isolated synonym edits. "
                "Do not append a new takeaway; if author-proxy wording is needed, place it inside the relevant source beat and mark it for review."
                + (" Use neighbor-context grounding because the source paragraph lacks concrete anchors." if grounding_plan["required"] else "")
            )
        ),
    }


def _author_proxy_grounding_plan(paragraph: Paragraph, actions: list[PlanAction], context: dict[str, Any]) -> dict[str, Any]:
    bridge_anchors = _grounding_bridge_anchors(paragraph, actions, context)
    tag_set = {tag for action in actions for tag in action.tags}
    quoted = _quoted_terms(paragraph.text)
    starts_with_context_pointer = bool(re.match(r"^\s*(this|that|these|those|it|they)\b", paragraph.text, flags=re.I))
    anchor_gap_tags = {"author_anchor_gap", "unsupported_claim_gap", "broad_claim", "context_anchor_gap", "semantic_bridge_gap"}
    has_anchor_gap = bool(tag_set & anchor_gap_tags)
    local_anchor_count = len(source_anchor_terms(_strip_leading_heading(paragraph.text), term_limit=16))
    abstract_or_pointer = starts_with_context_pointer or "semantic_bridge_gap" in tag_set
    claim_needs_grounding = (
        bool(tag_set & {"author_anchor_gap", "unsupported_claim_gap"})
        and (starts_with_context_pointer or local_anchor_count <= 12)
    ) or ("broad_claim" in tag_set and (local_anchor_count <= 8 or "context_anchor_gap" in tag_set))
    required = bool(
        bridge_anchors
        and has_anchor_gap
        and (
            quoted
            or abstract_or_pointer
            or claim_needs_grounding
        )
    )
    return {
        "required": required,
        "reason": (
            _author_proxy_grounding_reason(tag_set, quoted=quoted, starts_with_context_pointer=starts_with_context_pointer, local_anchor_count=local_anchor_count)
            if required
            else ""
        ),
        "bridge_anchors": bridge_anchors,
        "writer_instruction": (
            "Use one submitted bridge anchor to ground the abstract or unsupported claim, then mark the inferred bridge in author_review_items."
            if required
            else ""
        ),
        "review_rule": (
            "Any wording pulled from bridge anchors must be listed in author_review_items because it is author-proxy grounding, not direct source-paragraph wording."
            if required
            else ""
        ),
    }


def _grounding_bridge_anchors(paragraph: Paragraph, actions: list[PlanAction], context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = list(context.get("neighbor_bridge_anchors", [])[:4])
    rows.extend(_local_bridge_anchors(paragraph, actions))
    return rows[:6]


def _local_bridge_anchors(paragraph: Paragraph, actions: list[PlanAction]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gap_tags = {"author_anchor_gap", "unsupported_claim_gap", "broad_claim", "context_anchor_gap", "semantic_bridge_gap"}
    for action in actions:
        terms = source_anchor_terms(_strip_leading_heading(action.source_text), term_limit=16, phrase_limit=2)
        if len(terms) < 3:
            continue
        if set(action.tags) & gap_tags and len(terms) < 5:
            continue
        rows.append({
            "source": "selected_paragraph",
            "text": action.source_text[:260],
            "bridge_terms": terms[:12],
            "use": "ground an anchor-gap claim with submitted same-paragraph context",
        })
        if len(rows) >= 4:
            break
    return rows


def _author_proxy_grounding_reason(
    tag_set: set[str],
    *,
    quoted: list[str],
    starts_with_context_pointer: bool,
    local_anchor_count: int,
) -> str:
    if quoted and (starts_with_context_pointer or "context_anchor_gap" in tag_set):
        return "The paragraph depends on protected quoted anchors and an abstract context pointer, so source-only paraphrase is likely to stay generic."
    if "semantic_bridge_gap" in tag_set:
        return "The paragraph has a missing source-to-claim bridge, so the rewrite must ground the bridge with submitted anchors and mark reviewable inference."
    if tag_set & {"author_anchor_gap", "unsupported_claim_gap"}:
        return "The paragraph contains an evaluative or unsupported claim with limited local anchors, so the rewrite must use author-proxy grounding instead of a detached conclusion."
    if "broad_claim" in tag_set and local_anchor_count <= 10:
        return "The paragraph contains a broad claim with limited local anchors, so the rewrite must narrow it through submitted bridge anchors."
    return "The paragraph has an anchor gap that needs submitted bridge grounding."


def _dense_paragraph_plan(
    paragraph: Paragraph,
    actions: list[PlanAction],
    affected: list[PlanAction],
) -> dict[str, Any]:
    sentence_count = max(1, len(actions))
    affected_count = len(affected)
    density = affected_count / sentence_count
    max_cluster = _max_adjacent_affected_cluster(actions)
    tag_counts = Counter(tag for action in affected for tag in action.tags)
    semantic_roles = _semantic_role_map(actions)
    repair_unit = "paragraph" if (
        sentence_count >= 3
        and affected_count >= 3
        and (density >= 0.5 or max_cluster >= 3)
    ) else "sentence_cluster"
    low = max(1, min(sentence_count, affected_count or sentence_count))
    high = max(low, min(sentence_count + 2, int(sentence_count * 1.35) + 1))
    return {
        "repair_unit": repair_unit,
        "finding_density": round(density, 3),
        "affected_sentence_count": affected_count,
        "source_sentence_count": sentence_count,
        "max_adjacent_affected_cluster": max_cluster,
        "target_sentence_range": f"{low} to {high} ordinary sentences",
        "semantic_role_map": semantic_roles,
        "human_route": _human_route(actions, semantic_roles),
        "dominant_flow_problem": _dominant_flow_problem(tag_counts),
        "finding_interpretation_rule": (
            "Scanner findings are paragraph-flow symptoms. Preserve traceability, but do not allocate one final sentence to each finding."
        ),
        "flow_plan": _paragraph_flow_plan(paragraph, actions, affected, tag_counts),
        "merge_policy": (
            "Merge only semantically dependent or mechanically repetitive short beats; keep independent contrasts, examples, and limits visible."
        ),
        "natural_list_policy": (
            "A compact natural list is allowed when several source terms share one semantic role. "
            "Do not split a single role into catalogue sentences just to avoid commas."
        ),
        "writer_must_not": [
            "do not create one short sentence per source sentence or scanner finding",
            "do not turn must_include terms into separate sentence obligations",
            "do not repeat the same subject at the start of adjacent sentences",
            "do not expose route, beat, contract, source slot, or other planning language",
            "do not polish into a generic essay paragraph",
        ],
    }


def _human_route(actions: list[PlanAction], semantic_roles: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_type": "SOURCE_ORDER_PARAGRAPH_REBUILD",
        "movement": "opening source beat -> grouped development beats -> closing source beat",
        "human_logic": "The writer should rebuild the paragraph from its own source order instead of following sentence findings or importing a topic template.",
        "paragraph_jobs": ["Carry the opening source beat.", "Group middle source beats by meaning.", "Carry the closing source beat as the paragraph's own limit or contrast."],
        "allowed_texture": {"compact_list": 1, "short_sentences": 1, "compound_sentences": 2, "direct_contrast": True},
        "avoid": ["sentence-by-sentence repair", "new claims", "neighbor paragraph topics", "policy recommendation"],
    }


def _semantic_role_map(actions: list[PlanAction]) -> dict[str, Any]:
    return {
        "source_order_rule": "Use only selected paragraph source beats in their original order.",
        "opening_terms": actions[0].preserve_terms[:8] if actions else [],
        "development_terms": _dedupe([term for action in actions[1:-1] for term in action.preserve_terms[:4]])[:16],
        "closing_terms": actions[-1].preserve_terms[:8] if actions else [],
        "role_instruction": "Use these source-order roles to build paragraph movement. Do not allocate one final sentence to each term.",
    }


def _first_matching_term(text: str, candidates: list[str]) -> str:
    lowered = str(text or "").casefold()
    for candidate in candidates:
        if re.search(rf"\b{re.escape(candidate)}\b", lowered):
            return candidate
    terms = source_terms(text, limit=4)
    return terms[0] if terms else ""


def _condition_phrase(text: str) -> str:
    match = re.search(r"\b(?:surrounded by|facing|using|working with|learning from)\s+([^.!?]{3,80})", str(text or ""), flags=re.I)
    return match.group(0).strip(" .") if match else ""


def _source_channels(text: str) -> list[str]:
    matches: list[str] = []
    for match in re.finditer(r"\bfrom\s+([^.!?]+)", str(text or ""), flags=re.I):
        span = re.split(r"\b(?:because|which|where|when|while)\b", match.group(1), maxsplit=1, flags=re.I)[0]
        matches.extend(_list_items(span))
    named = re.findall(r"\b(?:[A-Z][A-Za-z0-9'’-]{2,}|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'’-]{2,}|[A-Z]{2,}))*\b", str(text or ""))
    matches.extend(term for term in named if _proper_or_acronym_channel(term))
    channels = _dedupe([
        term for term in matches
        if _source_channel_key(term) not in {"now", "this", "that", "they", "the"}
    ])
    return _normalise_source_channels(channels)[:18]


def _proper_or_acronym_channel(term: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", str(term or ""))
    if not words:
        return False
    if any(word.isupper() and len(word) >= 2 for word in words):
        return True
    if any(re.search(r"[a-z][A-Z]", word) for word in words):
        return True
    return len(words) > 1 and any(word[:1].isupper() for word in words[1:])


def _normalise_source_channels(channels: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for channel in channels:
        key = _source_channel_key(channel)
        if key not in seen:
            rows.append(channel)
            seen.add(key)
    return rows


def _source_channel_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _list_items(text: str) -> list[str]:
    visible = re.sub(r"\bbut\s+also\s+from\b", "", str(text or ""), flags=re.I)
    visible = re.sub(r"\balso\s+from\b", "", visible, flags=re.I)
    parts = [
        part.strip(" ,")
        for part in re.split(r",|\s+\band\b\s+", visible)
        if part.strip(" ,")
    ]
    rows: list[str] = []
    for part in parts:
        cleaned = re.sub(r"^(?:from|also|but)\s+", "", part, flags=re.I).strip(" ,")
        if cleaned:
            rows.append(cleaned)
    return rows


def _result_relation(actions: list[PlanAction]) -> str:
    for action in actions:
        if re.search(r"\b(?:created|produced|made|led to|resulted in)\b", action.source_text, flags=re.I):
            return _strip_leading_heading(action.source_text).strip(" .")
    return ""


def _contrast_relation(actions: list[PlanAction]) -> str:
    no_longer = [
        _strip_leading_heading(action.source_text).strip(" .")
        for action in actions
        if re.search(r"\bno longer\b", action.source_text, flags=re.I)
    ]
    return " ".join(no_longer[:2])


def _challenge_relation(actions: list[PlanAction]) -> str:
    for action in reversed(actions):
        if re.search(r"\b(?:challenge|problem|issue|risk|concern)\b", action.source_text, flags=re.I):
            return _strip_leading_heading(action.source_text).strip(" .")
    return ""


def _max_adjacent_affected_cluster(actions: list[PlanAction]) -> int:
    best = 0
    current = 0
    for action in actions:
        if action.tags:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _dominant_flow_problem(tag_counts: Counter[str]) -> str:
    if not tag_counts:
        return "light local route repair"
    if tag_counts.get("packed_list") or tag_counts.get("sentence_overload"):
        return "packed source material risks turning into item-by-item short sentences"
    if tag_counts.get("paragraph_rhythm") or tag_counts.get("predictable_start"):
        return "repeated sentence routes and starts need paragraph-level rhythm repair"
    if tag_counts.get("context_anchor_gap") or tag_counts.get("broad_claim"):
        return "claims need clearer paragraph anchors and scoped relations"
    return "multiple local findings need one coherent paragraph route"


def _paragraph_flow_plan(
    paragraph: Paragraph,
    actions: list[PlanAction],
    affected: list[PlanAction],
    tag_counts: Counter[str],
) -> list[dict[str, Any]]:
    first_terms = source_anchor_terms(_strip_leading_heading(paragraph.text), term_limit=12)[:5]
    affected_terms = _dedupe([
        term
        for action in affected
        for term in action.preserve_terms[:4]
    ])[:12]
    return [
        {
            "step_id": "paragraph_b001",
            "function": "open from a concrete source condition, actor, or contrast",
            "source_basis": [actions[0].sentence_id] if actions else [],
            "must_include": first_terms[:4],
            "must_avoid_shape": ["generic opener", "same source sentence with synonyms"],
        },
        {
            "step_id": "paragraph_b002",
            "function": "group related source material before explaining it",
            "source_basis": [action.sentence_id for action in affected[:4]],
            "must_include": affected_terms[:8],
            "must_avoid_shape": ["one sentence per finding", "comma-list compression", "repeated subject starts"],
        },
        {
            "step_id": "paragraph_b003",
            "function": "state the scoped relationship or limitation supported by the source beats",
            "source_basis": [action.sentence_id for action in affected[-4:]],
            "must_include": affected_terms[4:12],
            "must_avoid_shape": ["broad takeaway", "unsupported stronger claim"],
        },
        {
            "step_id": "paragraph_b004",
            "function": "close on the paragraph's submitted purpose without adding a new conclusion",
            "source_basis": [actions[-1].sentence_id] if actions else [],
            "must_include": source_anchor_terms(actions[-1].source_text if actions else paragraph.text, term_limit=8)[:5],
            "must_avoid_shape": ["detached final summary", "smooth generic essay close"],
        },
    ]


def _operation(tags: list[str]) -> str:
    if "author_anchor_gap" in tags and "unsupported_claim_gap" in tags:
        return "narrow the evaluative claim inside the same source beat; use author-proxy wording only as reviewable replacement, not as an added conclusion"
    if "author_anchor_gap" in tags:
        return "add a reviewable author-proxy bridge grounded in submitted context"
    if "unsupported_claim_gap" in tags:
        return "add source-supported support, narrow the claim, or mark reviewable author-proxy provenance"
    if "context_anchor_gap" in tags:
        return "replace unsupported demonstrative route with a source-grounded context bridge"
    if "semantic_bridge_gap" in tags:
        return "make the missing source-to-claim reasoning bridge explicit"
    if "citation_anchor" in tags:
        return "keep parenthetical citation spans close to the claim while reducing packed shape"
    if "broad_claim" in tags:
        return "narrow broad claim using source terms or mark reviewable bridge provenance"
    if "transition_stack" in tags:
        return "remove stacked transition and attach the sentence to a source beat"
    if "paraphrase_smoothing" in tags:
        return "replace smooth paraphrase wrapper with source-level sentence pressure"
    if "packed_list" in tags:
        return "split packed list material into two connected source-preserving prose beats without compression or repeated itemization"
    if "sentence_overload" in tags:
        return "split overloaded sentence without removing source ideas"
    if "predictable_start" in tags:
        return "change the opener using source-derived terms"
    if "abstract_density" in tags:
        return "replace abstract wrapping with concrete source terms"
    return "preserve meaning while varying sentence route"


def _method(tags: list[str]) -> str:
    if "author_anchor_gap" in tags and "unsupported_claim_gap" in tags:
        return "claim_scope_repair"
    if "author_anchor_gap" in tags:
        return "author_proxy_bridge"
    if "unsupported_claim_gap" in tags:
        return "author_proxy_bridge"
    if "context_anchor_gap" in tags:
        return "context_anchor_bridge"
    if "semantic_bridge_gap" in tags:
        return "semantic_bridge_repair"
    if "citation_anchor" in tags:
        return "citation_relation_repair"
    if "broad_claim" in tags:
        return "claim_scope_repair"
    if "transition_stack" in tags:
        return "transition_rebuild"
    if "paraphrase_smoothing" in tags:
        return "source_revoice"
    if "packed_list" in tags or "sentence_overload" in tags:
        return "list_rhythm_rebuild"
    if "predictable_start" in tags or "paragraph_rhythm" in tags:
        return "route_rebuild"
    return "preserve"


def _do_not(tags: list[str]) -> list[str]:
    rules: list[str] = []
    if "packed_list" in tags:
        rules.extend([
            "do not compress this list into a longer overloaded sentence",
            "do not split each item into repeated one-item sentences",
        ])
    if "author_anchor_gap" in tags or "unsupported_claim_gap" in tags:
        rules.append("do not add a detached conclusion or verified new fact")
    if "predictable_start" in tags:
        rules.append("do not keep the same opener with synonyms")
    if "context_anchor_gap" in tags:
        rules.append("do not leave this/that/it without a clear source antecedent")
    if "paraphrase_smoothing" in tags:
        rules.append("do not make the sentence smoother or more academic")
    return rules
