from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from .plan import Plan
from .paragraph_planner import build_paragraph_planner_brief
from .prose_repair_rules import consolidated_prose_repair_rules, risky_source_terms, writer_agent_profile
from .source_quality import created_effect_frame_anchor_blockers, created_effect_frame_constraints
from .text import Paragraph, source_anchor_terms


def build_writer_brief_prompt(paragraph: Paragraph, plan: Plan) -> str:
    raw_planner_brief = _planner_brief(plan)
    planner_brief = _writer_prompt_safe_brief(raw_planner_brief)
    planner_brief = _compact_writer_list_obligations(planner_brief)
    list_coverage_requirements = _list_coverage_requirements(plan)
    writer_execution_plan = _writer_execution_plan(planner_brief, list_coverage_requirements)
    payload = {
        "task": "rewrite_one_paragraph_from_curated_brief",
        "target_paragraph_id": paragraph.id,
        "writer_agent_profile": writer_agent_profile(),
        "source_paragraph": _redacted_source_paragraph(paragraph.text, raw_planner_brief),
        "source_paragraph_copy_policy": (
            "Copy-blocked source phrases are redacted here. Preserve their meaning through "
            "writer_execution_plan and coverage_beats; do not reconstruct the redacted wording."
        ),
        "writer_handoff_priority": [
            "writer_execution_plan",
            "validated_construction_route.sentence_jobs",
            "coverage_beats",
            "list_coverage_requirements",
            "source_frame_constraints",
            "coverage_beats.must_not_copy",
            "planner_brief.do_not_copy_route",
            "planner_brief.route_rewrite_guidance",
            "planner_brief.coverage_terms",
        ],
        "planner_brief": planner_brief,
        "writer_execution_plan": writer_execution_plan,
        "route_sequence_guards": _route_sequence_guards(planner_brief),
        "validated_construction_route": planner_brief.get("validated_construction_route", {}),
        "coverage_beats": planner_brief.get("coverage_beats", []),
        "list_coverage_requirements": list_coverage_requirements,
        "source_frame_constraints": created_effect_frame_constraints(paragraph.text),
        "writer_retry_feedback": _writer_retry_feedback(plan, raw_planner_brief),
        "must_keep_terms": _must_keep_terms(paragraph, plan),
        "global_prose_repair_rules": consolidated_prose_repair_rules(),
        "hard_reject_if": [
            "grammar fragment or missing verb",
            "sentence starts with a conjunction or dropped subject",
            "subject is joined to a predicate with 'and' instead of a verb",
            "with + noun phrase + finite verb",
            "connector fragment after a period or comma",
            "broken contrast pair such as 'does not only ... Yet it also ...'",
            "one final sentence per finding",
            "keyword dump or repeated and-chain",
            "long source list followed by a relative result clause such as ', which has created'",
            "repeated same subject starts",
            "generic reward-system or conclusion claim not present in the brief",
            "vague abstract risk-label opener",
            "vague demonstrative opener copied from the source route",
            "topic-announcement opener when a specific source tension can open the paragraph",
            "closing consequence appears before the planned cause or behaviour it is meant to follow",
            "the same consequence is stated twice in the same paragraph",
            "final consequence compresses several required judgement terms into one long abstract list",
            "meta-language such as 'provides evidence', 'bridge', 'author-proxy', or 'planner'",
            "new recommendation, solution, or policy shift that was not in the source paragraph",
            "redacted source sentence is restored as a standalone coverage sentence",
            "missing source-list item is appended to an unrelated final challenge, consequence, or assessment sentence",
            "copy-blocked source meaning is appended after the planned route",
            "neighbor context becomes a new paragraph topic",
            "proxy context becomes an adjective-stacked topic label",
            "proxy context becomes dramatic generic wording instead of source-grounded wording",
            "unsupported inflated quantity wording not present in the source or planner brief",
            "model/framework/system becomes 'longer' or 'longer than before'",
            "source meaning or polarity changes",
            "source frame changes from an abstract effect or tension into creating a concrete tool/entity",
            "source uncertainty is hardened, such as changing may/could/might become into become",
            "awkward double hedge where possible-risk wording is repeated in the same clause",
            "risk consequence starts with a vague demonstrative",
            "quoted concept is paraphrased into a different idea",
            "quoted concept is literalised as 'the words ...'",
            "author-proxy bridge is added as a standalone opening sentence before repeating the source claim",
            "required source meaning from coverage_beats disappears from the rewrite",
        ],
        "style_contract": [
            "Write normal paragraph prose, not a checklist.",
            "If the source opener or route is flagged, do not copy that same opener or route.",
            "The selected paragraph must reduce scanner findings or mean risk; preserve meaning while changing risky shape.",
            "Before returning, repair any sentence fragment, dropped subject, or malformed connector.",
            "Before returning, repair malformed negation or comparison wording.",
            "Do not use semicolons.",
            "Do not copy a proxy bridge as a long comma list; use it only as a compact bridge phrase.",
            "Return no variant that contains 'and in many', 'and as a result', 'A practice that', or 'As a result of this pressure.' as a fragment.",
            "Use only the Planner brief for any author-proxy bridge; raw Author-proxy content is not available to Writer.",
            "Follow writer_handoff_priority; never use coverage_terms as the main plan.",
            "Use validated_construction_route as the source of truth for sentence order and sentence jobs.",
            "Use writer_execution_plan as the immediate writing checklist.",
            "Each execution step must contribute to the paragraph; do not output step labels.",
            "Obey route_sequence_guards before writing any variant.",
            "Use coverage_beats as meaning obligations; coverage_terms are backup anchors only.",
            "Obey source_frame_constraints before writing the opening or any created/produced/generated relation.",
            "Use route_rewrite_guidance as route guidance, not wording to copy.",
            "Complete validated_construction_route.sentence_jobs in order; do not invent a different route.",
            "Before writing prose, reject any route that violates validated_construction_route.validation_rules.",
            "If the Planner includes an author-proxy bridge instruction, fuse it into that planned source beat.",
            "Never append a Planner bridge as a standalone tail sentence.",
            "Never leave a Planner bridge as a dangling Because/And/Which clause.",
            "A Planner bridge must not replace any source closing beat or required source term.",
            "Do not use em dashes to splice a proxy bridge into the sentence.",
            "For not only/also contrasts, keep the pair grammatical; do not write 'Yet it also' after 'does not only'.",
            "Prefer 4-6 natural sentences unless the source clearly needs another shape.",
            "Natural comma lists are allowed when items share one role.",
            "Do not attach a result clause to a long list with ', which has/have ...'; name the list or mix in a separate sentence.",
            "For packed source lists, preserve the source role first; group items by role if exact item coverage would recreate packed-list risk.",
            "When a writer_execution_plan step has list_coverage_items, use that step to distribute the items; do not leave list handling to a later step.",
            "If a source-list step has more than five items, split it into two adjacent source-list sentences or one compact sentence plus one comparison sentence.",
            "When variant_requirements asks for semantic_list_grouping, preserve the list role through natural grouping instead of forcing every item into one packed sentence.",
            "For semantic_list_grouping, exact source items may be named when needed for coverage, but distribute them across adjacent source-list sentences and do not append leftovers to later beats.",
            "Do not recover omitted list items by appending them to a later challenge, consequence, or assessment sentence.",
            "Do not split every coverage term into its own sentence.",
            "For risk paragraphs, follow the planned route order rather than moving the consequence forward.",
            "Final-consequence terms belong only in the final consequence beat.",
            "If the final beat has several judgement terms that modify the same object, keep them in one grammatical closing sentence; split only when each term has a distinct actor or action.",
            "Do not announce a broad topic opener if the paragraph can open with a practical tension.",
            "Do not convert a source frame such as an effect, tension, opportunity, or risk into a claim that the topic created a concrete tool or entity.",
            "For benefit-risk routes, open with the practical actor/action or support use; keep the broad frame as tension, not as a creator sentence.",
            "Do not state the closing consequence before the planned cause or risk beat, and do not repeat it later.",
            "For final judgement terms, prefer an actor judging the issue over a compressed phrase such as raising concerns.",
            "Do not add a final advice sentence; rewrite only the submitted paragraph's claim path.",
            "A redacted source sentence is not missing content. Fuse its coverage beat into the planned route; never restore it as a standalone sentence.",
            "Never append a source-coverage sentence after the final planned consequence.",
            "If a redacted sentence contains the opener or final consequence, realize its meaning through the planned first or final beat only.",
            "When using an author-proxy bridge, integrate it into the relevant source beat instead of appending a separate judgement sentence.",
            "Do not introduce the author-proxy bridge as a separate first sentence and then restate the original claim; that repeats sentence intent.",
            "Do not turn an author-proxy bridge into a compound adjective label in the opening; use it as a plain reason clause or omit it.",
            "Author profile terms are routing labels, not wording permission; do not insert profile terms unless the planner bridge itself uses them.",
            "When an author-proxy bridge explains why a claim matters, write it as a short reason beat before/after the claim; do not pack the reason and claim into one overloaded sentence.",
            "For role-importance claims, prefer plain actor logic from the Planner route; avoid inflated labels not present in the source.",
            "If a source opener is blocked, open with the actor, condition, or source action instead.",
            "When the brief blocks a vague demonstrative opener, avoid rebuilding the same abstract-noun route.",
            "Never write visible meta-language such as 'provides evidence that' or 'this bridge shows'.",
            "Do not add final suitability or appropriateness claims unless they appear in the source.",
            "For 'no longer reflects', preserve the negation relation exactly; never turn it into 'longer than before' or 'has become longer'.",
            "Preserve quoted concepts exactly unless the brief explicitly says to remove the quote marks.",
            "Keep the wording plain and specific.",
            "If the source frames a risk as possible, keep it possible.",
        ],
        "variant_requirements": _variant_requirements(plan),
        "output_schema": {
            "variants": [
                {
                    "id": "v1|v2|v3",
                    "mode": "one of variant_requirements.mode",
                    "text": "rewritten paragraph only",
                    "author_proxy_provenance": [],
                    "author_review_items": [],
                }
            ]
        },
        "output_rules": [
            "Return exactly the variant ids requested in variant_requirements.",
            "Each variant must be a complete rewritten paragraph.",
            "Do not return only one variant when three are requested.",
            "If writer_retry_feedback is present, fix those failures and do not repeat the failed route.",
            "If writer_retry_feedback includes failed_sentences, address each repair_instruction in the next variants.",
            "If writer_retry_feedback includes residual_rewrite_obligations, satisfy every obligation before returning variants.",
            "If writer_retry_feedback includes source_beat_coverage_gaps, restore each missing source beat in the next variants.",
            "If writer_retry_feedback includes retry_goal, the next variants must beat the failed candidate, not paraphrase it.",
            "Never repeat text listed in writer_retry_feedback.do_not_repeat.",
            "For semantic_list_grouping, the variant is invalid if source-list items are packed into one long sentence or appended to an unrelated final beat.",
            "For semantic_list_grouping, list any intentionally grouped exact item names in author_review_items so the user can review coverage.",
            "Never restore a redacted source sentence as a separate coverage sentence.",
            "Never append blocked-source meaning after completing the planned route.",
            "Every variant must end on the final writer_execution_plan step; discard any variant that appends an earlier step after the final step.",
            "If a writer_execution_plan step has step_order_rule, obey it before returning the variant.",
            "Before returning, scan each variant against route_sequence_guards and discard it if a final-consequence term appears too early.",
        ],
    }
    if _writer_topk_pressure_enabled():
        pressure = _topk_pressure(paragraph.text)
        if pressure:
            payload["topk_pressure"] = pressure
    return "Return valid JSON only with a variants array.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _writer_topk_pressure_enabled() -> bool:
    return os.environ.get("DRAFTPROOF_V6_WRITER_TOPK_PRESSURE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _topk_pressure(text: str) -> dict[str, Any]:
    """Scanner-derived top-k pressure for writer prompts."""
    try:
        from poc.predictability.scanner import PredictabilityScanner
    except ImportError:
        from predictability.scanner import PredictabilityScanner
    try:
        scanner = PredictabilityScanner(model_name="gpt2")
        result = scanner.scan_sentence(str(text or ""))
        tokens = getattr(result, "token_results", None) or []
        if not tokens:
            return {}
        top_tokens = [getattr(token, "token", "") for token in tokens if getattr(token, "top_10", False)]
        return {
            "purpose": "scanner-derived predictability pressure",
            "source_topk_fraction": round(
                sum(1 for token in tokens if getattr(token, "top_10", False)) / max(len(tokens), 1),
                4,
            ),
            "topk_k": 10,
            "high_topk_token_ledger": [
                {"token": token, "hits": count}
                for token, count in Counter(top_tokens).most_common(12)
                if token
            ],
            "rules": [
                "Use this as pressure to change predictable sentence routes.",
                "Do not damage grammar, meaning, names, or numbers.",
            ],
        }
    except Exception:
        return {}


def _planner_brief(plan: Plan) -> dict[str, Any]:
    decision = plan.ai_safe_route.get("llm_planner_decision") if isinstance(plan.ai_safe_route, dict) else {}
    if isinstance(decision, dict) and _usable_planner_decision(decision):
        return _writer_facing_planner_brief(decision)
    brief = build_paragraph_planner_brief(plan)
    if isinstance(decision, dict) and decision:
        brief["planner_status"] = _drop_empty({
            "status": decision.get("status"),
            "reason": decision.get("reason"),
        })
    return brief


def _writer_facing_planner_brief(decision: dict[str, Any]) -> dict[str, Any]:
    return _drop_empty({
        "status": decision.get("status"),
        "paragraph_route": decision.get("paragraph_route"),
        "validated_construction_route": decision.get("validated_construction_route"),
        "coverage_beats": decision.get("coverage_beats"),
        "do_not_copy_route": decision.get("do_not_copy_route"),
        "route_rewrite_guidance": decision.get("route_rewrite_guidance"),
        "coverage_terms": decision.get("coverage_terms"),
    })


def _compact_writer_list_obligations(planner_brief: dict[str, Any]) -> dict[str, Any]:
    route = planner_brief.get("validated_construction_route")
    if not isinstance(route, dict):
        return planner_brief
    jobs = route.get("sentence_jobs")
    if not isinstance(jobs, list):
        return planner_brief
    beats_by_job: dict[str, list[dict[str, Any]]] = {}
    for beat in planner_brief.get("coverage_beats") if isinstance(planner_brief.get("coverage_beats"), list) else []:
        if isinstance(beat, dict):
            beats_by_job.setdefault(str(beat.get("route_job_id") or ""), []).append(beat)
    compact_jobs: list[Any] = []
    changed = False
    for job in jobs:
        if not isinstance(job, dict):
            compact_jobs.append(job)
            continue
        rows = _string_rows(job.get("must_use_meaning"))
        job_id = str(job.get("job_id") or "")
        beat_targets = _dedupe([
            target
            for beat in beats_by_job.get(job_id, [])
            for target in _string_rows(beat.get("must_cover"))
        ])
        if _overlong_list_obligation(rows) and beat_targets:
            compact_job = dict(job)
            compact_job["must_use_meaning"] = beat_targets[:8]
            compact_jobs.append(compact_job)
            changed = True
        else:
            compact_jobs.append(job)
    if not changed:
        return planner_brief
    compact_route = dict(route)
    compact_route["sentence_jobs"] = compact_jobs
    return {**planner_brief, "validated_construction_route": compact_route}


def _overlong_list_obligation(rows: list[str]) -> bool:
    if len(rows) < 7:
        return False
    short_anchor_count = sum(1 for row in rows if len(str(row).split()) <= 3)
    return short_anchor_count >= 6


def _writer_prompt_safe_brief(value: Any) -> Any:
    if isinstance(value, list):
        return [_writer_prompt_safe_brief(item) for item in value]
    if not isinstance(value, dict):
        return value
    row = {key: _writer_prompt_safe_brief(item) for key, item in value.items()}
    for key in ("do_not_copy_route", "do_not_copy", "must_not_copy", "must_not_use"):
        if isinstance(row.get(key), list):
            # Use indexed labels so the writer knows how many distinct blocked slots exist.
            # All entries collapse to the same opaque label to prevent leaking blocked wording,
            # but the index preserves count information (e.g. '#1 of 3', '#2 of 3').
            items = [str(item) for item in row[key] if str(item).strip()]
            row[key] = [_copy_blocker_label(item, index, len(items)) for index, item in enumerate(items, start=1)]
    return row


def _copy_blocker_label(value: str, index: int = 1, total: int = 1) -> str:
    """Return an opaque label that hides blocked wording but signals slot count."""
    if total <= 1:
        return "copy-blocked source wording"
    return f"copy-blocked source wording #{index} of {total}"


def _writer_retry_feedback(plan: Plan, planner_brief: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = plan.ai_safe_route.get("writer_retry_feedback") if isinstance(plan.ai_safe_route, dict) else []
    if not isinstance(feedback, list):
        return []
    blockers = _string_rows(planner_brief.get("do_not_copy_route"))
    return [_redact_feedback(row, blockers) for row in feedback if isinstance(row, dict)]


def _redact_feedback(value: Any, blockers: list[str]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, blockers)
    if isinstance(value, list):
        return [_redact_feedback(item, blockers) for item in value]
    if isinstance(value, dict):
        return {key: _redact_feedback(item, blockers) for key, item in value.items()}
    return value


def _writer_execution_plan(planner_brief: dict[str, Any], list_requirements: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    route = planner_brief.get("validated_construction_route")
    jobs = route.get("sentence_jobs") if isinstance(route, dict) else []
    beats = planner_brief.get("coverage_beats")
    guidance = _string_rows(planner_brief.get("route_rewrite_guidance"))
    global_blockers = _string_rows(planner_brief.get("do_not_copy_route"))
    if not isinstance(jobs, list):
        return []
    beats_by_job: dict[str, list[dict[str, Any]]] = {}
    for beat in beats if isinstance(beats, list) else []:
        if isinstance(beat, dict):
            beats_by_job.setdefault(str(beat.get("route_job_id") or ""), []).append(beat)
    list_requirements_by_source = {
        str(row.get("source_sentence_id") or ""): row
        for row in list_requirements or []
        if isinstance(row, dict) and str(row.get("source_sentence_id") or "").strip()
    }
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id") or f"job_{index}")
        linked_beats = beats_by_job.get(job_id, [])
        meaning_targets = _meaning_targets(job, linked_beats)
        if not meaning_targets:
            meaning_targets = _string_rows(job.get("must_use_meaning"))[:4]
        linked_list_requirements = [
            list_requirements_by_source[source_id]
            for source_id in _string_rows(job.get("source_basis"))
            if source_id in list_requirements_by_source
        ]
        rows.append(_drop_empty({
            "step": index,
            "route_job_id": job_id,
            "writer_job": job.get("job"),
            "source_basis": job.get("source_basis"),
            "meaning_targets": meaning_targets[:10],
            "list_coverage_items": _dedupe([
                item
                for requirement in linked_list_requirements
                for item in _string_rows(requirement.get("items"))
            ])[:16],
            "list_coverage_groups": _list_coverage_groups(linked_list_requirements),
            "list_distribution_rule": _list_distribution_rule(linked_list_requirements),
            "step_order_rule": _step_order_rule(index, len(jobs), job, linked_list_requirements),
            "must_not_copy": _dedupe([
                *_string_rows(job.get("must_not_use")),
                *[item for beat in linked_beats for item in _string_rows(beat.get("must_not_copy"))],
                *global_blockers,
            ])[:12],
            "route_guidance": guidance[:4],
        }))
    return rows[:8]


def _list_coverage_groups(requirements: list[dict[str, Any]]) -> list[list[str]]:
    items = _dedupe([
        item
        for requirement in requirements
        for item in _string_rows(requirement.get("items"))
    ])
    if len(items) <= 5:
        return []
    midpoint = (len(items) + 1) // 2
    return [items[:midpoint], items[midpoint:]]


def _step_order_rule(
    index: int,
    total_steps: int,
    job: dict[str, Any],
    list_requirements: list[dict[str, Any]],
) -> str:
    rules: list[str] = []
    item_count = sum(len(_string_rows(requirement.get("items"))) for requirement in list_requirements)
    if item_count > 5:
        rules.append(
            "Complete this source-list material inside this step using list_coverage_groups as adjacent sentences; do not pack all items into one sentence and do not move leftovers to a later step."
        )
    if index < total_steps:
        rules.append("After moving to a later step, do not return to this step's source material as a tail sentence.")
    else:
        rules.append("This is the final planned step; the paragraph must end here with no extra source-coverage or support sentence after it.")
    return " ".join(rules)


def _list_distribution_rule(requirements: list[dict[str, Any]]) -> str:
    if not requirements:
        return ""
    return (
        "Handle listed items inside this source-list step. When list_coverage_groups are present, use one adjacent sentence per group. "
        "For source-close variants, distribute exact items if needed. "
        "For semantic_list_grouping variants, use natural grouping and adjacent source-list sentences; exact items may be named when needed for coverage. "
        "Do not append leftovers to a later challenge, consequence, or assessment sentence."
    )


def _redacted_source_paragraph(text: str, planner_brief: dict[str, Any]) -> str:
    return _redact_sentences_with_blockers(text, _string_rows(planner_brief.get("do_not_copy_route")))


def _redact_sentences_with_blockers(text: str, blockers: list[str]) -> str:
    visible = str(text or "")
    parts = re.split(r"(?<=[.!?])\s+", visible)
    redacted: list[str] = []
    for index, part in enumerate(parts, start=1):
        sentence = part.strip()
        if not sentence:
            continue
        if _contains_copy_blocker(sentence, blockers):
            redacted.append(f"[copy-blocked source wording] sentence {index}; use coverage_beats for meaning")
        else:
            redacted.append(_redact_text(sentence, blockers))
    return " ".join(redacted).strip()


def _contains_copy_blocker(text: str, blockers: list[str]) -> bool:
    lowered = str(text or "").casefold()
    return any(
        len(blocker.split()) >= 2 and blocker.casefold() in lowered
        for blocker in blockers
        if str(blocker).strip()
    )


def _redact_text(text: str, blockers: list[str]) -> str:
    redacted = str(text or "")
    blockers = sorted(blockers, key=len, reverse=True)
    for blocker in blockers:
        if len(blocker.split()) < 2:
            continue
        redacted = re.sub(
            re.escape(blocker),
            "[copy-blocked source wording]",
            redacted,
            flags=re.I,
        )
    redacted = re.sub(r"(?:\[copy-blocked source wording\]\s*){2,}", "[copy-blocked source wording] ", redacted)
    return re.sub(r"\s+", " ", redacted).strip()


def _route_sequence_guards(planner_brief: dict[str, Any]) -> list[dict[str, Any]]:
    route = planner_brief.get("validated_construction_route")
    if not isinstance(route, dict):
        return []
    jobs = route.get("sentence_jobs")
    if not isinstance(jobs, list) or len(jobs) < 2:
        return []
    final_terms = _final_step_terms(route)
    if not final_terms:
        return []
    return [
        {
            "guard_id": "final_consequence_after_risk_behaviour",
            "terms": final_terms,
            "allowed_only_in_route_job": _last_job_id(route),
            "must_appear_after_route_jobs": _earlier_job_ids(route),
            "reject_if": "A final-consequence term appears before the planned cause, risk, or support beats.",
        }
    ]


def _final_step_terms(route: dict[str, Any]) -> list[str]:
    jobs = route.get("sentence_jobs") if isinstance(route, dict) else []
    if not isinstance(jobs, list) or not jobs:
        return []
    return _string_rows(jobs[-1].get("must_use_meaning")) if isinstance(jobs[-1], dict) else []


def _last_job_id(route: dict[str, Any]) -> str:
    jobs = route.get("sentence_jobs") if isinstance(route, dict) else []
    if not isinstance(jobs, list) or not jobs or not isinstance(jobs[-1], dict):
        return ""
    return str(jobs[-1].get("job_id") or "")


def _earlier_job_ids(route: dict[str, Any]) -> list[str]:
    jobs = route.get("sentence_jobs") if isinstance(route, dict) else []
    if not isinstance(jobs, list):
        return []
    return [
        str(job.get("job_id") or "")
        for job in jobs[:-1]
        if isinstance(job, dict) and str(job.get("job_id") or "").strip()
    ][-2:]


def _meaning_targets(job: dict[str, Any], beats: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for beat in beats:
        rows.extend(_string_rows(beat.get("must_cover")))
        meaning = str(beat.get("meaning") or "").strip()
        if meaning and len(meaning.split()) <= 14:
            rows.append(meaning)
    if rows:
        return _dedupe(rows)
    return _dedupe(_string_rows(job.get("must_use_meaning")))


def _usable_planner_decision(decision: dict[str, Any]) -> bool:
    # Exclude "invalid_source_basis": a decision with unknown sentence IDs would cause the
    # writer execution plan to reference non-existent source sentences, so fall back to the
    # deterministic paragraph planner brief instead.
    if decision.get("status") not in (None, "ok", "degraded_contract_gaps", "invalid_route_shape"):
        return False
    route = decision.get("validated_construction_route")
    jobs = route.get("sentence_jobs") if isinstance(route, dict) else []
    return bool(
        decision.get("paragraph_route")
        and decision.get("flow_plan")
        and isinstance(jobs, list)
        and jobs
    )


def _must_keep_terms(paragraph: Paragraph, plan: Plan) -> list[str]:
    terms = source_anchor_terms(paragraph.text, term_limit=36, phrase_limit=6)
    for action in plan.actions:
        terms.extend(action.preserve_terms)
    terms.extend(_quoted_terms(paragraph.text))
    risky_terms = risky_source_terms(paragraph.text)
    frame_anchor_blockers = created_effect_frame_anchor_blockers(paragraph.text)
    blocked_components = _copy_blocker_component_terms(paragraph.text)
    overloaded_list_components = _overloaded_list_component_terms(plan)
    seen: set[str] = set()
    kept: list[str] = []
    for term in terms:
        normalized = _normalize_keep_term(term)
        key = normalized.casefold()
        if (
            not normalized
            or key in seen
            or key in risky_terms
            or key in frame_anchor_blockers
            or key in blocked_components
            or key in overloaded_list_components
        ):
            continue
        seen.add(key)
        kept.append(normalized)
    return kept[:42]


def _normalize_keep_term(term: str) -> str:
    text = str(term or "").strip()
    if " " not in text and text.isalpha() and text.istitle():
        return text[:1].lower() + text[1:]
    return text


def _quoted_terms(text: str) -> list[str]:
    import re

    return [
        match.group(1).strip()
        for match in re.finditer(r"[\"“”']([^\"“”']{2,80})[\"“”']", str(text or ""))
        if match.group(1).strip()
    ]


def _copy_blocker_component_terms(text: str) -> set[str]:
    from .prose_repair_rules import risky_source_copy_phrases

    visible = str(text or "")
    blocked = risky_source_copy_phrases(visible)
    rows: set[str] = set()
    for phrase in blocked:
        remainder = re.sub(re.escape(phrase), " ", visible, flags=re.I)
        for token in re.findall(r"[A-Za-z][A-Za-z'’-]*", phrase):
            key = token.casefold()
            if len(key) <= 3:
                continue
            if not re.search(rf"\b{re.escape(key)}\b", remainder, flags=re.I):
                rows.add(key)
    return rows


def _overloaded_list_component_terms(plan: Plan) -> set[str]:
    rows: set[str] = set()
    for action in plan.actions:
        if not _packed_action(action.tags, action.source_text):
            continue
        parts = _list_like_parts(action.source_text)
        if len(parts) < 5:
            continue
        for part in parts:
            for term in source_anchor_terms(part, term_limit=8, phrase_limit=2):
                key = _normalize_keep_term(term).casefold()
                if len(key) > 2:
                    rows.add(key)
    return rows


def _list_coverage_requirements(plan: Plan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in plan.actions:
        if not _packed_action(action.tags, action.source_text):
            continue
        items = _source_list_items(action.source_text)
        if len(items) < 5:
            continue
        rows.append({
            "source_sentence_id": action.sentence_id,
            "source_role": "source-list coverage",
            "items": items[:12],
            "minimum_coverage": "preserve the source-list role; exact items may be distributed across adjacent source-list sentences when needed for coverage",
            "distribution_rule": "use grouping or adjacent source-beat sentences; do not place list leftovers in the final challenge or consequence beat",
            "semantic_grouping_rule": "when requested by variant_requirements, group item names by shared source role inferred from the submitted sentence; do not import example categories or domain labels",
        })
    return rows[:4]


def _source_list_items(text: str) -> list[str]:
    items: list[str] = []
    for part in _list_like_parts(text):
        anchors = source_anchor_terms(part, term_limit=8, phrase_limit=2)
        if not anchors:
            continue
        if len(anchors) == 1:
            item = anchors[0]
        elif len(anchors) == 2 and anchors[0].casefold() in {"learn", "use", "uses", "using", "from", "turn", "turns"}:
            item = anchors[1]
        elif len(anchors) <= 2 and len(part.split()) <= 4:
            item = " ".join(anchors)
        else:
            item = anchors[-1]
        if item and item.casefold() not in {row.casefold() for row in items}:
            items.append(item)
    return items


def _packed_action(tags: list[str], text: str) -> bool:
    tag_set = {str(tag).casefold() for tag in tags}
    return bool(
        tag_set & {"packed_list", "sentence_overload", "paragraph_rhythm"}
        and len(_list_like_parts(text)) >= 5
    )


def _list_like_parts(text: str) -> list[str]:
    visible = re.sub(r"\b(?:but\s+also|as\s+well\s+as|along\s+with|together\s+with)\b", ", ", str(text or ""), flags=re.I)
    visible = re.sub(r",\s+(?:and|or)\s+", ", ", visible, flags=re.I)
    return [
        part.strip(" .,!?:;")
        for part in re.split(r",\s+|\s+\band\b\s+|\s+\bor\b\s+", visible, flags=re.I)
        if part.strip(" .,!?:;")
    ]


def _string_rows(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(row).strip() for row in value if str(row).strip()]


def _dedupe(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows


def _variant_requirements(plan: Plan) -> list[dict[str, str]]:
    if _list_coverage_requirements(plan):
        base = [
            {"id": "v1", "mode": "plain_human_route", "instruction": "Best direct repair from the brief with natural grouped list handling."},
            {"id": "v2", "mode": "semantic_list_grouping", "instruction": "For any packed source list, preserve the list role through natural grouping. Exact items may be named when needed for coverage, but distribute them across adjacent source-list sentences instead of one packed sentence."},
            {"id": "v3", "mode": "source_close", "instruction": "Closest to source meaning while fixing flagged shapes; preserve more exact list items only if grammar and scanner movement remain acceptable."},
        ]
    else:
        base = [
            {"id": "v1", "mode": "plain_human_route", "instruction": "Best direct repair from the brief."},
            {"id": "v2", "mode": "flow_rebuild", "instruction": "Same meaning, stronger paragraph flow, no added claims."},
            {"id": "v3", "mode": "source_close", "instruction": "Closest to source meaning while fixing flagged shapes."},
        ]
    try:
        count = int(os.environ.get("DRAFTPROOF_V6_WRITER_VARIANTS", "3"))
    except ValueError:
        count = 3
    return base[: max(1, min(3, count))]


def _drop_empty(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value not in (None, "", [], {})}
