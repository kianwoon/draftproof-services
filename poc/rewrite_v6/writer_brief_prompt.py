from __future__ import annotations

import json
import os
import re
from typing import Any

from .plan import Plan
from .paragraph_planner import build_paragraph_planner_brief
from .prose_repair_rules import consolidated_prose_repair_rules, risky_source_terms, writer_agent_profile
from .text import Paragraph, source_anchor_terms


def build_writer_brief_prompt(paragraph: Paragraph, plan: Plan) -> str:
    raw_planner_brief = _planner_brief(plan)
    planner_brief = _writer_prompt_safe_brief(raw_planner_brief)
    writer_execution_plan = _writer_execution_plan(planner_brief)
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
        "writer_retry_feedback": _writer_retry_feedback(plan, raw_planner_brief),
        "must_keep_terms": _must_keep_terms(paragraph, plan),
        "global_prose_repair_rules": consolidated_prose_repair_rules(),
        "hard_reject_if": [
            "grammar fragment or missing verb",
            "sentence starts with a conjunction or dropped subject",
            "subject is joined to a predicate with 'and' instead of a verb",
            "with + subject + finite verb, such as 'with students received'",
            "connector fragment such as 'and in many schools', 'and as a result', or a sentence fragment after a period",
            "broken contrast pair such as 'does not only ... Yet it also ...'",
            "one final sentence per finding",
            "keyword dump or repeated and-chain",
            "repeated same subject starts",
            "generic reward-system or conclusion claim not present in the brief",
            "vague abstract risk-label opener",
            "topic-announcement opener when a specific source tension can open the paragraph",
            "closing consequence appears before the planned cause or behaviour it is meant to follow",
            "the same consequence is stated twice in the same paragraph",
            "meta-language such as 'provides evidence', 'bridge', 'author-proxy', or 'planner'",
            "new recommendation, solution, or policy shift that was not in the source paragraph",
            "neighbor context becomes a new paragraph topic",
            "proxy context becomes an adjective-stacked label such as digital-media-rich classrooms",
            "proxy context becomes dramatic generic wording such as flood of information or floods classrooms with information",
            "model/framework/system becomes 'longer' or 'longer than before'",
            "source meaning or polarity changes",
            "source uncertainty is hardened, such as changing may/could/might become into become",
            "awkward double hedge where possible-risk wording is repeated in the same clause",
            "risk consequence starts with a vague demonstrative",
            "quoted concept is paraphrased into a different idea",
            "quoted concept is literalised as 'the words ...'",
            "author-proxy bridge is added as a standalone opening sentence before repeating the source claim",
            "required source terms disappear from the rewrite",
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
            "Do not split every coverage term into its own sentence.",
            "For risk paragraphs, follow the planned route order rather than moving the consequence forward.",
            "Final-consequence terms belong only in the final consequence beat.",
            "Do not announce a broad topic opener if the paragraph can open with a practical tension.",
            "Do not state the closing consequence before the planned cause or risk beat, and do not repeat it later.",
            "Do not add a final advice sentence; rewrite only the submitted paragraph's claim path.",
            "When using an author-proxy bridge, integrate it into the relevant source beat instead of appending a separate judgement sentence.",
            "Do not introduce the author-proxy bridge as a separate first sentence and then restate the original claim; that repeats sentence intent.",
            "Do not turn an author-proxy bridge into a compound adjective label in the opening; use it as a plain reason clause or omit it.",
            "Author profile terms are routing labels, not wording permission; do not insert profile terms unless the planner bridge itself uses them.",
            "When an author-proxy bridge explains why a claim matters, write it as a short reason beat before/after the claim; do not pack the reason and claim into one overloaded sentence.",
            "For role-importance claims, prefer plain actor logic from the Planner route; avoid inflated labels not present in the source.",
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
            "If writer_retry_feedback includes retry_goal, the next variants must beat the failed candidate, not paraphrase it.",
            "Never repeat text listed in writer_retry_feedback.do_not_repeat.",
            "Before returning, scan each variant against route_sequence_guards and discard it if a final-consequence term appears too early.",
        ],
    }
    return "Return valid JSON only with a variants array.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


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
        "author_proxy_request": decision.get("author_proxy_request"),
    })


def _writer_prompt_safe_brief(value: Any) -> Any:
    if isinstance(value, list):
        return [_writer_prompt_safe_brief(item) for item in value]
    if not isinstance(value, dict):
        return value
    row = {key: _writer_prompt_safe_brief(item) for key, item in value.items()}
    for key in ("do_not_copy_route", "do_not_copy", "must_not_copy", "must_not_use"):
        if isinstance(row.get(key), list):
            row[key] = _dedupe([_copy_blocker_label(str(item)) for item in row[key] if str(item).strip()])
    return row


def _copy_blocker_label(value: str) -> str:
    return "copy-blocked source wording"


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


def _writer_execution_plan(planner_brief: dict[str, Any]) -> list[dict[str, Any]]:
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
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id") or f"job_{index}")
        linked_beats = beats_by_job.get(job_id, [])
        meaning_targets = _meaning_targets(job, linked_beats)
        if not meaning_targets:
            meaning_targets = _string_rows(job.get("must_use_meaning"))[:4]
        rows.append(_drop_empty({
            "step": index,
            "route_job_id": job_id,
            "writer_job": job.get("job"),
            "source_basis": job.get("source_basis"),
            "meaning_targets": meaning_targets[:6],
            "must_not_copy": _dedupe([
                *_string_rows(job.get("must_not_use")),
                *[item for beat in linked_beats for item in _string_rows(beat.get("must_not_copy"))],
                *global_blockers,
            ])[:12],
            "route_guidance": guidance[:4],
        }))
    return rows[:8]


def _redacted_source_paragraph(text: str, planner_brief: dict[str, Any]) -> str:
    return _redact_text(text, _string_rows(planner_brief.get("do_not_copy_route")))


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
    if decision.get("status") not in (None, "ok", "degraded_contract_gaps", "invalid_route_shape", "invalid_source_basis"):
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
    seen: set[str] = set()
    kept: list[str] = []
    for term in terms:
        normalized = _normalize_keep_term(term)
        key = normalized.casefold()
        if not normalized or key in seen or key in risky_terms:
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
