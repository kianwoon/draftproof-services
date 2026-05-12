from __future__ import annotations

import json
import os

from rewrite.guards import detect_protected_spans
from rewrite_pipeline_core.config import _env_flag, _float_env, _phase_sampling_arg
from rewrite_pipeline_core.phases.micro_texture import (
    _anchor_lock_mapping,
    _freeze_anchor_payload,
    _restore_anchor_placeholders,
)
from rewrite_pipeline_core.prompts.reconstruction import (
    ReconstructionPlanningDeps,
    build_reconstruction_meaning_brief as _core_build_reconstruction_meaning_brief,
    build_regeneration_blueprint as _core_build_regeneration_blueprint,
)
from rewrite_pipeline_core.prompts.reconstruction_helpers import (
    _clean_full_document_candidate,
    _clean_section_candidate,
    _generation_context_ledger,
    _integrity_driver_rows,
    _reconstruction_failure_feedback,
    _reconstruction_gate_controls,
    _reference_entries_from_text,
    _target_segment_rows,
    _word_count_band,
)
from rewrite_pipeline_core.prompts.staged_reconstruction import (
    StagedReconstructionPromptDeps,
    staged_generation_section_plan as _core_staged_generation_section_plan,
    staged_reconstruction_section_prompt as _core_staged_reconstruction_section_prompt,
)
from rewrite_pipeline_core.scoring.profiles import _contribution_scores, _integrity_scores
from rewrite_pipeline_core.text_processing.text_utils import _brief_sentences, _text_word_count


def _staged_reconstruction_prompt_deps() -> StagedReconstructionPromptDeps:
    return StagedReconstructionPromptDeps(
        text_word_count=_text_word_count,
        float_env=_float_env,
        anchor_lock_mapping=_anchor_lock_mapping,
        freeze_anchor_payload=_freeze_anchor_payload,
    )

def _staged_generation_section_plan(context_ledger: dict, *, max_sections: int | None = None) -> list[dict]:
    return _core_staged_generation_section_plan(
        context_ledger,
        max_sections=max_sections,
        deps=_staged_reconstruction_prompt_deps(),
    )

def _staged_reconstruction_section_prompt(
    context_ledger: dict,
    gate_controls: dict,
    section_plan: dict,
    *,
    strategy: str,
    attempt_index: int,
) -> str:
    return _core_staged_reconstruction_section_prompt(
        context_ledger,
        gate_controls,
        section_plan,
        strategy=strategy,
        attempt_index=attempt_index,
        deps=_staged_reconstruction_prompt_deps(),
    )

def _staged_reconstruction_candidate(
    gateway: LLMGateway,
    source_text: str,
    raw_json: dict,
    *,
    attempt_index: int,
    strategy: str,
    prior_attempts: list[dict] | None = None,
    max_calls: int | None = None,
) -> tuple[str, dict]:
    """Generate a candidate through section prompts and deterministic assembly."""
    brief = _build_reconstruction_meaning_brief(source_text, raw_json)
    blueprint = _build_regeneration_blueprint(source_text, raw_json, strategy)
    context_ledger = _generation_context_ledger(brief, blueprint)
    gate_controls = _reconstruction_gate_controls(prior_attempts)
    section_plans = _staged_generation_section_plan(context_ledger)
    title = section_plans[0].get("title") if section_plans else ""
    parts: list[str] = [str(title).strip()] if title else []
    section_results: list[dict] = []
    call_count = 0
    for section_plan in section_plans:
        if max_calls is not None and call_count >= max(0, int(max_calls)):
            section_results.append({
                "heading": section_plan.get("heading"),
                "target_words": section_plan.get("target_words"),
                "actual_words": 0,
                "empty": True,
                "skipped": True,
                "skip_reason": "llm_call_budget_exhausted",
            })
            continue
        prompt = _staged_reconstruction_section_prompt(
            context_ledger,
            gate_controls,
            section_plan,
            strategy=strategy,
            attempt_index=attempt_index,
        )
        response = gateway.chat(
            prompt,
            system=(
                "You are DraftProof's staged AI-Mitigation section generator. "
                "Return only bounded section prose from the structured context ledger."
            ),
            temperature=float(os.environ.get("DRAFTPROOF_RECONSTRUCTION_TEMPERATURE", "0.78")),
            max_tokens=int(os.environ.get("DRAFTPROOF_STAGED_SECTION_MAX_TOKENS", "1800")),
            top_p=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "TOP_P"),
            top_k=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "TOP_K"),
            presence_penalty=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "PRESENCE_PENALTY"),
            frequency_penalty=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "FREQUENCY_PENALTY"),
        )
        call_count += 1
        body = _clean_section_candidate(response.content, str(section_plan.get("heading") or ""))
        anchor_lock = _anchor_lock_mapping(section_plan.get("must_preserve_anchors") or [])
        missing_placeholders = []
        if anchor_lock:
            missing_placeholders = [
                item["placeholder"]
                for item in anchor_lock
                if item.get("placeholder") and item["placeholder"] not in body
            ]
            if missing_placeholders and body:
                body = f"{body.rstrip()} {' '.join(missing_placeholders)}"
        if anchor_lock:
            body = _restore_anchor_placeholders(body, anchor_lock)
        if body:
            parts.append(str(section_plan.get("heading") or "").strip())
            parts.append(body)
        section_results.append({
            "heading": section_plan.get("heading"),
            "target_words": section_plan.get("target_words"),
            "actual_words": _text_word_count(body),
            "empty": not bool(body),
            "anchor_lock_enabled": bool(anchor_lock),
            "missing_placeholders_repaired": missing_placeholders,
        })

    references = [
        str(item).strip()
        for item in context_ledger.get("reference_entries") or []
        if str(item).strip()
    ]
    if references:
        parts.append("References")
        parts.extend(references)
    candidate = "\n\n".join(part for part in parts if str(part).strip()).strip()
    metadata = {
        "schema_version": "staged_reconstruction.v1",
        "enabled": True,
        "llm_calls": call_count,
        "sections": section_results,
        "assembled_word_count": _text_word_count(candidate),
        "reference_entries_preserved": len(references),
        "source_draft_included": False,
        "context_ledger_schema": context_ledger.get("schema_version"),
        "max_calls": max_calls,
        "budget_exhausted": bool(
            max_calls is not None and call_count >= max(0, int(max_calls))
        ),
    }
    return _clean_full_document_candidate(candidate, source_text), metadata

def _reconstruction_planning_deps() -> ReconstructionPlanningDeps:
    return ReconstructionPlanningDeps(
        detect_protected_spans=detect_protected_spans,
        word_count_band=_word_count_band,
        reference_entries_from_text=_reference_entries_from_text,
        brief_sentences=_brief_sentences,
        integrity_driver_rows=_integrity_driver_rows,
        target_segment_rows=_target_segment_rows,
    )

def _build_reconstruction_meaning_brief(source_text: str, raw_json: dict | None) -> dict:
    return _core_build_reconstruction_meaning_brief(
        source_text,
        raw_json,
        deps=_reconstruction_planning_deps(),
    )

def _build_regeneration_blueprint(source_text: str, raw_json: dict | None, strategy: str) -> dict:
    return _core_build_regeneration_blueprint(
        source_text,
        raw_json,
        strategy,
        deps=_reconstruction_planning_deps(),
    )

def _reconstruction_mitigation_prompt(
    source_text: str,
    raw_json: dict,
    ai_mitigation: dict | None,
    *,
    attempt_index: int,
    strategy: str,
    prior_attempts: list[dict] | None = None,
) -> str:
    contribution = _contribution_scores(raw_json)
    integrity = _integrity_scores(raw_json)
    brief = _build_reconstruction_meaning_brief(source_text, raw_json)
    blueprint = _build_regeneration_blueprint(source_text, raw_json, strategy)
    context_ledger = _generation_context_ledger(brief, blueprint)
    failure_feedback = _reconstruction_failure_feedback(prior_attempts)
    gate_controls = _reconstruction_gate_controls(prior_attempts)
    include_source_draft = _env_flag("DRAFTPROOF_RECONSTRUCTION_INCLUDE_SOURCE_DRAFT", False)
    compact_failure_rows = []
    for item in failure_feedback:
        compact_failure_rows.append(
            "- "
            + "; ".join(
                part
                for part in [
                    f"strategy={item.get('strategy')}",
                    f"reason={item.get('reason')}",
                    f"human_shift={item.get('human_shift_score')}",
                    f"ai_authorship_delta={item.get('ai_authorship_delta')}",
                    f"human_delta={item.get('human_delta')}",
                    f"ai_transformation_delta={item.get('ai_transformation_delta')}",
                ]
                if not part.endswith("=None")
            )
        )
    strategy_guidance = {
        "conservative_reconstruction": (
            "Keep the same claim set, but rebuild paragraph routes, sentence openings, and causal bridges. "
            "Prefer narrower claims over adding new evidence."
        ),
        "reasoning_dense_reconstruction": (
            "Compress generic explanation and make each paragraph carry a clearer reasoning move: context, friction, evidence relation, implication."
        ),
        "domain_grounded_reconstruction": (
            "Use domain-specific operational language already present in the draft. Do not add new workplace, class, source, or personal details."
        ),
    }.get(strategy, "Rebuild the document structure while preserving meaning and protected facts.")
    return (
        "DraftProof AI-Mitigation Reconstruction.\n"
        "This is not sentence-level revision, not paraphrasing, and not modification of the submitted prose. "
        "Generate a new document from the scanner context ledger.\n"
        "Goal: produce a human-authored regeneration that moves the next scan toward Human Contribution >= 80 where the submitted evidence permits it. "
        "If 80 is not reachable without inventing facts, maximize Human Shift Score while preserving what the submitted content conveys.\n\n"
        f"Current scores: AI Authorship={integrity.get('ai_authorship')}, Human={contribution.get('human')}, "
        f"AI Transformation={contribution.get('ai_transformation')}, Grounding Risk={integrity.get('grounding')}.\n"
        f"Strategy: {strategy}. {strategy_guidance}\n\n"
        "Scanner context ledger for generation. This is the generation input; do not ask for or reconstruct from the original prose order:\n"
        f"{json.dumps(context_ledger, ensure_ascii=False)[:9000]}\n\n"
        "Scanner-derived regeneration blueprint to follow before writing prose. Source previews have been removed so you do not scaffold from submitted sentences:\n"
        f"{json.dumps({k: v for k, v in blueprint.items() if k != 'paragraph_plans'}, ensure_ascii=False)[:2600]}\n\n"
        "Previous failed attempts to correct:\n"
        f"{json.dumps(failure_feedback, ensure_ascii=False) if failure_feedback else '- None yet.'}\n"
        f"{chr(10).join(compact_failure_rows) if compact_failure_rows else ''}\n\n"
        "Scanner/gate feedback controls for this attempt:\n"
        f"{json.dumps(gate_controls, ensure_ascii=False)[:4200]}\n\n"
        "Word-count requirement:\n"
        f"- The submitted draft has {brief['word_count_band']['source_word_count']} words. "
        f"Return {brief['word_count_band']['min_words']} to {brief['word_count_band']['max_words']} words only.\n\n"
        "Regeneration blueprint:\n"
        "- Follow the scanner-derived context ledger and blueprint above. They are the plan.\n"
        "- Follow the scanner/gate feedback controls above. They override generic writing preferences for this attempt.\n"
        "- Do not follow the submitted sentence order as a scaffold. The submitted prose is not the generation substrate.\n"
        "- Build a fresh paragraph route from the blueprint: concrete context, pressure point, evidence/source relation, author reasoning, bounded implication.\n"
        "- Target the industry-baseline AI Authorship drivers directly: token predictability, burstiness regularity, discourse regularity, semantic uniformity, template phrase signal, and rewrite smoothness.\n"
        "- Increase industry-baseline suppressors through real authorship friction: causal reasoning, local constraint awareness, domain cognition, and natural paragraph variance. Do not use typo/noise tricks.\n"
        "- Treat grounding quality as separate from AI authorship. Narrow unsupported claims or preserve source relations; never invent citations, dates, names, statistics, or evidence.\n"
        "- Give adjacent paragraphs different jobs. One may start from a problem, another from a source relation, another from a limitation or consequence.\n"
        "- Use target segments as the highest-priority places to change the route, not as sentences to lightly paraphrase.\n"
        "- Use only allowed existing additions and implied process detail already licensed by the scan constraints.\n\n"
        "Candidate-family instruction:\n"
        f"- This candidate must use the '{strategy}' family. It must be structurally different from other families, not a temperature variant.\n"
        "- Make the prose less template-like by changing paragraph purpose, not by adding odd wording.\n"
        "- Do not make the draft smoother. Smoothness without local reasoning is a failure.\n\n"
        "Allowed reconstruction moves:\n"
        "- Reorder paragraphs and claims when the meaning is preserved.\n"
        "- Split over-smooth paragraphs and vary sentence pacing naturally.\n"
        "- Compress broad explanatory padding into denser reasoning.\n"
        "- Make causal links explicit when the submitted content already implies them.\n"
        "- Replace generic academic transitions with context-specific connections.\n"
        "- Narrow unsupported claims rather than inventing evidence.\n\n"
        "Human Shift acceptance requirements:\n"
        "- The next scan must not trade a Human Contribution gain for higher AI Authorship.\n"
        "- Avoid increasing semantic uniformity, review burden, or critical/high findings.\n"
        "- A candidate that raises Human Contribution but raises AI Authorship will be rejected.\n"
        "- Prefer a less polished, more locally reasoned draft over a smoother academic rewrite.\n\n"
        "Forbidden moves:\n"
        "- Do not invent personal observations, examples, dates, statistics, sources, citations, institutions, or facts.\n"
        "- Do not change quoted text, citations, years, names, numbers, headings, or source relations.\n"
        "- Do not use placeholders, review brackets, comments, labels, markdown fences, or explanations.\n"
        "- Do not preserve the original sentence order or sentence shape just to be safe; preserve meaning instead.\n\n"
        f"Attempt {attempt_index}: return only the complete regenerated document."
        + (
            "\n\nSOURCE DRAFT DEBUG FALLBACK:\n"
            f"<TARGET_DOCUMENT>\n{source_text.strip()}\n</TARGET_DOCUMENT>"
            if include_source_draft
            else ""
        )
    )
