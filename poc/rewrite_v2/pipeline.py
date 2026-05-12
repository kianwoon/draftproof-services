"""Scan-driven rewrite pipeline V2 entrypoint."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from detect.run import DetectionRunner
from llm.gateway import LLMConfig, LLMGateway
from report.pdf import render_pdf
from report.render_rewrite import render_rewrite_report
from report.report import ReportBuilder, report_to_dict
from rewrite.guards import check_semantic_drift, detect_protected_spans, protected_spans_preserved

from .goal_contract import RewriteGoalStatus, evaluate_rewrite_goal, needs_author_context
from .goal_contract import RewriteGoalEvaluation
from .selection import CandidateLane, decide_candidate, select_best_candidate
from .strategy import (
    build_single_paragraph_reconstruction_prompt,
    build_strategy_prompt,
    clean_candidate_output,
    route_strategies,
    targeted_paragraph_briefs,
)


def _semantic_scan_allowed(strategy_kind: str | None, semantic_safe: bool) -> bool:
    if semantic_safe:
        return True
    if strategy_kind == "full_rewrite":
        return True
    return os.environ.get("DRAFTPROOF_REWRITE_V2_SCAN_REVIEW_CANDIDATES", "1").lower() not in {"0", "false", "no"}


def _extract_original_text(detect_json: dict[str, Any]) -> str:
    for key in ("input_text", "original_text", "document_text", "text", "content"):
        value = detect_json.get(key)
        if isinstance(value, str) and value.strip():
            return value
    sentence_map = detect_json.get("sentence_map")
    if isinstance(sentence_map, dict):
        rows = [
            str((row or {}).get("text") or "").strip()
            for _, row in sorted(sentence_map.items())
            if isinstance(row, dict) and str(row.get("text") or "").strip()
        ]
        if rows:
            return " ".join(rows)
    raise ValueError("rewrite_v2 requires original text in detect JSON or sentence_map")


def _scan_report(text: str) -> dict[str, Any]:
    detect_report = DetectionRunner().run_all(text)
    builder = ReportBuilder()
    builder.add_detection_report(detect_report)
    if detect_report.postprocess_results:
        builder.add_postprocess_results(detect_report.postprocess_results)
    builder.set_meta(scan_time=0, original_text=text)
    return report_to_dict(builder.build())


def _badge_ai(report: dict | None) -> float | None:
    score = ((report or {}).get("ai_risk_badge") or {}).get("ai_likelihood_score")
    return float(score) if isinstance(score, (int, float)) else None


def _badge_wq(report: dict | None) -> float | None:
    score = ((report or {}).get("ai_risk_badge") or {}).get("writing_quality_score")
    return float(score) if isinstance(score, (int, float)) else None


def _first_applied_paragraph_patch(candidate: dict[str, Any]) -> dict[str, Any] | None:
    for patch in candidate.get("patches") or []:
        if isinstance(patch, dict) and patch.get("applied") and patch.get("target_paragraph") and patch.get("rewritten_paragraph"):
            return patch
    return None


def _paragraph_local_score(candidate: dict[str, Any]) -> dict[str, Any] | None:
    patch = _first_applied_paragraph_patch(candidate)
    if not patch:
        return None
    original_report = _scan_report(str(patch["target_paragraph"]))
    rewritten_report = _scan_report(str(patch["rewritten_paragraph"]))
    original_ai = _badge_ai(original_report)
    rewritten_ai = _badge_ai(rewritten_report)
    original_wq = _badge_wq(original_report)
    rewritten_wq = _badge_wq(rewritten_report)
    ai_drop = (
        round(float(original_ai) - float(rewritten_ai), 3)
        if isinstance(original_ai, (int, float)) and isinstance(rewritten_ai, (int, float))
        else None
    )
    return {
        "paragraph_id": patch.get("paragraph_id"),
        "original_ai": original_ai,
        "rewritten_ai": rewritten_ai,
        "ai_drop": ai_drop,
        "original_wq": original_wq,
        "rewritten_wq": rewritten_wq,
        "improved": bool(isinstance(ai_drop, (int, float)) and ai_drop > 0.0),
    }


def _compose_local_winners(original_text: str, rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    winners: dict[str, dict[str, Any]] = {}
    for row in rows:
        local = row.get("paragraph_local_score")
        if not isinstance(local, dict) or not local.get("improved"):
            continue
        paragraph_id = str(local.get("paragraph_id") or row.get("paragraph_id") or "")
        if not paragraph_id:
            continue
        current = winners.get(paragraph_id)
        if current is None or float(local.get("ai_drop") or 0.0) > float((current.get("paragraph_local_score") or {}).get("ai_drop") or 0.0):
            winners[paragraph_id] = row
    text = original_text
    applied: list[dict[str, Any]] = []
    for paragraph_id, row in winners.items():
        patch = _first_applied_paragraph_patch(row)
        if not patch:
            continue
        target = str(patch.get("target_paragraph") or "")
        replacement = str(patch.get("rewritten_paragraph") or "")
        if target and replacement and target in text:
            text = text.replace(target, replacement, 1)
            applied.append({
                "paragraph_id": paragraph_id,
                "local_ai_drop": (row.get("paragraph_local_score") or {}).get("ai_drop"),
                "candidate_ai": row.get("candidate_ai"),
                "candidate_number": row.get("candidate_number"),
                "strategy": row.get("strategy"),
            })
    return text, applied


def _compose_full_doc_delta_winners(
    original_text: str,
    rows: list[dict[str, Any]],
    reference_ai: float | None,
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(reference_ai, (int, float)):
        return original_text, []
    winners: dict[str, dict[str, Any]] = {}
    for row in rows:
        paragraph_id = str(row.get("paragraph_id") or "")
        candidate_ai = row.get("candidate_ai")
        if not paragraph_id or not isinstance(candidate_ai, (int, float)):
            continue
        full_doc_delta = float(reference_ai) - float(candidate_ai)
        if full_doc_delta <= 0.0:
            continue
        patch = _first_applied_paragraph_patch(row)
        if not patch:
            continue
        current = winners.get(paragraph_id)
        if current is None or full_doc_delta > float(current.get("full_doc_delta") or 0.0):
            winners[paragraph_id] = {
                "row": row,
                "patch": patch,
                "full_doc_delta": round(full_doc_delta, 3),
            }
    text = original_text
    applied: list[dict[str, Any]] = []
    for paragraph_id, winner in winners.items():
        patch = winner["patch"]
        target = str(patch.get("target_paragraph") or "")
        replacement = str(patch.get("rewritten_paragraph") or "")
        if target and replacement and target in text:
            row = winner["row"]
            text = text.replace(target, replacement, 1)
            applied.append({
                "paragraph_id": paragraph_id,
                "full_doc_delta": winner["full_doc_delta"],
                "local_ai_drop": (row.get("paragraph_local_score") or {}).get("ai_drop"),
                "candidate_ai": row.get("candidate_ai"),
                "candidate_number": row.get("candidate_number"),
                "strategy": row.get("strategy"),
                "tactic": row.get("tactic"),
            })
    return text, applied


def _paragraph_target_map(scan_report: dict | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for brief in (scan_report or {}).get("rewrite_edit_briefs") or []:
        if not isinstance(brief, dict):
            continue
        paragraph_id = str(brief.get("paragraph_id") or "").strip()
        paragraph = str(brief.get("paragraph_excerpt") or "").strip()
        if paragraph_id and paragraph and paragraph_id not in result:
            result[paragraph_id] = paragraph
    return result


def _json_from_response(raw: str) -> dict[str, Any]:
    text = clean_candidate_output(raw)
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}


def _apply_targeted_patches(original_text: str, payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text = original_text
    applied: list[dict[str, Any]] = []
    patches = payload.get("patches") if isinstance(payload, dict) else []
    if not isinstance(patches, list):
        return original_text, applied
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        target = str(patch.get("target_paragraph") or patch.get("target_sentence") or "").strip()
        replacement = str(patch.get("rewritten_paragraph") or patch.get("rewritten_sentence") or "").strip()
        if not target or not replacement or target == replacement:
            continue
        if target not in text:
            applied.append({
                "finding_id": patch.get("finding_id"),
                "finding_ids": patch.get("finding_ids"),
                "paragraph_id": patch.get("paragraph_id"),
                "applied": False,
                "reason": "target_paragraph_not_found",
                "target_paragraph": target,
                "rewritten_paragraph": replacement,
            })
            continue
        text = text.replace(target, replacement, 1)
        applied.append({
            "finding_id": patch.get("finding_id"),
            "finding_ids": patch.get("finding_ids"),
            "paragraph_id": patch.get("paragraph_id"),
            "applied": True,
            "target_paragraph": target,
            "rewritten_paragraph": replacement,
            "rationale": patch.get("rationale"),
        })
    return text, applied


def _targeted_candidate_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if isinstance(candidates, list) and candidates:
        return [row for row in candidates if isinstance(row, dict)]
    patches = payload.get("patches") if isinstance(payload, dict) else None
    if isinstance(patches, list):
        return [{"candidate_id": "variant_1", "patches": patches}]
    return []


def _patch_filter_failures(patches: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    banned = [
        "is often described",
        "often described as",
        "one of the most",
        "in modern history",
        "highly influential",
        "key role",
        "played a key role",
        "shaping modern history",
        "significantly affected",
        "influence has extended",
        "worldwide",
        "for decades",
        "emerging from",
        "stands as",
        "significant global entity",
        "various sectors",
        "notable over time",
        "swiftly ascended",
        "melting pot",
        "grapples with",
        "pressing",
        "complex relationship",
        "complex interplay",
        "deeply intertwined",
        "multifaceted",
        "dynamic landscape",
        "continues to shape",
        "evaluating",
        "judging",
        "coexist",
        "persistent challenges",
    ]
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        target = str(patch.get("target_paragraph") or patch.get("target_sentence") or "").strip()
        rewrite = str(patch.get("rewritten_paragraph") or patch.get("rewritten_sentence") or "").strip()
        lowered = rewrite.lower()
        if not rewrite:
            failures.append("empty_rewrite")
            continue
        if target and rewrite == target:
            failures.append("unchanged_target_sentence")
        for phrase in banned:
            if phrase in lowered:
                failures.append(f"banned_phrase:{phrase}")
    return failures


def _attach_hidden_paragraph_targets(candidate_payload: dict[str, Any], target_map: dict[str, str]) -> dict[str, Any]:
    patches = candidate_payload.get("patches")
    if not isinstance(patches, list):
        return candidate_payload
    enriched = {**candidate_payload, "patches": []}
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        row = dict(patch)
        paragraph_id = str(row.get("paragraph_id") or "").strip()
        if paragraph_id and not row.get("target_paragraph") and paragraph_id in target_map:
            row["target_paragraph"] = target_map[paragraph_id]
        enriched["patches"].append(row)
    return enriched


def _paragraph_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "paragraph_reconstruction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "paragraph_id": {
                        "type": "string",
                        "description": "The paragraph id from the input brief.",
                    },
                    "rewritten_paragraph": {
                        "type": "string",
                        "description": "Replacement paragraph only, no commentary.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Short reason tied to rhythm, predictable spans, and preservation.",
                    },
                },
                "required": ["paragraph_id", "rewritten_paragraph", "rationale"],
                "additionalProperties": False,
            },
        },
    }


def _supports_openai_penalties(model: str | None) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized in {
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "deepseek/deepseek-chat",
    }


def _supports_repetition_penalty(model: str | None) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized in {
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.3-70b-instruct",
    }


def _paragraph_tactics() -> list[str]:
    raw = os.environ.get(
        "DRAFTPROOF_REWRITE_V2_TACTICS",
        "minimal_carrier,compressed_power,broken_choppy,choppy_analytic,simple_subject_stack,specific_noun_action",
    )
    tactics = [item.strip() for item in raw.split(",") if item.strip()]
    return tactics or ["plain_student_draft"]


def _use_paragraph_local_score_gate() -> bool:
    return os.environ.get("DRAFTPROOF_REWRITE_V2_USE_PARAGRAPH_LOCAL_SCORE", "0").lower() in {
        "1",
        "true",
        "yes",
    }


def _sentence_comparison(original_text: str, final_text: str) -> list[dict[str, Any]]:
    return [{
        "index": 1,
        "original": original_text,
        "rewritten": final_text,
        "changed": original_text.strip() != final_text.strip(),
    }]


def _candidate_rows_from_replay(
    replay_candidate_records: list[dict[str, Any]],
    *,
    original_text: str,
    original_report: dict[str, Any],
    required_ai_drop: float,
    target_ai_score: float | None,
) -> list[dict[str, Any]]:
    rows = []
    reference_ai = _badge_ai(original_report)
    for index, record in enumerate(replay_candidate_records, start=1):
        candidate_report = record.get("report") if isinstance(record.get("report"), dict) else dict(record)
        if isinstance(record.get("ai"), (int, float)) and "ai_risk_badge" not in candidate_report:
            candidate_report = {
                **candidate_report,
                "ai_risk_badge": {
                    "ai_likelihood_score": record.get("ai"),
                    "writing_quality_score": record.get("writing_quality"),
                    "ai_components": (
                        ((record.get("ai_footprint_gate") or {}).get("after") or {}).get("authorship_footprint")
                        or {}
                    ),
                },
            }
        candidate_text = str(record.get("text") or original_text)
        if isinstance(record.get("ai_footprint_gate"), dict) and isinstance(record.get("turnitin_like_ai_gate"), dict):
            density_gate = record.get("eligible_span_density_gate") if isinstance(record.get("eligible_span_density_gate"), dict) else {}
            strict_safe = bool(record["ai_footprint_gate"].get("safe_band"))
            turnitin_target = bool(
                record["turnitin_like_ai_gate"].get("target_met")
                or record["turnitin_like_ai_gate"].get("safe_band")
            )
            density_safe = bool(density_gate.get("safe"))
            detector_safe = bool(strict_safe and turnitin_target and density_safe)
            goal = RewriteGoalEvaluation(
                status=RewriteGoalStatus.AI_MITIGATED if detector_safe else RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE,
                goal_met=detector_safe,
                detector_safe=detector_safe,
                strict_ai_safe_band_achieved=strict_safe,
                turnitin_like_target_met=turnitin_target,
                eligible_span_density_safe=density_safe,
                reason="replay_candidate_goal_met" if detector_safe else "replay_candidate_failed_strict_goal",
                ai_footprint_gate=record["ai_footprint_gate"],
                turnitin_like_gate=record["turnitin_like_ai_gate"],
                eligible_span_density_gate=density_gate,
            )
        elif "report" in record:
            goal = evaluate_rewrite_goal(
                original_text=original_text,
                candidate_text=candidate_text,
                original_report=original_report,
                candidate_report=candidate_report,
            )
        else:
            goal = RewriteGoalEvaluation(
                status=RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE,
                goal_met=False,
                detector_safe=False,
                strict_ai_safe_band_achieved=False,
                turnitin_like_target_met=False,
                eligible_span_density_safe=False,
                reason="replay_record_without_rescan_report_failed_strict_goal",
                ai_footprint_gate={},
                turnitin_like_gate={},
                eligible_span_density_gate={},
            )
        decision = decide_candidate(
            goal=goal,
            original_report=original_report,
            candidate_report=candidate_report,
            reference_ai=reference_ai,
            required_ai_drop=required_ai_drop,
            target_ai_score=target_ai_score,
            semantic_safe=True,
            quality_safe=True,
            cost=index,
        )
        if (
            "report" not in record
            and not isinstance(record.get("ai_footprint_gate"), dict)
            and not isinstance(record.get("turnitin_like_ai_gate"), dict)
        ):
            decision_payload = {
                "lane": CandidateLane.REJECT.value,
                "selected_as_success": False,
                "goal_met": False,
                "ai_target_gap": None,
                "required_drop_met": False,
                "quality_safe": False,
                "semantic_safe": True,
                "reason": "replay_record_missing_rescan_gates",
                "rank": [],
            }
        else:
            decision_payload = decision.to_dict()
        rows.append({
            "strategy": record.get("strategy") or f"replay_candidate_{index}",
            "candidate_ai": _badge_ai(candidate_report),
            "goal": goal.to_dict(),
            "decision": decision_payload,
            "report": candidate_report,
            "text": candidate_text,
        })
    return rows


def _generate_candidates(
    *,
    original_text: str,
    scan_report: dict[str, Any],
    strategies: list[Any],
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    if not api_key:
        return []
    paragraph_targets = _paragraph_target_map(scan_report)
    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model or os.environ.get("LLM_MODEL") or "openai/gpt-4.1-mini",
        base_url=base_url or os.environ.get("LLM_BASE_URL", ""),
        timeout=timeout_seconds,
        max_retries=1,
        max_tokens=6500,
        temperature=0.45,
    ))
    candidates = []
    for strategy in strategies:
        if getattr(strategy, "kind", None).value == "targeted":
            paragraph_briefs = targeted_paragraph_briefs(scan_report)
            tactics = _paragraph_tactics()
            for brief in paragraph_briefs:
                paragraph_id = str(brief.get("paragraph_id") or "")
                variant_limit = max(1, int(strategy.max_candidates or 1))
                for number, tactic in enumerate(tactics[:variant_limit], start=1):
                    prompt = build_single_paragraph_reconstruction_prompt(brief, strategy, tactic=tactic)
                    response = gateway.chat(
                        prompt,
                        system="You are DraftProof's paragraph reconstruction engine.",
                        max_tokens=1800,
                        temperature=0.65,
                        top_p=0.9,
                        presence_penalty=0.35 if _supports_openai_penalties(model) else None,
                        frequency_penalty=0.45 if _supports_openai_penalties(model) else None,
                        repetition_penalty=1.08 if _supports_repetition_penalty(model) else None,
                        seed=(1701 + number),
                        response_format=_paragraph_response_format(),
                        provider={"require_parameters": True},
                    )
                    payload = _json_from_response(response.content)
                    patch = {
                        "paragraph_id": str(payload.get("paragraph_id") or paragraph_id),
                        "rewritten_paragraph": str(payload.get("rewritten_paragraph") or "").strip(),
                        "rationale": payload.get("rationale"),
                    }
                    candidate_payload = _attach_hidden_paragraph_targets(
                        {"candidate_id": f"{paragraph_id or 'paragraph'}_variant_{number}", "patches": [patch]},
                        paragraph_targets,
                    )
                    patch_payload = {"patches": candidate_payload.get("patches") or []}
                    candidate_text, applied_patches = _apply_targeted_patches(original_text, patch_payload)
                    filter_failures = _patch_filter_failures(patch_payload["patches"])
                    candidates.append({
                        "strategy": strategy.strategy_id,
                        "strategy_kind": strategy.kind.value,
                        "candidate_number": number,
                        "paragraph_id": paragraph_id,
                        "tactic": tactic,
                        "text": candidate_text,
                        "candidate_response": {key: value for key, value in payload.items() if key != "target_paragraph"} or payload,
                        "local_filter_passed": not filter_failures,
                        "local_filter_failures": filter_failures,
                        "applied_patch_count": sum(1 for row in applied_patches if row.get("applied")),
                        "patch_count": len(applied_patches),
                        "patches": applied_patches,
                    })
            continue
        for number in range(1, max(1, int(strategy.max_candidates or 1)) + 1):
            prompt = build_strategy_prompt(original_text, scan_report, strategy)
            response = gateway.chat(
                prompt,
                system="You are DraftProof's scan-driven AI-risk mitigation rewrite engine.",
                max_tokens=6500,
                temperature=0.45,
                top_p=0.82,
                presence_penalty=0.15,
                frequency_penalty=0.25,
            )
            candidates.append({
                "strategy": strategy.strategy_id,
                "strategy_kind": strategy.kind.value,
                "candidate_number": number,
                "text": clean_candidate_output(response.content),
                "candidate_response": clean_candidate_output(response.content),
            })
    return candidates


def run_rewrite_pipeline_v2(
    *,
    detect_json: dict[str, Any],
    output_dir: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    replay_candidate_records: list[dict[str, Any]] | None = None,
    max_runtime_seconds: int = 300,
    required_ai_drop: float = 5.0,
    full_rewrite_allowed: bool = True,
) -> dict[str, Any]:
    started = time.time()

    def progress(percent: int, message: str) -> None:
        if progress_callback:
            progress_callback(percent, message)

    progress(62, "Starting scan-driven rewrite V2")
    original_text = _extract_original_text(detect_json)
    original_report = detect_json
    reference_ai = _badge_ai(original_report)
    target_ai_score = (
        float(reference_ai) - float(required_ai_drop)
        if isinstance(reference_ai, (int, float))
        else None
    )
    strategies = route_strategies(original_report, full_rewrite_allowed=full_rewrite_allowed)
    author_context_blocked = (
        needs_author_context(original_report)
        and os.environ.get("DRAFTPROOF_REWRITE_V2_FAIL_FAST_AUTHOR_CONTEXT", "0").lower() in {"1", "true", "yes"}
    )
    if author_context_blocked and replay_candidate_records is None:
        elapsed = time.time() - started
        final_goal_eval = evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=original_text,
            original_report=original_report,
            candidate_report=original_report,
        )
        final_goal = {
            **final_goal_eval.to_dict(),
            "status": RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT.value,
            "goal_met": False,
            "reason": "scan_requires_author_context_before_rewrite_budget",
        }
        summary = {
            "rewrite_pipeline_version": "rewrite_v2_scan_driven",
            "outcome": RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT.value,
            "rewrite_goal_status": final_goal,
            "reference_ai": reference_ai,
            "required_ai_drop": required_ai_drop,
            "target_ai_score": target_ai_score,
            "strategy_trace": [strategy.to_dict() for strategy in strategies],
            "candidate_trace": [],
            "selected_candidate": None,
            "stage_timings": [{
                "stage": "rewrite_v2_scan_driven",
                "seconds": round(elapsed, 3),
                "candidates": 0,
                "selected": False,
                "stop_reason": "needs_author_context",
            }],
            "detect_scan_original": original_report,
            "detect_scan_rewritten": original_report,
            "final_text": original_text,
        }
        sentence_comparison = _sentence_comparison(original_text, original_text)
        result_obj = SimpleNamespace(
            summary=summary,
            sentence_comparison=sentence_comparison,
            rewrite_plan=None,
            mp_result=SimpleNamespace(
                original_text=original_text,
                final_text=original_text,
                converged=False,
                convergence_reason="rewrite_v2_needs_author_context",
                passes=[],
            ),
        )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        md_path = out_dir / f"draftproof_rewrite_v2_{ts}.md"
        pdf_path = out_dir / f"draftproof_rewrite_v2_{ts}.pdf"
        json_path = out_dir / f"draftproof_rewrite_v2_{ts}.json"
        md_text = render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False)
        md_path.write_text(md_text, encoding="utf-8")
        render_pdf(md_text, str(pdf_path))
        json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        progress(88, "Scan-driven rewrite V2 stopped: author context required")
        return {
            "status": RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT.value,
            "md_path": str(md_path),
            "pdf_path": str(pdf_path),
            "json_path": str(json_path),
            "result": result_obj,
            "elapsed": elapsed,
        }
    candidate_rows: list[dict[str, Any]]
    if replay_candidate_records is not None:
        candidate_rows = _candidate_rows_from_replay(
            replay_candidate_records,
            original_text=original_text,
            original_report=original_report,
            required_ai_drop=required_ai_drop,
            target_ai_score=target_ai_score,
        )
    else:
        generated = _generate_candidates(
            original_text=original_text,
            scan_report=original_report,
            strategies=strategies,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=max(30, min(120, int(max_runtime_seconds))),
        )
        candidate_rows = []
        protected = detect_protected_spans(original_text)
        for index, generated_candidate in enumerate(generated, start=1):
            if time.time() - started >= max_runtime_seconds:
                break
            candidate_text = str(generated_candidate.get("text") or "").strip()
            if not candidate_text:
                continue
            if generated_candidate.get("local_filter_passed") is False:
                candidate_rows.append({
                    **generated_candidate,
                    "candidate_ai": None,
                    "decision": {
                        "lane": CandidateLane.REJECT.value,
                        "reason": "targeted_local_filter_rejected",
                        "rank": [],
                    },
                })
                continue
            local_score = None
            if (
                str(generated_candidate.get("strategy_kind") or "") == "targeted"
                and _use_paragraph_local_score_gate()
            ):
                local_score = _paragraph_local_score(generated_candidate)
                if isinstance(local_score, dict) and not local_score.get("improved"):
                    candidate_rows.append({
                        **generated_candidate,
                        "candidate_ai": None,
                        "paragraph_local_score": local_score,
                        "decision": {
                            "lane": CandidateLane.REJECT.value,
                            "reason": "paragraph_local_score_not_improved",
                            "rank": [],
                        },
                    })
                    continue
            semantic = check_semantic_drift(original_text, candidate_text, threshold=0.15)
            anchors_safe = protected_spans_preserved(original_text, candidate_text, protected)
            semantic_safe = bool(semantic.accepted)
            strategy_kind = str(generated_candidate.get("strategy_kind") or "")
            if not anchors_safe or not _semantic_scan_allowed(strategy_kind, semantic_safe):
                candidate_rows.append({
                    **generated_candidate,
                    "decision": {
                        "lane": CandidateLane.REJECT.value,
                        "reason": "protected_anchor_or_semantic_scan_guard_rejected",
                        "rank": [],
                    },
                    "semantic_safe": semantic_safe,
                    "protected_anchors_safe": bool(anchors_safe),
                    "semantic_review_required": not semantic_safe,
                    "semantic_similarity": getattr(semantic, "similarity", None),
                    "semantic_reasons": getattr(semantic, "reasons", None),
                })
                continue
            progress(min(80, 64 + index * 4), f"Scanning V2 candidate {index}")
            candidate_report = _scan_report(candidate_text)
            goal = evaluate_rewrite_goal(
                original_text=original_text,
                candidate_text=candidate_text,
                original_report=original_report,
                candidate_report=candidate_report,
            )
            decision = decide_candidate(
                goal=goal,
                original_report=original_report,
                candidate_report=candidate_report,
                reference_ai=reference_ai,
                required_ai_drop=required_ai_drop,
                target_ai_score=target_ai_score,
                semantic_safe=semantic_safe,
                quality_safe=anchors_safe,
                cost=index,
            )
            candidate_rows.append({
                **generated_candidate,
                "candidate_ai": _badge_ai(candidate_report),
                "candidate_wq": _badge_wq(candidate_report),
                "paragraph_local_score": local_score,
                "goal": goal.to_dict(),
                "decision": decision.to_dict(),
                "semantic_safe": semantic_safe,
                "protected_anchors_safe": bool(anchors_safe),
                "semantic_review_required": not semantic_safe,
                "semantic_similarity": getattr(semantic, "similarity", None),
                "semantic_reasons": getattr(semantic, "reasons", None),
                "report": candidate_report,
                "text": candidate_text,
            })
            if decision.lane == CandidateLane.GOAL_MET:
                break
        composed_text, composed_patches = _compose_full_doc_delta_winners(original_text, candidate_rows, reference_ai)
        if composed_patches and composed_text.strip() != original_text.strip():
            index = len(candidate_rows) + 1
            if time.time() - started < max_runtime_seconds:
                progress(min(84, 64 + index * 4), "Scanning V2 composed local winners")
                candidate_report = _scan_report(composed_text)
                goal = evaluate_rewrite_goal(
                    original_text=original_text,
                    candidate_text=composed_text,
                    original_report=original_report,
                    candidate_report=candidate_report,
                )
                semantic = check_semantic_drift(original_text, composed_text, threshold=0.15)
                anchors_safe = protected_spans_preserved(original_text, composed_text, protected)
                decision = decide_candidate(
                    goal=goal,
                    original_report=original_report,
                    candidate_report=candidate_report,
                    reference_ai=reference_ai,
                    required_ai_drop=required_ai_drop,
                    target_ai_score=target_ai_score,
                    semantic_safe=bool(semantic.accepted),
                    quality_safe=anchors_safe,
                    cost=index,
                )
                candidate_rows.append({
                    "strategy": "scan_targeted_composed_full_doc_delta_winners",
                    "strategy_kind": "targeted_composition",
                    "candidate_number": 1,
                    "candidate_ai": _badge_ai(candidate_report),
                    "candidate_wq": _badge_wq(candidate_report),
                    "composed_patches": composed_patches,
                    "goal": goal.to_dict(),
                    "decision": decision.to_dict(),
                    "semantic_safe": bool(semantic.accepted),
                    "protected_anchors_safe": bool(anchors_safe),
                    "semantic_review_required": not bool(semantic.accepted),
                    "semantic_similarity": getattr(semantic, "similarity", None),
                    "semantic_reasons": getattr(semantic, "reasons", None),
                    "report": candidate_report,
                    "text": composed_text,
                })
    best = select_best_candidate(candidate_rows)
    if best and (best.get("decision") or {}).get("lane") == CandidateLane.GOAL_MET.value:
        final_text = str(best.get("text") or original_text)
        final_report = best.get("report") if isinstance(best.get("report"), dict) else original_report
        final_goal = best.get("goal")
        public_status = RewriteGoalStatus.AI_MITIGATED.value
        converged = True
        convergence_reason = "rewrite_v2_strict_goal_met"
    else:
        final_text = original_text
        final_report = original_report
        preserved_goal = evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=original_text,
            original_report=original_report,
            candidate_report=original_report,
            no_text_change=True,
        )
        if candidate_rows:
            preserved_goal = evaluate_rewrite_goal(
                original_text=original_text,
                candidate_text=str((best or {}).get("text") or original_text),
                original_report=original_report,
                candidate_report=(best or {}).get("report") if isinstance((best or {}).get("report"), dict) else original_report,
            )
        status = (
            RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT
            if preserved_goal.status == RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT
            else RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE
        )
        final_goal = {**preserved_goal.to_dict(), "status": status.value, "goal_met": False}
        public_status = status.value
        converged = False
        convergence_reason = "rewrite_v2_no_candidate_met_strict_goal"
    elapsed = time.time() - started
    summary = {
        "rewrite_pipeline_version": "rewrite_v2_scan_driven",
        "outcome": public_status,
        "rewrite_goal_status": final_goal,
        "reference_ai": reference_ai,
        "required_ai_drop": required_ai_drop,
        "target_ai_score": target_ai_score,
        "strategy_trace": [strategy.to_dict() for strategy in strategies],
        "candidate_trace": [
            {key: value for key, value in row.items() if key not in {"text", "report"}}
            for row in candidate_rows
        ],
        "selected_candidate": {
            key: value for key, value in (best or {}).items() if key not in {"text", "report"}
        } if best else None,
        "diagnostic_candidate_text": (
            str(best.get("text") or "")
            if best and os.environ.get("DRAFTPROOF_REWRITE_V2_EXPOSE_DIAGNOSTIC_TEXT", "0").lower() in {"1", "true", "yes"}
            else None
        ),
        "stage_timings": [{
            "stage": "rewrite_v2_scan_driven",
            "seconds": round(elapsed, 3),
            "candidates": len(candidate_rows),
            "selected": public_status == RewriteGoalStatus.AI_MITIGATED.value,
        }],
        "detect_scan_original": original_report,
        "detect_scan_rewritten": final_report,
        "final_text": final_text,
    }
    sentence_comparison = _sentence_comparison(original_text, final_text)
    result_obj = SimpleNamespace(
        summary=summary,
        sentence_comparison=sentence_comparison,
        rewrite_plan=None,
        mp_result=SimpleNamespace(
            original_text=original_text,
            final_text=final_text,
            converged=converged,
            convergence_reason=convergence_reason,
            passes=[],
        ),
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"draftproof_rewrite_v2_{ts}.md"
    pdf_path = out_dir / f"draftproof_rewrite_v2_{ts}.pdf"
    json_path = out_dir / f"draftproof_rewrite_v2_{ts}.json"
    md_text = render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False)
    md_path.write_text(md_text, encoding="utf-8")
    render_pdf(md_text, str(pdf_path))
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    progress(88, "Scan-driven rewrite V2 complete")
    return {
        "status": public_status,
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
        "result": result_obj,
        "elapsed": elapsed,
    }
