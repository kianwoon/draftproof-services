"""External-calibrated rewrite pipeline V3 entrypoint."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from detect.run import DetectionRunner
from llm.gateway import LLMConfig, LLMGateway
from report.pdf import render_pdf
from report.render_rewrite import render_rewrite_report
from report.report import ReportBuilder, report_to_dict
from rewrite.guards import check_semantic_drift
from rewrite_v2.contracts import build_rewrite_contract
from rewrite_v2.goal_contract import RewriteGoalStatus, evaluate_rewrite_goal
from rewrite_v2.pipeline import _badge_ai, _badge_wq, _extract_original_text, _sentence_comparison
from rewrite_v2.selection import CandidateLane, decide_candidate

from .anchor_validation import validate_v3_candidate
from .authorship_window_gate import select_authorship_window_targets
from .candidate_loop import CandidateAction, decide_next_action, select_candidate_index
from .compression_policy import compression_policy_for_family, compression_status
from .document_units import compose_units, document_units, word_count
from .external_proxy import evaluate_external_proxy
from .layers.boundary_adapter import build_boundary_adapter_prompt
from .layers.cited_practice_voice import build_cited_practice_voice_chunk_prompt, build_cited_practice_voice_prompt
from .layers.contract_repair import build_contract_repair_prompt
from .layers.contrast_boundary import build_contrast_boundary_prompt, extract_contrast_boundary_output
from .layers.document_rhythm import build_document_rhythm_chunk_prompt, build_document_rhythm_prompt
from .layers.plain_reasoning_broad_prose import build_plain_reasoning_broad_prose_prompt
from .layers.recovery_revision import build_recovery_revision_prompt
from .layers.structure_repair import build_structure_repair_prompt
from .layers.authorship_window_repair import (
    apply_authorship_window_replacements,
    build_authorship_window_repair_prompt,
    extract_authorship_window_replacements,
)
from .output_cleaning import clean_v3_candidate_output
from .portfolio import select_portfolio_candidate
from .router import route_from_scan_contract
from .scanner_contract import RewriteRiskClass, ScanContract, build_scan_contract
from .style_library import examples_for_family
from .strategy_plan import build_strategy_plan


def _scan_report(text: str) -> dict[str, Any]:
    detect_report = DetectionRunner().run_all(text)
    builder = ReportBuilder()
    builder.add_detection_report(detect_report)
    if detect_report.postprocess_results:
        builder.add_postprocess_results(detect_report.postprocess_results)
    builder.set_meta(scan_time=0, original_text=text)
    payload = report_to_dict(builder.build())
    payload["input_text"] = text
    return payload


def _topk(report: dict | None) -> float | None:
    value = (((report or {}).get("ai_risk_badge") or {}).get("ai_components") or {}).get("topk_pattern_raw")
    return float(value) if isinstance(value, (int, float)) else None


def _authorship_window_profile(report: dict | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    direct = report.get("authorship_window_profile")
    if isinstance(direct, dict) and direct:
        return direct
    scan_intel = report.get("scan_intelligence") if isinstance(report.get("scan_intelligence"), dict) else {}
    profile = scan_intel.get("authorship_window_profile")
    if isinstance(profile, dict) and profile:
        return profile
    document = scan_intel.get("document") if isinstance(scan_intel.get("document"), dict) else {}
    profile = document.get("authorship_window_profile")
    return profile if isinstance(profile, dict) else {}


def _logical_unit_count(original_text: str, report: dict | None) -> int:
    return len(_logical_source_units(original_text, report))


def _logical_source_units(original_text: str, report: dict | None) -> list[dict[str, Any]]:
    structural_units = document_units(original_text)
    structural_payload = [unit.to_dict() for unit in structural_units]
    profile = _authorship_window_profile(report)
    windows = profile.get("windows") if isinstance(profile.get("windows"), list) else []
    if len(structural_units) != 1 or len(windows) <= 1:
        return structural_payload

    text = str(original_text or "")
    sorted_windows = sorted(
        [window for window in windows if isinstance(window, dict)],
        key=lambda window: int(window.get("start_index") or 0),
    )
    units: list[dict[str, Any]] = []
    previous_end = 0
    for index, window in enumerate(sorted_windows, start=1):
        raw_end = int(window.get("end_index") or previous_end)
        end = max(previous_end, min(len(text), raw_end))
        if index == len(sorted_windows):
            end = len(text)
        unit_text = text[previous_end:end].strip()
        previous_end = end
        if not unit_text:
            continue
        units.append({
            "unit_id": f"u{len(units) + 1}",
            "text": unit_text,
            "word_count": word_count(unit_text),
            "is_heading": False,
            "source_window_id": window.get("window_id"),
            "source_paragraph_id": window.get("paragraph_id"),
        })
    return units or structural_payload


def _single_shot_word_limit() -> int:
    try:
        return max(300, int(os.environ.get("DRAFTPROOF_REWRITE_V3_SINGLE_SHOT_WORD_LIMIT", "1600") or 1600))
    except (TypeError, ValueError):
        return 1600


def _chunk_word_limit() -> int:
    try:
        return max(250, int(os.environ.get("DRAFTPROOF_REWRITE_V3_CHUNK_WORD_LIMIT", "900") or 900))
    except (TypeError, ValueError):
        return 900


def _llm_timeout_seconds() -> int:
    try:
        return max(20, int(os.environ.get("DRAFTPROOF_REWRITE_V3_LLM_TIMEOUT_SECONDS", "160") or 160))
    except (TypeError, ValueError):
        return 160


def _max_tokens_for_words(words: int) -> int:
    return max(1200, min(12000, int(words * 2.4) + 900))


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _compression_accepted(compression: dict[str, Any]) -> bool:
    if compression.get("in_band"):
        return True
    policy = compression.get("policy") if isinstance(compression.get("policy"), dict) else {}
    source_words = int(compression.get("source_words") or 0)
    candidate_words = int(compression.get("candidate_words") or 0)
    tolerance = max(0, int(round(source_words * max(0.0, _float_env("DRAFTPROOF_REWRITE_V3_COMPRESSION_TOLERANCE_RATIO", 0.02)))))
    if compression.get("status") == "above_ceiling":
        return candidate_words <= int(policy.get("max_words") or 0) + tolerance
    if compression.get("status") == "below_floor":
        return candidate_words >= int(policy.get("min_words") or 0) - tolerance
    return False


def _family_for_route(content_mode: str, contract_anchor_count: int) -> str:
    if content_mode == "academic_cited_text" or contract_anchor_count >= 3:
        return "cited_practice_voice"
    return "document_rhythm"


def _should_use_chunked_generation(
    *,
    source_words: int,
    scan_contract: ScanContract,
    v3_route: Any,
    exact_anchor_count: int,
) -> bool:
    if source_words > _single_shot_word_limit():
        return True
    if exact_anchor_count > 0:
        return True
    if scan_contract.anchor_preservation_pressure >= 0.55:
        return True
    protected_classes = {
        RewriteRiskClass.CITED_ACADEMIC,
        RewriteRiskClass.TECHNICAL_STRUCTURED,
        RewriteRiskClass.REGULATED_POLICY,
    }
    if v3_route.primary_class in protected_classes:
        return True
    return (
        v3_route.primary_class == RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY
        and scan_contract.evidence_anchor_score >= 0.5
    )


def _should_force_unit_chunks(*, scan_contract: ScanContract, v3_route: Any, exact_anchor_count: int) -> bool:
    if exact_anchor_count > 0:
        return True
    if scan_contract.anchor_preservation_pressure >= 0.55:
        return True
    protected_classes = {
        RewriteRiskClass.CITED_ACADEMIC,
        RewriteRiskClass.TECHNICAL_STRUCTURED,
        RewriteRiskClass.REGULATED_POLICY,
    }
    if v3_route.primary_class in protected_classes:
        return True
    return (
        v3_route.primary_class == RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY
        and scan_contract.evidence_anchor_score >= 0.5
    )


def _restore_exact_quote_anchors(text: str, contract: Any) -> str:
    repaired = str(text or "")
    quote_chars = {'"', "'", "“", "”", "‘", "’"}
    for anchor in getattr(contract, "anchors", []) or []:
        anchor_text = str(getattr(anchor, "text", "") or "")
        if not anchor_text or anchor_text in repaired:
            continue
        kind = str(getattr(anchor, "kind", "") or "")
        severity = getattr(getattr(anchor, "severity", None), "value", "")
        if kind != "direct_quote" or severity != "hard_exact":
            continue
        inner = anchor_text.strip().strip("\"'“”‘’").strip()
        if not inner:
            continue
        start = repaired.find(inner)
        if start < 0:
            continue
        left = start - 1 if start > 0 and repaired[start - 1] in quote_chars else start
        right = start + len(inner)
        while right < len(repaired) and repaired[right] in ".,;:!?":
            right += 1
        if right < len(repaired) and repaired[right] in quote_chars:
            right += 1
        repaired = repaired[:left] + anchor_text + repaired[right:]
    return repaired


def _v3_content_mode(scan_contract: ScanContract, v3_route: Any) -> tuple[str, dict[str, Any]]:
    scan_mode = str(scan_contract.content_mode or "").strip()
    confidence = float(scan_contract.content_mode_confidence or 0.0)
    if scan_mode and scan_mode != "unknown" and confidence > 0:
        return scan_mode, {
            "content_mode": scan_mode,
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "reasons": ["scan_contract_content_mode", *list(v3_route.reasons)],
            "mode_scores": list(scan_contract.mode_scores),
            "router_source": "rewrite_v3_scan_contract",
        }

    fallback_by_class = {
        RewriteRiskClass.BROAD_PROSE: "broad_explanatory_essay",
        RewriteRiskClass.CITED_ACADEMIC: "academic_cited_text",
        RewriteRiskClass.TECHNICAL_STRUCTURED: "technical_content",
        RewriteRiskClass.REGULATED_POLICY: "regulated_policy_content",
        RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY: "quote_heavy",
        RewriteRiskClass.PERSONAL_REFLECTIVE: "personal_reflection",
        RewriteRiskClass.CREATIVE_MARKETING: "creative_marketing",
        RewriteRiskClass.SHORT_OR_SPARSE: "short_text",
    }
    content_mode = fallback_by_class.get(v3_route.primary_class, "generic_expository")
    return content_mode, {
        "content_mode": content_mode,
        "confidence": v3_route.confidence,
        "reasons": ["derived_from_v3_route", *list(v3_route.reasons)],
        "mode_scores": [
            {"content_mode": content_mode, "score": v3_route.confidence, "source": "rewrite_v3_route"}
        ],
        "router_source": "rewrite_v3_route_fallback",
    }


def _gateway(api_key: str | None, model: str | None, base_url: str | None, *, max_tokens: int) -> LLMGateway:
    return LLMGateway(LLMConfig(
        api_key=api_key,
        model=model or os.environ.get("LLM_MODEL") or "deepseek/deepseek-chat",
        base_url=base_url or os.environ.get("LLM_BASE_URL", ""),
        max_tokens=max_tokens,
        temperature=float(os.environ.get("DRAFTPROOF_REWRITE_V3_TEMPERATURE", "0.82") or 0.82),
        top_p=float(os.environ.get("DRAFTPROOF_REWRITE_V3_TOP_P", "0.93") or 0.93),
        presence_penalty=float(os.environ.get("DRAFTPROOF_REWRITE_V3_PRESENCE_PENALTY", "0.42") or 0.42),
        frequency_penalty=float(os.environ.get("DRAFTPROOF_REWRITE_V3_FREQUENCY_PENALTY", "0.40") or 0.40),
        repetition_penalty=float(os.environ.get("DRAFTPROOF_REWRITE_V3_REPETITION_PENALTY", "1.07") or 1.07),
        timeout=_llm_timeout_seconds(),
        max_retries=1,
    ))


def _generate_single_candidate(
    *,
    original_text: str,
    family: str,
    contract: Any,
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    examples = examples_for_family(family)
    if family == "cited_practice_voice":
        prompt = build_cited_practice_voice_prompt(
            original_text=original_text,
            contract=contract,
            compression_policy=compression_policy,
            style_examples=examples,
        )
    else:
        prompt = build_document_rhythm_prompt(
            original_text=original_text,
            compression_policy=compression_policy,
            style_examples=examples,
        )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the rewritten document as plain text.").content)


def _generate_recovery_candidate(
    *,
    original_text: str,
    failed_candidate: str,
    family: str,
    proxy_feedback: dict[str, Any],
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_recovery_revision_prompt(
        original_text=original_text,
        failed_candidate=failed_candidate,
        strategy_family=family,
        proxy_feedback=proxy_feedback,
        compression_policy=compression_policy,
        style_examples=examples_for_family(family),
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the rewritten document as plain text.").content)


def _generate_contract_repair_candidate(
    *,
    original_text: str,
    failed_candidate: str,
    family: str,
    candidate_trace: dict[str, Any],
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_contract_repair_prompt(
        original_text=original_text,
        failed_candidate=failed_candidate,
        strategy_family=family,
        candidate_trace=candidate_trace,
        compression_policy=compression_policy,
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the repaired rewritten document as plain text.").content)


def _generate_boundary_candidate(
    *,
    original_text: str,
    failed_candidates: list[str],
    family: str,
    proxy_feedback: list[dict[str, Any]],
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_boundary_adapter_prompt(
        original_text=original_text,
        failed_candidates=failed_candidates,
        strategy_family=family,
        proxy_feedback=proxy_feedback,
        compression_policy=compression_policy,
        style_examples=examples_for_family(family),
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the rewritten document as plain text.").content)


def _generate_contrast_boundary_candidate(
    *,
    original_text: str,
    failed_candidate: str,
    family: str,
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_contrast_boundary_prompt(
        original_text=original_text,
        failed_candidate=failed_candidate,
        family=family,
        compression_policy=compression_policy,
        style_examples=examples_for_family(family),
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    raw = gateway.chat(prompt, system="Return only valid JSON with rewritten_document and notes.").content
    return clean_v3_candidate_output(extract_contrast_boundary_output(raw))


def _generate_plain_reasoning_candidate(
    *,
    original_text: str,
    failed_candidates: list[str],
    family: str,
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_plain_reasoning_broad_prose_prompt(
        original_text=original_text,
        failed_candidates=failed_candidates,
        compression_policy=compression_policy,
        style_examples=examples_for_family(family),
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the rewritten document as plain text.").content)


def _max_authorship_window_repair_targets() -> int:
    try:
        return max(1, min(4, int(os.environ.get("DRAFTPROOF_REWRITE_V3_WINDOW_REPAIR_TARGETS", "2") or 2)))
    except (TypeError, ValueError):
        return 2


def _generate_authorship_window_repair_candidate(
    *,
    candidate_text: str,
    candidate_trace: dict[str, Any],
    family: str,
    contract: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    profile = candidate_trace.get("authorship_window_profile") if isinstance(candidate_trace.get("authorship_window_profile"), dict) else {}
    target_windows = select_authorship_window_targets(
        profile,
        max_targets=_max_authorship_window_repair_targets(),
    )
    if not target_windows:
        return str(candidate_text or "")
    prompt = build_authorship_window_repair_prompt(
        candidate_text=candidate_text,
        target_windows=target_windows,
        strategy_family=family,
        contract=contract,
    )
    target_words = sum(int(window.get("word_count") or 0) for window in target_windows)
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(max(120, target_words + 120)))
    raw = gateway.chat(prompt, system="Return only valid JSON with a replacements array.").content
    replacements = extract_authorship_window_replacements(raw)
    if not replacements:
        return str(candidate_text or "")
    return clean_v3_candidate_output(apply_authorship_window_replacements(
        candidate_text=candidate_text,
        target_windows=target_windows,
        replacements=replacements,
    ))


def _generate_structure_repair_candidate(
    *,
    original_text: str,
    candidate_text: str,
    validation: dict[str, Any],
    expected_unit_count: int | None,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_structure_repair_prompt(
        source_text=original_text,
        candidate_text=candidate_text,
        validation=validation,
        expected_unit_count=expected_unit_count,
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(word_count(candidate_text)))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the repaired text as plain text.").content)


def _unit_chunks(source_units: list[dict[str, Any]] | str, *, force_unit_chunks: bool = False) -> list[list[dict[str, Any]]]:
    if isinstance(source_units, str):
        units_payload = [unit.to_dict() for unit in document_units(source_units)]
    else:
        units_payload = source_units
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_words = 0
    limit = _chunk_word_limit()
    for payload in units_payload:
        unit_words = int(payload.get("word_count") or word_count(str(payload.get("text") or "")))
        if force_unit_chunks:
            chunks.append([payload])
            continue
        if current and current_words + unit_words > limit:
            chunks.append(current)
            current = []
            current_words = 0
        current.append(payload)
        current_words += unit_words
    if current:
        chunks.append(current)
    return chunks


def _normalize_chunk_unit_boundaries(text: str, *, expected_units: int) -> str:
    units = document_units(text)
    if expected_units == 1 and len(units) > 1:
        return "\n".join(unit.text.strip() for unit in units if unit.text.strip()).strip()
    return str(text or "").strip()


def _generate_chunked_candidate(
    *,
    original_text: str,
    source_units: list[dict[str, Any]],
    family: str,
    contract: Any,
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    force_unit_chunks: bool = False,
) -> str:
    examples = examples_for_family(family)
    global_plan = {
        "family": family,
        "source_words": word_count(original_text),
        "source_units": len(source_units),
        "compression_policy": compression_policy.to_dict(),
        "instruction": "Keep chunk outputs compatible when joined; do not add global conclusions unless present in the chunk.",
    }
    rewritten_chunks: list[str] = []
    for chunk in _unit_chunks(source_units, force_unit_chunks=force_unit_chunks):
        chunk_words = sum(int(unit.get("word_count") or 0) for unit in chunk)
        chunk_policy = compression_policy_for_family(family, chunk_words)
        if family == "cited_practice_voice":
            prompt = build_cited_practice_voice_chunk_prompt(
                source_units=chunk,
                contract=contract,
                global_plan=global_plan,
                compression_policy=chunk_policy,
                style_examples=examples,
            )
        else:
            prompt = build_document_rhythm_chunk_prompt(
                source_units=chunk,
                global_plan=global_plan,
                compression_policy=chunk_policy,
                style_examples=examples,
            )
        gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(chunk_policy.max_words))
        rewritten_chunk = clean_v3_candidate_output(gateway.chat(prompt, system="Return only rewritten plain text for this chunk.").content)
        rewritten_chunk = _normalize_chunk_unit_boundaries(rewritten_chunk, expected_units=len(chunk))
        rewritten_chunks.append(_restore_exact_quote_anchors(rewritten_chunk, contract))
    return _restore_exact_quote_anchors(compose_units(rewritten_chunks), contract)


def _candidate_from_replay(record: dict[str, Any], original_report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    text = str(record.get("text") or record.get("candidate_text") or "").strip()
    report = record.get("report") if isinstance(record.get("report"), dict) else None
    if report is None and isinstance(record.get("ai"), (int, float)):
        report = {
            "ai_risk_badge": {
                "ai_likelihood_score": record.get("ai"),
                "writing_quality_score": record.get("wq") or record.get("writing_quality"),
                "ai_components": {"topk_pattern_raw": record.get("topk")},
            },
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
        }
    return text, report or original_report


def run_rewrite_pipeline_v3(
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

    progress(62, "Starting external-calibrated rewrite V3")
    original_text = _extract_original_text(detect_json)
    original_report = detect_json
    scan_contract = build_scan_contract(original_report, original_text)
    v3_route = route_from_scan_contract(scan_contract)
    strategy_plan = build_strategy_plan(v3_route, scan_contract)
    content_mode, content_router_trace = _v3_content_mode(scan_contract, v3_route)
    contract = build_rewrite_contract(original_text, content_mode=content_mode)
    exact_anchor_count = sum(1 for anchor in contract.anchors if anchor.severity.value == "hard_exact")
    family = _family_for_route(content_mode, exact_anchor_count)
    compression_policy = compression_policy_for_family(family, word_count(original_text))
    source_generation_units = _logical_source_units(original_text, original_report)
    force_unit_chunks = _should_force_unit_chunks(
        scan_contract=scan_contract,
        v3_route=v3_route,
        exact_anchor_count=exact_anchor_count,
    )
    generation_mode = "replay"
    candidate_text = ""
    candidate_report: dict[str, Any] | None = None
    generation_error = None

    if replay_candidate_records:
        candidate_text, candidate_report = _candidate_from_replay(replay_candidate_records[0], original_report)
    elif full_rewrite_allowed:
        try:
            if _should_use_chunked_generation(
                source_words=word_count(original_text),
                scan_contract=scan_contract,
                v3_route=v3_route,
                exact_anchor_count=exact_anchor_count,
            ):
                generation_mode = "chunked"
                candidate_text = _generate_chunked_candidate(
                    original_text=original_text,
                    source_units=source_generation_units,
                    family=family,
                    contract=contract,
                    compression_policy=compression_policy,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    force_unit_chunks=force_unit_chunks,
                )
            else:
                generation_mode = "single_shot"
                candidate_text = _generate_single_candidate(
                    original_text=original_text,
                    family=family,
                    contract=contract,
                    compression_policy=compression_policy,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                )
        except Exception as exc:
            generation_error = str(exc)

    reference_ai = _badge_ai(original_report)
    target_ai_score = float(reference_ai) - float(required_ai_drop) if isinstance(reference_ai, (int, float)) else None
    expected_unit_count = len(source_generation_units)

    def assess_candidate(
        *,
        text: str,
        report: dict[str, Any] | None,
        mode: str,
        cost: int,
        error: str | None = None,
    ) -> dict[str, Any]:
        text = _restore_exact_quote_anchors(text, contract)
        validation_result = validate_v3_candidate(
            original_text=original_text,
            candidate_text=text,
            contract=contract,
            require_unit_count=True,
            expected_unit_count=expected_unit_count,
        )
        compression_result = compression_status(original_text, text, compression_policy)
        compression_ok = _compression_accepted(compression_result)
        should_scan_candidate = bool(text and validation_result.passed and compression_ok)
        scanned_report = report
        if should_scan_candidate and scanned_report is None:
            progress(78, f"Scanning V3 {mode} candidate")
            scanned_report = _scan_report(text)
        elif scanned_report is None:
            scanned_report = original_report
        goal_result = evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=text or original_text,
            original_report=original_report,
            candidate_report=scanned_report,
        )
        semantic_result = check_semantic_drift(original_text, text or original_text, threshold=0.15)
        decision_result = decide_candidate(
            goal=goal_result,
            original_report=original_report,
            candidate_report=scanned_report,
            reference_ai=reference_ai,
            required_ai_drop=required_ai_drop,
            target_ai_score=target_ai_score,
            semantic_safe=bool(semantic_result.accepted),
            quality_safe=bool(validation_result.passed and compression_ok),
            cost=cost,
        )
        proxy_result = evaluate_external_proxy(
            family=family,
            reference_ai=reference_ai,
            candidate_ai=_badge_ai(scanned_report),
            reference_wq=_badge_wq(original_report),
            candidate_wq=_badge_wq(scanned_report),
            reference_topk=_topk(original_report),
            candidate_topk=_topk(scanned_report),
            candidate_authorship_profile=_authorship_window_profile(scanned_report),
            compression=compression_result,
            validation_passed=bool(validation_result.passed),
            compression_accepted=bool(compression_ok),
            semantic_safe=bool(semantic_result.accepted),
        )
        return {
            "text": text,
            "report": scanned_report,
            "should_scan": should_scan_candidate,
            "strict_selected": decision_result.lane == CandidateLane.GOAL_MET,
            "external_selected": bool(should_scan_candidate and text and proxy_result.accepted),
            "trace": {
                "strategy_family": family,
                "generation_mode": mode,
                "candidate_ai": _badge_ai(scanned_report),
                "candidate_wq": _badge_wq(scanned_report),
                "candidate_topk": _topk(scanned_report),
                "authorship_window_profile": _authorship_window_profile(scanned_report),
                "validation": validation_result.to_dict(),
                "compression": compression_result,
                "compression_accepted": compression_ok,
                "goal": goal_result.to_dict(),
                "decision": decision_result.to_dict(),
                "semantic_safe": bool(semantic_result.accepted),
                "semantic_similarity": getattr(semantic_result, "similarity", None),
                "external_proxy": proxy_result.to_dict(),
                "generation_error": error,
            },
        }

    candidate_evaluations: list[dict[str, Any]] = []
    candidate_evaluations.append(assess_candidate(
        text=candidate_text,
        report=candidate_report,
        mode=generation_mode,
        cost=1,
        error=generation_error,
    ))
    loop_trace = []
    tried_actions: set[CandidateAction] = set()
    if replay_candidate_records and len(replay_candidate_records) > 1:
        for index, record in enumerate(replay_candidate_records[1:], start=2):
            replay_text, replay_report = _candidate_from_replay(record, original_report)
            candidate_evaluations.append(assess_candidate(
                text=replay_text,
                report=replay_report,
                mode=f"replay_recovery_{index}",
                cost=index,
            ))
    elif full_rewrite_allowed:
        while (time.time() - started) < max_runtime_seconds and len(candidate_evaluations) < 5:
            loop_decision = decide_next_action(
                candidate_evaluations,
                has_positive_boundaries=bool(examples_for_family(family).get("positive") or []),
                tried_actions=tried_actions,
            )
            loop_trace.append(loop_decision.to_dict())
            if loop_decision.action in {
                CandidateAction.ACCEPT_STRICT,
                CandidateAction.ACCEPT_EXTERNAL,
                CandidateAction.RETURN_BEST_FOR_REVIEW,
            }:
                break
            tried_actions.add(loop_decision.action)
            source_item = candidate_evaluations[loop_decision.source_index]
            try:
                if loop_decision.action == CandidateAction.REPAIR_STRUCTURE:
                    progress(84, "Repairing V3 candidate structure")
                    new_text = _generate_structure_repair_candidate(
                        original_text=original_text,
                        candidate_text=str(source_item.get("text") or ""),
                        validation=source_item["trace"]["validation"],
                        expected_unit_count=expected_unit_count,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "structure_repair"
                elif loop_decision.action == CandidateAction.REPAIR_CONTRACT:
                    progress(81, "Repairing V3 candidate contract")
                    new_text = _generate_contract_repair_candidate(
                        original_text=original_text,
                        failed_candidate=str(source_item.get("text") or candidate_text),
                        family=family,
                        candidate_trace=source_item["trace"],
                        compression_policy=compression_policy,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "contract_repair"
                elif loop_decision.action == CandidateAction.ADAPT_BOUNDARY:
                    progress(82, "Running V3 boundary adapter")
                    new_text = _generate_boundary_candidate(
                        original_text=original_text,
                        failed_candidates=[str(item.get("text") or "") for item in candidate_evaluations if item.get("text")],
                        family=family,
                        proxy_feedback=[item["trace"]["external_proxy"] for item in candidate_evaluations],
                        compression_policy=compression_policy,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "boundary_adapter"
                elif loop_decision.action == CandidateAction.CONTRAST_BOUNDARY:
                    progress(83, "Running V3 contrast boundary")
                    new_text = _generate_contrast_boundary_candidate(
                        original_text=original_text,
                        failed_candidate=str(source_item.get("text") or candidate_text),
                        family=family,
                        compression_policy=compression_policy,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "contrast_boundary"
                elif loop_decision.action == CandidateAction.PLAIN_REASONING:
                    progress(84, "Running V3 plain reasoning broad prose")
                    new_text = _generate_plain_reasoning_candidate(
                        original_text=original_text,
                        failed_candidates=[str(item.get("text") or "") for item in candidate_evaluations if item.get("text")],
                        family=family,
                        compression_policy=compression_policy,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "plain_reasoning_broad_prose"
                elif loop_decision.action == CandidateAction.REPAIR_AUTHORSHIP_WINDOWS:
                    progress(85, "Repairing V3 authorship windows")
                    new_text = _generate_authorship_window_repair_candidate(
                        candidate_text=str(source_item.get("text") or candidate_text),
                        candidate_trace=source_item["trace"],
                        family=family,
                        contract=contract,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "authorship_window_repair"
                else:
                    progress(80, "Running V3 targeted repair")
                    new_text = _generate_recovery_candidate(
                        original_text=original_text,
                        failed_candidate=str(source_item.get("text") or candidate_text),
                        family=family,
                        proxy_feedback={
                            "loop_decision": loop_decision.to_dict(),
                            "external_proxy": source_item["trace"]["external_proxy"],
                        },
                        compression_policy=compression_policy,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "targeted_repair"
                candidate_evaluations.append(assess_candidate(
                    text=new_text,
                    report=None,
                    mode=mode,
                    cost=len(candidate_evaluations) + 1,
                ))
            except Exception as exc:
                candidate_evaluations.append(assess_candidate(
                    text="",
                    report=original_report,
                    mode=loop_decision.action.value,
                    cost=len(candidate_evaluations) + 1,
                    error=str(exc),
                ))

    selected_index, selected_action, selected_reason = select_candidate_index(candidate_evaluations)
    portfolio_scores: list[dict[str, Any]] = []
    if selected_action == CandidateAction.RETURN_BEST_FOR_REVIEW:
        selected_index, portfolio_scores = select_portfolio_candidate(candidate_evaluations, family=family)
    selected = candidate_evaluations[selected_index]
    public_status = {
        CandidateAction.ACCEPT_STRICT: RewriteGoalStatus.AI_MITIGATED.value,
        CandidateAction.ACCEPT_EXTERNAL: "external_calibrated_candidate_applied",
        CandidateAction.RETURN_BEST_FOR_REVIEW: "rewrite_candidate_generated_needs_external_review",
    }[selected_action]
    converged = selected_action == CandidateAction.ACCEPT_STRICT
    convergence_reason = {
        CandidateAction.ACCEPT_STRICT: "rewrite_v3_strict_goal_met",
        CandidateAction.ACCEPT_EXTERNAL: "rewrite_v3_external_calibrated_candidate_applied",
        CandidateAction.RETURN_BEST_FOR_REVIEW: "rewrite_v3_best_candidate_needs_external_review",
    }[selected_action]
    final_text = selected["text"] or original_text
    final_report = selected["report"] or original_report

    elapsed = time.time() - started
    candidate_trace = [item["trace"] for item in candidate_evaluations]
    final_goal = selected["trace"]["goal"]
    if public_status != RewriteGoalStatus.AI_MITIGATED.value:
        final_goal = {
            **final_goal,
            "status": (
                RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value
                if public_status != "rewrite_candidate_generated_needs_external_review"
                else public_status
            ),
            "goal_met": False,
            "reason": convergence_reason,
        }
    summary = {
        "rewrite_pipeline_version": "rewrite_v3_external_calibrated",
        "outcome": public_status,
        "strict_goal_status": final_goal.get("status"),
        "rewrite_goal_status": final_goal,
        "reference_ai": reference_ai,
        "required_ai_drop": required_ai_drop,
        "target_ai_score": target_ai_score,
        "content_router_trace": content_router_trace,
        "scan_contract": scan_contract.to_dict(),
        "v3_route": v3_route.to_dict(),
        "v3_strategy_plan": strategy_plan.to_dict(),
        "strategy_trace": [{
            "strategy_family": family,
            "external_calibrated": True,
            "generation_mode": generation_mode,
            "compression_policy": compression_policy.to_dict(),
            "single_shot_word_limit": _single_shot_word_limit(),
            "chunk_word_limit": _chunk_word_limit(),
            "force_unit_chunks": force_unit_chunks,
        }],
        "candidate_trace": candidate_trace,
        "candidate_loop_trace": loop_trace,
        "portfolio_scores": portfolio_scores,
        "selected_candidate": candidate_trace[selected_index],
        "diagnostic_candidate_text": (
            final_text
            if os.environ.get("DRAFTPROOF_REWRITE_V3_EXPOSE_DIAGNOSTIC_TEXT", "0").lower() in {"1", "true", "yes"}
            else None
        ),
        "stage_timings": [{
            "stage": "rewrite_v3_external_calibrated",
            "seconds": round(elapsed, 3),
            "selected": public_status != RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value,
            "stop_reason": convergence_reason,
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
    md_path = out_dir / f"draftproof_rewrite_v3_{ts}.md"
    pdf_path = out_dir / f"draftproof_rewrite_v3_{ts}.pdf"
    json_path = out_dir / f"draftproof_rewrite_v3_{ts}.json"
    md_text = render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False)
    md_path.write_text(md_text, encoding="utf-8")
    render_pdf(md_text, str(pdf_path))
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    progress(88, "External-calibrated rewrite V3 complete")
    return {
        "status": public_status,
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
        "result": result_obj,
        "elapsed": elapsed,
    }
