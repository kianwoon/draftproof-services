from __future__ import annotations

from dataclasses import replace
from typing import Any

from .plan import Plan


def extract_report_signal_contracts(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    rows: list[dict[str, Any]] = []
    targets = _target_excerpts(report)
    for signal in _core_signals(report):
        key = str(signal.get("key") or "").strip()
        score = _score(signal.get("score"))
        contract = _contract_for_signal(key, score)
        if contract:
            rows.append(contract)
    contracts = _dedupe_contracts(rows)[:8]
    if targets:
        for contract in contracts:
            contract["target_excerpts"] = targets
    return contracts


def apply_report_signal_contracts(plan: Plan, contracts: list[dict[str, Any]] | None) -> Plan:
    rows = [row for row in (contracts or []) if isinstance(row, dict)]
    if not rows:
        return plan
    route = dict(plan.ai_safe_route)
    route["document_signal_contracts"] = rows
    route["document_signal_instruction"] = (
        "Resolve paragraph findings while also moving these document-level signal groups. "
        "Use Author-Proxy bridges for grounding and human/context anchors; mark inferred bridges for review instead of blocking generation."
    )
    decision = dict(route.get("llm_planner_decision") or {})
    decision["document_signal_contracts"] = rows
    route["llm_planner_decision"] = decision
    return replace(plan, ai_safe_route=route)


def _core_signals(report: dict[str, Any]) -> list[dict[str, Any]]:
    intelligence = report.get("scan_intelligence") if isinstance(report.get("scan_intelligence"), dict) else {}
    transformation = intelligence.get("transformation") if isinstance(intelligence.get("transformation"), dict) else {}
    signals = transformation.get("core_signals") if isinstance(transformation.get("core_signals"), list) else []
    rows = [row for row in signals if isinstance(row, dict)]
    mitigation = report.get("ai_mitigation") if isinstance(report.get("ai_mitigation"), dict) else {}
    actions = mitigation.get("component_actions") if isinstance(mitigation.get("component_actions"), list) else []
    rows.extend(_component_action_signals(actions))
    rows.extend(_finding_signals(_all_findings(report)))
    if rows:
        return rows
    badge = report.get("ai_risk_badge") if isinstance(report.get("ai_risk_badge"), dict) else {}
    classification = badge.get("transformation_classification") if isinstance(badge.get("transformation_classification"), dict) else {}
    features = classification.get("features") if isinstance(classification.get("features"), dict) else {}
    return [{"key": key, "score": _score(value) * 100 if 0 <= _score(value) <= 1 else _score(value)} for key, value in features.items()]


def _contract_for_signal(key: str, score: float) -> dict[str, Any] | None:
    normalized = key.casefold()
    if score < 25:
        return None
    if "semantic_uniformity" in normalized or "uniform" in normalized or "discourse" in normalized or "smooth" in normalized:
        return _row("thinking_path_route", score, "show uneven reasoning through source basis, concrete detail, interpretation, and careful close")
    if "expansion" in normalized or "patchwork" in normalized or "paraphrase" in normalized or "drift" in normalized or "section_style_variance" in normalized:
        return _row("source_coverage_route", score, "avoid polished expansion and maintain source-level vocabulary, coverage, and paragraph voice")
    if "topk" in normalized or "predictability" in normalized or "ai_generation" in normalized or normalized == "ai_likelihood":
        return _row("predictability_route", score, "change opener, clause route, sentence boundary, and list route before word choice")
    if "ground" in normalized or "citation" in normalized or "source_similarity" in normalized:
        return _row("grounding_route", score, "keep each claim next to submitted source, citation, named reference, or reviewable author-proxy bridge")
    if "unsupported" in normalized or "broad_claim" in normalized:
        return _row("claim_scope_route", score, "narrow broad or unsupported claims with submitted scope, source support, or reviewable author-proxy bridge")
    if "generic" in normalized or "specificity" in normalized or "lived_detail" in normalized:
        return _row("context_specificity_route", score, "replace reusable assertions with submitted setting, task, method, source-use, observation, or decision anchors")
    if "human_anchor" in normalized or "authorship" in normalized:
        return _row("human_anchor_route", score, "add submitted role, setting, observation, comparison, source-use, or decision anchor where the paragraph supports it")
    return None


def _component_action_signals(actions: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        key = str(action.get("component") or action.get("lever") or "").strip()
        score = _score(action.get("current_score") if action.get("current_score") is not None else action.get("score"))
        if key and score:
            rows.append({"key": key, "score": score})
    return rows


def _finding_signals(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for finding in findings:
        key = " ".join(
            str(finding.get(name) or "")
            for name in ("title", "category", "signal_category", "subtype")
        ).strip()
        score = _normalized_percent(finding.get("score") if finding.get("score") is not None else finding.get("adjusted_risk"))
        if key and score:
            rows.append({"key": key, "score": score})
    return rows


def _target_excerpts(report: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    targets = set(_rewrite_targets(report))
    for finding in _all_findings(report):
        finding_id = str(finding.get("finding_id") or finding.get("id") or "")
        if targets and finding_id not in targets:
            continue
        context = finding.get("rewrite_context") if isinstance(finding.get("rewrite_context"), dict) else {}
        excerpt = str(context.get("paragraph_excerpt") or finding.get("evidence") or "").strip()
        if excerpt:
            rows.append(excerpt)
    profile = report.get("ai_footprint_profile") if isinstance(report.get("ai_footprint_profile"), dict) else {}
    for window in profile.get("top_risky_windows") or []:
        if isinstance(window, dict):
            excerpt = str(window.get("source_text") or window.get("source_excerpt") or "").strip()
            if excerpt:
                rows.append(excerpt)
    return _dedupe_text(rows)[:12]


def _rewrite_targets(report: dict[str, Any]) -> list[str]:
    decision = report.get("rewrite_decision") if isinstance(report.get("rewrite_decision"), dict) else {}
    return [str(target) for target in decision.get("targets") or [] if str(target).strip()]


def _all_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings = report.get("findings")
    if isinstance(findings, list):
        return [row for row in findings if isinstance(row, dict)]
    if not isinstance(findings, dict):
        return []
    rows: list[dict[str, Any]] = []
    for bucket in findings.values():
        if isinstance(bucket, list):
            rows.extend(row for row in bucket if isinstance(row, dict))
    return rows


def _row(signal_group: str, score: float, writer_obligation: str) -> dict[str, Any]:
    return {
        "signal_group": signal_group,
        "score": round(score, 3),
        "writer_obligation": writer_obligation,
        "author_proxy_policy": "Allowed for reviewable grounding/context bridges; do not present unsupported external facts as verified.",
    }


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalized_percent(value: Any) -> float:
    score = _score(value)
    return score * 100 if 0 < score <= 1 else score


def _dedupe_contracts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: float(item.get("score") or 0), reverse=True):
        key = str(row.get("signal_group") or "")
        if not key or key in seen:
            continue
        out.append(row)
        seen.add(key)
    return out


def _dedupe_text(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value).split())
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows
