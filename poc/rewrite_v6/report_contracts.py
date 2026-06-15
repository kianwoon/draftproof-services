from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from .plan import Plan

# Per-paragraph scanner diagnosis (the report explainer's main_issue / how-to-fix / rewrite_hint),
# bound by the production entry so the planner can plan around the concrete fix and direct the
# writer. Keyed by paragraph_id. Empty unless bound.
_PARAGRAPH_DIAGNOSES: ContextVar[dict[str, dict[str, Any]]] = ContextVar(
    "v6_paragraph_diagnoses", default={}
)


def extract_paragraph_diagnoses(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Pull the per-paragraph scanner diagnosis keyed by paragraph_id, for the writer to act on.

    Carries the explainer's prose diagnosis (main_issue, why_flagged, recommendation, rewrite_hint)
    AND ``predictable_phrases`` -- the exact token spans the detector scored most statistically
    predictable (the dominant topk_pattern/predictability signal). Relaying those exact phrases lets
    the writer change the specific flagged wording instead of guessing what reads as generic.
    """
    if not isinstance(report, dict):
        return {}
    phrases_by_paragraph = _paragraph_predictable_phrases(report)
    explanations = report.get("paragraph_explanations")
    rows = explanations.get("paragraphs") if isinstance(explanations, dict) else None
    rows = rows if isinstance(rows, list) else []
    diagnoses: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        paragraph_id = str(row.get("paragraph_id") or "").strip()
        if not paragraph_id:
            continue
        diagnosis = {
            "main_issue": row.get("main_issue"),
            "why_flagged": row.get("why_flagged"),
            "recommendation": row.get("recommendation"),
            "rewrite_hint": row.get("rewrite_hint"),
            "predictable_phrases": phrases_by_paragraph.get(paragraph_id, []),
        }
        if any(diagnosis.values()):
            diagnoses[paragraph_id] = diagnosis
    # A paragraph the explainer skipped can still have flagged predictable phrases -- keep them so
    # the writer is told what to change even without a prose diagnosis.
    for paragraph_id, phrases in phrases_by_paragraph.items():
        if phrases and paragraph_id not in diagnoses:
            diagnoses[paragraph_id] = {
                "main_issue": None,
                "why_flagged": None,
                "recommendation": None,
                "rewrite_hint": None,
                "predictable_phrases": phrases,
            }

    # Critical Thinking Control per-paragraph tag (deterministic, additive): the specific
    # thinking gap the scan flagged for this paragraph + the action to take. Carried through so
    # the writer can target it. A paragraph with ONLY a CT tag still gets a diagnosis entry.
    for paragraph_id, tag in _paragraph_critical_thinking(report).items():
        entry = diagnoses.get(paragraph_id)
        if entry is None:
            entry = {
                "main_issue": None,
                "why_flagged": None,
                "recommendation": None,
                "rewrite_hint": None,
                "predictable_phrases": phrases_by_paragraph.get(paragraph_id, []),
            }
            diagnoses[paragraph_id] = entry
        entry["critical_thinking_action"] = tag.get("action")
        entry["critical_thinking_dimension"] = tag.get("dimension")
    return diagnoses


def _paragraph_critical_thinking(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-paragraph Critical Thinking tags from ai_risk_badge.critical_thinking_control.paragraphs,
    keyed by paragraph_id. Each value carries the dimension code + its coaching ``action``."""
    badge = report.get("ai_risk_badge") if isinstance(report, dict) else None
    ctc = badge.get("critical_thinking_control") if isinstance(badge, dict) else None
    rows = ctc.get("paragraphs") if isinstance(ctc, dict) else None
    out: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            paragraph_id = str(row.get("paragraph_id") or "").strip()
            action = str(row.get("action") or "").strip()
            if paragraph_id and action:
                out[paragraph_id] = {"action": action, "dimension": row.get("dimension")}
    return out


def _paragraph_predictable_phrases(
    report: dict[str, Any], *, per_paragraph_limit: int = 10
) -> dict[str, list[str]]:
    """Group the detector's predictable_token_spans by paragraph_id (deduped, in order).

    Source of truth is the raw per-sentence detector output on ``highlight_segments`` (always
    populated), not the LLM explainer prose -- so the writer gets the exact flagged wording.
    """
    segments = report.get("highlight_segments")
    if not isinstance(segments, list):
        return {}
    by_paragraph: dict[str, list[str]] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        paragraph_id = str(segment.get("paragraph_id") or "").strip()
        if not paragraph_id:
            continue
        predictability = segment.get("predictability")
        spans = predictability.get("predictable_token_spans") if isinstance(predictability, dict) else None
        if not isinstance(spans, list):
            continue
        bucket = by_paragraph.setdefault(paragraph_id, [])
        for span in spans:
            phrase = " ".join(str(span or "").split()).strip()
            # drop trivial / punctuation-only spans -- they aren't actionable for the writer
            if len(phrase) < 3 or not any(ch.isalpha() for ch in phrase):
                continue
            if phrase not in bucket:
                bucket.append(phrase)
    return {pid: phrases[:per_paragraph_limit] for pid, phrases in by_paragraph.items() if phrases}


@contextlib.contextmanager
def paragraph_diagnoses_context(diagnoses: dict[str, dict[str, Any]] | None):
    token = _PARAGRAPH_DIAGNOSES.set(diagnoses or {})
    try:
        yield
    finally:
        _PARAGRAPH_DIAGNOSES.reset(token)


def paragraph_diagnosis(paragraph_id: str) -> dict[str, Any] | None:
    return _PARAGRAPH_DIAGNOSES.get().get(str(paragraph_id or ""))


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
