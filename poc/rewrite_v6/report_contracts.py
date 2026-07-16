from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar
from dataclasses import replace
from typing import Any


def _flagged_sentences_enabled() -> bool:
    """Kill switch for P1 (per-sentence grounding/reasoning targeting). Default ON; set
    DRAFTPROOF_V6_FLAGGED_SENTENCES=0 to fall back to the pre-P1 prompt (byte-identical: no
    ``flagged_sentences`` is attached, so the writer prompt is unchanged)."""
    return os.environ.get("DRAFTPROOF_V6_FLAGGED_SENTENCES", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }

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

    # Critical Thinking reflective questions matched to their anchored paragraph -- a sharp,
    # specific target the writer should visibly DEMONSTRATE addressing (showcase). Additive;
    # only the few anchored paragraphs get questions.
    for paragraph_id, questions in _paragraph_questions(report).items():
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
        entry["critical_thinking_questions"] = questions

    # Per-sentence grounding/reasoning targets (P1): the enhanced scan knows WHICH sentences are weak,
    # not just which paragraphs. Attach them so the writer fixes the exact flagged sentences. A
    # paragraph with ONLY flagged sentences (no prose diagnosis) still gets an entry so it is rewritten.
    for paragraph_id, flagged in (
        _paragraph_flagged_sentences(report).items() if _flagged_sentences_enabled() else ()
    ):
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
        entry["flagged_sentences"] = flagged

    # Per-claim source-entailment targets (P2): claims the scan could NOT verify against their cited
    # source. Attach so the writer qualifies/softens exactly those claims (never fabricates a fix).
    for paragraph_id, unsupported in (
        _paragraph_unsupported_claims(report).items() if _unsupported_claims_enabled() else ()
    ):
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
        entry["unsupported_claims"] = unsupported
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


def _paragraph_questions(
    report: dict[str, Any], *, per_paragraph_limit: int = 2
) -> dict[str, list[str]]:
    """Match each Critical Thinking reflective question to the paragraph its ``anchor_quote``
    came from, keyed by paragraph_id. The questions are anchored to verbatim spans of
    ``scan_intelligence.document.paragraphs`` (their source), so a normalised substring match
    is reliable. A question whose quote matches no paragraph is skipped (never attach an
    unanchored question -- precision-first)."""
    badge = report.get("ai_risk_badge") if isinstance(report, dict) else None
    ctc = badge.get("critical_thinking_control") if isinstance(badge, dict) else None
    questions = ctc.get("questions") if isinstance(ctc, dict) else None
    if not isinstance(questions, list) or not questions:
        return {}

    def _norm(value: Any) -> str:
        return " ".join(str(value or "").split()).casefold()

    # paragraph_id -> normalised text. Prefer document.paragraphs (full paragraph text, the
    # questions' source); fall back to grouping highlight_segments text per paragraph.
    para_text: dict[str, str] = {}
    intel = report.get("scan_intelligence")
    document = intel.get("document") if isinstance(intel, dict) else None
    rows = document.get("paragraphs") if isinstance(document, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                pid = str(row.get("paragraph_id") or "").strip()
                txt = _norm(row.get("text"))
                if pid and txt:
                    para_text[pid] = txt
    if not para_text:
        for segment in report.get("highlight_segments") or []:
            if isinstance(segment, dict):
                pid = str(segment.get("paragraph_id") or "").strip()
                if pid:
                    para_text[pid] = (para_text.get(pid, "") + " " + _norm(segment.get("text"))).strip()

    out: dict[str, list[str]] = {}
    for q in questions:
        if not isinstance(q, dict):
            continue
        question = str(q.get("question") or "").strip()
        quote = _norm(q.get("anchor_quote"))
        if not question or not quote:
            continue
        paragraph_id = next((pid for pid, text in para_text.items() if quote in text), None)
        if not paragraph_id:
            continue
        bucket = out.setdefault(paragraph_id, [])
        if question not in bucket and len(bucket) < per_paragraph_limit:
            bucket.append(question)
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


# ── Per-sentence grounding/reasoning targets (P1) ──────────────────────────────
# The enhanced scan tags individual sentences as weak-grounding or reasoning-jump. The display
# composer (poc/report/sentence_issue_tags.py) is the authority; we read the SAME two trustworthy
# finding titles here so the writer targets the exact sentences the user sees underlined, instead of
# only a paragraph-wide category. KEEP IN SYNC with sentence_issue_tags._GROUNDING_TITLE /
# _REASONING_TITLE (module-private there; duplicated deliberately rather than importing an underscore
# name across packages). AI (red) tags are intentionally NOT relayed — predictable_phrases already
# carries the predictability signal at higher fidelity.
_FLAGGED_SENTENCE_ISSUE_BY_TITLE = {
    "low_specificity": "grounding",
    "semantic_drift": "reasoning",
}
_FLAGGED_SENTENCES_PER_PARAGRAPH = 3


def _paragraph_flagged_sentences(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group per-sentence grounding/reasoning findings by paragraph_id, for the writer to target.

    Joins each trustworthy finding's ``sentence_id`` to its paragraph + verbatim sentence text via
    ``highlight_segments`` (which carry sentence_id + paragraph_id + text together). A finding whose
    sentence_id resolves to no segment, or a document-level tag with no sentence_id, is dropped — an
    unanchored tag is never paragraph-guessed (precision-first). Capped per paragraph to bound prompt
    growth. Returns {paragraph_id: [{text, issue, fix}]}."""
    sid_index: dict[str, tuple[str, str]] = {}
    for segment in report.get("highlight_segments") or []:
        if not isinstance(segment, dict):
            continue
        sid = str(segment.get("sentence_id") or "").strip()
        paragraph_id = str(segment.get("paragraph_id") or "").strip()
        text = " ".join(str(segment.get("text") or "").split())
        if sid and paragraph_id and text and sid not in sid_index:
            sid_index[sid] = (paragraph_id, text)

    by_paragraph: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for finding in _all_findings(report):
        issue = _FLAGGED_SENTENCE_ISSUE_BY_TITLE.get(str(finding.get("title") or "").strip())
        if not issue:
            continue
        sid = str(finding.get("sentence_id") or "").strip()
        if not sid or sid not in sid_index:
            continue  # document-level or unanchored tag → never guessed onto a paragraph
        paragraph_id, text = sid_index[sid]
        if (paragraph_id, sid) in seen:
            continue
        seen.add((paragraph_id, sid))
        bucket = by_paragraph.setdefault(paragraph_id, [])
        if len(bucket) >= _FLAGGED_SENTENCES_PER_PARAGRAPH:
            continue
        row: dict[str, Any] = {"text": text, "issue": issue}
        fix = " ".join(str(finding.get("recommendation") or "").split())
        if fix:
            row["fix"] = fix
        bucket.append(row)
    return {pid: rows for pid, rows in by_paragraph.items() if rows}


# ── Per-claim source-entailment targets (P2) ───────────────────────────────────
# The enhanced scan's claim graph checks specific claims against their cited source and records a
# verdict (verified / contradicted / paywalled / unresolved) + entailment score. We relay the
# NON-verified ones to the writer so it can qualify/attribute/soften exactly those claims — the direct
# attack on citation_grounding_risk. Gated: the claim graph itself only exists under DRAFTPROOF_CLAIM_GRAPH,
# so this is empty (byte-identical) in a normal report. We join UPSTREAM from the raw graph
# (authorship_evidence.claim_graph), NOT the display panel — the panel is a pinned, capped, truncated
# render contract and drops the paragraph_id we need.
_UNSUPPORTED_CLAIMS_PER_PARAGRAPH = 2
_UNSUPPORTED_VERDICT_WHY = {
    "contradicted": "a cited source contradicts this claim — soften or correct it, or attribute the dispute",
    "paywalled": "the cited source could not be checked (paywalled) — do not assert it as verified",
    "unresolved": "this claim could not be verified against a source — qualify it or attribute it",
}


def _unsupported_claims_enabled() -> bool:
    """Kill switch for P2 (per-claim source-entailment targeting). Default ON; set
    DRAFTPROOF_V6_UNSUPPORTED_CLAIMS=0 to stop relaying claim-graph verdicts to the writer."""
    return os.environ.get("DRAFTPROOF_V6_UNSUPPORTED_CLAIMS", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _raw_claim_graph(report: dict[str, Any]) -> dict[str, Any] | None:
    authorship = report.get("authorship_evidence")
    graph = authorship.get("claim_graph") if isinstance(authorship, dict) else None
    return graph if isinstance(graph, dict) else None


def _paragraph_unsupported_claims(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group NON-verified claim-source checks by the paragraph the claim came from.

    Reuses the display composer's ``_row_status`` so rewrite's notion of a verdict is identical to the
    panel the user sees. Keeps only ``contradicted`` / ``paywalled`` / ``unresolved`` (a ``verified``
    claim needs no action). Joins to the paragraph via the raw claim's ``source.paragraph_id`` and
    passes the UNtruncated claim text so the writer edits the right sentence. Returns
    {paragraph_id: [{claim, verdict, why, entailment_score}]}. Empty when the graph is absent."""
    graph = _raw_claim_graph(report)
    if not graph:
        return {}
    try:
        from report.claim_graph_panel import _row_status  # verdict-mapping authority (keep in sync)
    except ImportError:  # pragma: no cover
        from poc.report.claim_graph_panel import _row_status

    claims_by_id = {c.get("id"): c for c in (graph.get("claims") or []) if isinstance(c, dict) and c.get("id")}
    by_paragraph: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for ev in (graph.get("evidence") or []):
        if not isinstance(ev, dict):
            continue
        resolution = ev.get("detail", {}).get("resolution") if isinstance(ev.get("detail"), dict) else None
        if not isinstance(resolution, dict):
            continue
        entailment = resolution.get("entailment") if isinstance(resolution.get("entailment"), dict) else {}
        for cid in sorted(c for c in (ev.get("claim_ids") or []) if isinstance(c, str)):
            claim = claims_by_id.get(cid)
            if not isinstance(claim, dict):
                continue
            source = claim.get("source") if isinstance(claim.get("source"), dict) else {}
            paragraph_id = str(source.get("paragraph_id") or "").strip()
            claim_text = " ".join(str(claim.get("text") or "").split())
            if not paragraph_id or not claim_text:
                continue
            verdict = entailment.get(cid) if isinstance(entailment.get(cid), dict) else None
            status = _row_status(resolution, verdict)
            if status not in _UNSUPPORTED_VERDICT_WHY:  # skip "verified" (and any unknown)
                continue
            if (paragraph_id, cid) in seen:
                continue
            seen.add((paragraph_id, cid))
            bucket = by_paragraph.setdefault(paragraph_id, [])
            if len(bucket) >= _UNSUPPORTED_CLAIMS_PER_PARAGRAPH:
                continue
            row: dict[str, Any] = {"claim": claim_text, "verdict": status, "why": _UNSUPPORTED_VERDICT_WHY[status]}
            if verdict is not None:
                try:
                    row["entailment_score"] = round(float(verdict.get("entailment_score")), 4)
                except (TypeError, ValueError):
                    pass
            bucket.append(row)
    return {pid: rows for pid, rows in by_paragraph.items() if rows}


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
