from __future__ import annotations

import re
from typing import Any


_FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|we|our|ours)\b", re.I)
_FORMAL_MARKERS = re.compile(r"\b(?:system|policy|institution|organisation|organization|report|study|framework|standard)\b", re.I)
_RISK_MARKERS = re.compile(r"\b(?:risk|concern|issue|problem|barrier|challenge|harm|impact|consequence|inequality)\b", re.I)
_CONDITION_MARKERS = re.compile(r"\b(?:when|if|after|before|unless|because|where|while)\b", re.I)
_SOURCE_MARKERS = re.compile(r"\b(?:source|evidence|report|article|data|citation|reference|text|claim)\b", re.I)
_ACTION_MARKERS = re.compile(r"\b(?:decide|choose|review|check|compare|apply|use|teach|work|practice|approve|revise)\b", re.I)
_SCALE_MARKERS = re.compile(r"\b(?:\d[\d,.]*%?|percent|percentage|rate|ratio|scale|many|few|several|majority|minority|some|most)\b", re.I)


_ROUTES: dict[str, dict[str, str]] = {
    "observed_process": {
        "purpose": "ground a broad claim in what the author saw, tried, checked, taught, reviewed, or had to manage",
        "use_when": "reflective prose, working-practice prose, or a paragraph that needs author-owned concrete texture",
        "avoid": "do not turn the process into a detached scenario with no author vantage",
    },
    "local_constraint": {
        "purpose": "show the practical limitation or pressure that shapes the claim",
        "use_when": "the paragraph names a gap, barrier, old habit, resource limit, rule, or implementation pressure",
        "avoid": "do not invent a named institution, policy, or verified external event",
    },
    "decision_moment": {
        "purpose": "anchor the claim in a choice the author, actor, or team must make",
        "use_when": "the paragraph argues that something should change, be evaluated, balanced, or redesigned",
        "avoid": "do not add a new thesis; keep the decision tied to the submitted claim",
    },
    "source_use": {
        "purpose": "ground the claim in how evidence, sources, examples, or submitted material are used",
        "use_when": "the paragraph mentions information, evidence, claims, source material, reports, or comparison",
        "avoid": "do not invent citations, source names, study titles, or research findings",
    },
    "actor_interaction": {
        "purpose": "make the claim concrete through the relationship between local actors and objects",
        "use_when": "the paragraph contains people, roles, users, teams, institutions, tools, or workflows",
        "avoid": "do not invent named people; use roles or source-provided actors",
    },
    "condition_trigger": {
        "purpose": "turn a generic assertion into a when/if/after condition that explains when it becomes true",
        "use_when": "the paragraph makes a broad causal or evaluative claim",
        "avoid": "do not overstate possibility as certainty",
    },
    "risk_consequence": {
        "purpose": "connect the author-owned observation to the consequence the claim is warning about",
        "use_when": "the paragraph discusses concern, fairness, risk, misunderstanding, pressure, or failure mode",
        "avoid": "do not add dramatic stakes not supported by the paragraph",
    },
    "scale_detail": {
        "purpose": "preserve or lightly support submitted scale inside an author-proxy beat",
        "use_when": "the source already has numeric or scale language, or the diagnosis asks for scale",
        "avoid": "do not make scale the paragraph spine and do not invent exact verified statistics",
    },
}


def select_author_proxy_routes(
    paragraph_text: str,
    diagnosis: dict[str, Any] | None,
    finding_tags: list[str] | tuple[str, ...] | None,
    authorship_targets: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return content-agnostic author-proxy route cards for diversified direct rewrite.

    The routes are not a selector score and not topic-specific. They only describe different shapes
    for the same author-proxy move: author-owned vantage plus concrete particular plus claim purpose.
    """
    text = str(paragraph_text or "")
    diag_text = _diagnosis_text(diagnosis)
    tags = {str(tag).casefold() for tag in (finding_tags or [])}
    selected: list[str] = []

    # observed_process is the FIRST-PERSON mode. Reserve it for a genuine first-person / first-hand
    # signal in the source (or explicit authorship grounding); never inject it on generic action verbs.
    # For every other claim the non-first-person modes below carry the grounding, so the writer is not
    # nudged to invent experience the author never had.
    first_hand = bool(_FIRST_PERSON.search(text) or _authorship_grounding_present(authorship_targets))
    if first_hand:
        selected.append("observed_process")
    if _ACTION_MARKERS.search(text) or tags & {"author_anchor_gap", "context_anchor_gap", "low_specificity"}:
        selected.extend(["decision_moment", "actor_interaction"])
    if _SOURCE_MARKERS.search(text) or "source_grounding" in tags:
        selected.append("source_use")
    if _RISK_MARKERS.search(text) or "broad_claim" in tags:
        selected.append("risk_consequence")
    if _CONDITION_MARKERS.search(text) or "generic_assertion_risk" in tags:
        selected.append("condition_trigger")
    if _FORMAL_MARKERS.search(text) and not _FIRST_PERSON.search(text):
        selected.extend(["actor_interaction", "local_constraint"])

    selected.extend(["actor_interaction", "condition_trigger", "source_use", "local_constraint"])
    selected = _dedupe(selected)

    non_numeric = [mode for mode in selected if mode != "scale_detail"][:4]
    if len(non_numeric) < 3:
        # first-person (observed_process) only backfills when the source is genuinely first-hand.
        fallbacks = ("actor_interaction", "condition_trigger", "source_use", "local_constraint")
        if first_hand:
            fallbacks = ("observed_process", *fallbacks)
        for fallback in fallbacks:
            if fallback not in non_numeric:
                non_numeric.append(fallback)
            if len(non_numeric) >= 3:
                break

    modes = non_numeric[:4]
    if _SCALE_MARKERS.search(text) or _SCALE_MARKERS.search(diag_text) or "scale" in diag_text.casefold():
        modes.append("scale_detail")
    return [{"mode": mode, **_ROUTES[mode]} for mode in modes]


def _diagnosis_text(diagnosis: dict[str, Any] | None) -> str:
    if not isinstance(diagnosis, dict):
        return ""
    return " ".join(str(value) for value in diagnosis.values() if isinstance(value, (str, int, float)))


def _authorship_grounding_present(authorship_targets: dict[str, Any] | None) -> bool:
    if not isinstance(authorship_targets, dict):
        return False
    targets = authorship_targets.get("grounding_targets")
    return isinstance(targets, list) and any(str(item).strip() for item in targets)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out
