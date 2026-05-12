"""Human Anchor and formula-portfolio candidate builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import re


@dataclass(frozen=True)
class HumanAnchorCandidateDeps:
    env_flag: Callable[[str, bool], bool]
    float_env: Callable[[str, float], float]
    human_anchor_driver_contract: Callable[..., dict]
    logical_paragraphs: Callable[[str], list[str]]
    is_heading_like_paragraph: Callable[[str], bool]
    split_sentences: Callable[[str], list[str]]
    ordered_concept_origin_terms: Callable[..., list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]
    formula_portfolio_plan: Callable[..., dict]
    turnitin_like_ai_profile: Callable[[dict | None], dict]
    blocker_scores: Callable[[dict | None], dict]
    compress_score_drag_paragraph: Callable[..., str]


def human_anchor_amplifier_candidates(
    source_text: str,
    report_dict: dict | None,
    *,
    limit: int = 3,
    deps: HumanAnchorCandidateDeps,
) -> list[tuple[str, str, dict]]:
    """Add bounded implied-context anchors across low-anchor prose spans.

    This targets the scanner's lived-detail density band directly. The edits do
    not claim a new named event, source, person, date, or statistic; they frame
    existing claims as process, judgement, or practice conditions.
    """
    if not deps.env_flag("DRAFTPROOF_HUMAN_ANCHOR_AMPLIFIER", True):
        return []
    contract = deps.human_anchor_driver_contract(report_dict, text=source_text)
    before = contract.get("before") or {}
    lived_risk = before.get("lived_detail_risk")
    human_anchor = before.get("human_anchor_score")
    if (
        isinstance(lived_risk, (int, float))
        and float(lived_risk) < deps.float_env("DRAFTPROOF_HUMAN_ANCHOR_LIVED_RISK_TRIGGER", 65.0)
        and isinstance(human_anchor, (int, float))
        and float(human_anchor) >= deps.float_env("DRAFTPROOF_HUMAN_ANCHOR_SCORE_TRIGGER", 50.0)
    ):
        return []

    paragraphs = deps.logical_paragraphs(source_text)
    if not paragraphs:
        return []

    anchor_re = re.compile(
        r"\b(?:\d+|during|when|after|before|feedback|testing|case|example|"
        r"in practice|I would|I think|I worry|we observed|"
        r"what (?:I|we) (?:would|need|want)|my judgement)\b",
        re.I,
    )
    assertion_re = re.compile(
        r"\b(?:is|are|was|has|have|can|should|must|need(?:s|ed)?|"
        r"creates?|makes?|means|requires?|shows?|supports?|helps?|allows?)\b",
        re.I,
    )
    reference_heading_seen = False
    flat_rows: list[dict] = []
    flat_index = 0
    for paragraph_index, paragraph in enumerate(paragraphs):
        if re.match(r"^\s*references?\s*$", paragraph, flags=re.I):
            reference_heading_seen = True
        sentences = deps.split_sentences(paragraph)
        for sentence_index, sentence in enumerate(sentences):
            words = sentence.split()
            row = {
                "flat_index": flat_index,
                "paragraph_index": paragraph_index,
                "sentence_index": sentence_index,
                "sentence": sentence,
                "skip": False,
            }
            flat_index += 1
            if (
                reference_heading_seen
                or deps.is_heading_like_paragraph(paragraph)
                or len(words) < 8
                or re.search(r"https?://|www\.", sentence, flags=re.I)
                or anchor_re.search(sentence)
            ):
                row["skip"] = True
                flat_rows.append(row)
                continue
            score = (
                (2.0 if assertion_re.search(sentence) else 0.0)
                + min(len(words) / 32.0, 1.5)
                + (1.5 if paragraph_index not in {0, len(paragraphs) - 1} else 0.5)
            )
            row["score"] = round(score, 3)
            flat_rows.append(row)

    target_pool = sorted(
        [row for row in flat_rows if not row.get("skip") and float(row.get("score") or 0.0) > 0],
        key=lambda row: float(row.get("score") or 0.0),
        reverse=True,
    )
    if not target_pool:
        return []

    density = contract.get("estimated_anchor_density") or {}
    eligible_count = int(density.get("eligible_sentence_count") or 0)
    current_hits = int(density.get("anchor_sentence_count") or 0)
    next_required = int(contract.get("required_anchor_sentences_for_next_band") or 0)
    base_needed = max(1, next_required - current_hits)
    sentence_cap = max(1, min(len(target_pool), int(math.ceil(max(eligible_count, 1) * 0.45))))
    profiles = [
        ("human_anchor_amplifier_next_band", min(sentence_cap, base_needed)),
        (
            "human_anchor_amplifier_density_step",
            min(sentence_cap, max(base_needed + 2, int(math.ceil(max(eligible_count, 1) * 0.20)) - current_hits)),
        ),
        (
            "human_anchor_amplifier_strong_density",
            min(sentence_cap, max(base_needed + 4, int(math.ceil(max(eligible_count, 1) * 0.30)) - current_hits)),
        ),
    ]

    def contextualize(sentence: str) -> tuple[str, str]:
        stripped = sentence.strip()
        if not stripped:
            return "", ""
        terms = deps.ordered_concept_origin_terms(stripped, limit=4)
        if len(terms) < 2:
            return "", ""
        first, second = terms[0], terms[1]
        addition = (
            f" This point should stay tied to {first} and {second}, "
            "rather than treated as a general claim."
        )
        if addition.strip().lower() in stripped.lower():
            return "", ""
        return stripped + addition, "local_concept_limit"

    target_pool = [
        {**row, "contextual_operation": contextualize(str(row.get("sentence") or ""))}
        for row in target_pool
    ]
    target_pool = [row for row in target_pool if row.get("contextual_operation", ("", ""))[0]]
    if not target_pool:
        return []

    def build(limit_count: int, label: str) -> tuple[str, list[dict]]:
        selected = sorted(target_pool[:max(1, limit_count)], key=lambda row: int(row["flat_index"]))
        selected_by_flat = {int(row["flat_index"]): pos for pos, row in enumerate(selected)}
        selected_replacements = {
            int(row["flat_index"]): row.get("contextual_operation", ("", ""))
            for row in selected
        }
        rebuilt_paragraphs = []
        flat = 0
        changes = []
        for paragraph in paragraphs:
            rebuilt_sentences = []
            for sentence in deps.split_sentences(paragraph):
                if flat in selected_by_flat:
                    replacement, operation = selected_replacements.get(flat, ("", ""))
                    if replacement:
                        rebuilt_sentences.append(replacement)
                        changes.append({
                            "flat_sentence_index": flat,
                            "operation": operation,
                            "original": sentence[:180],
                            "replacement": replacement[:180],
                        })
                    else:
                        rebuilt_sentences.append(sentence)
                else:
                    rebuilt_sentences.append(sentence)
                flat += 1
            rebuilt_paragraphs.append(" ".join(rebuilt_sentences))
        return deps.join_logical_paragraphs(rebuilt_paragraphs), changes

    candidates: list[tuple[str, str, dict]] = []
    seen: set[str] = {str(source_text or "").strip()}
    for strategy, target_count in profiles[:max(1, int(limit or 1))]:
        if target_count <= 0:
            continue
        candidate, changes = build(target_count, strategy)
        normalized = candidate.strip()
        if not changes or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append((
            strategy,
            candidate,
            {
                "operation": "human_anchor_amplifier",
                "human_anchor_amplifier": True,
                "scope": "implied_context_only",
                "changed_sentence_frames": len(changes),
                "contract_before": contract,
                "target_lived_detail_band": contract.get("next_lived_detail_band"),
                "changes": changes[:12],
            },
        ))
    return candidates[:max(1, int(limit or 1))]


def formula_portfolio_candidates(
    source_text: str,
    report_dict: dict | None,
    *,
    topk_route_candidates: list[tuple[str, str, dict]] | None = None,
    blocker_operation_candidates: list[tuple[str, str, dict]] | None = None,
    generic_assertion_candidates: list[tuple[str, str, dict]] | None = None,
    pruning_candidates: list[tuple[str, str, dict]] | None = None,
    limit: int = 6,
    deps: HumanAnchorCandidateDeps,
) -> list[tuple[str, str, dict]]:
    """Build portfolio candidates that move positive drivers and Human Anchor together.

    The existing candidate families are useful, but isolated. This controller
    composes them according to the Turnitin-like formula plan so the search can
    test combined gap-closure moves instead of hoping one signal moves enough.
    """
    if not deps.env_flag("DRAFTPROOF_FORMULA_PORTFOLIO_GENERATOR", True):
        return []
    limit = max(1, int(limit or 1))
    plan = deps.formula_portfolio_plan(report_dict, report_dict)
    selected_drivers = {
        str(row.get("driver"))
        for row in (plan.get("selected_driver_portfolio") or [])
        if isinstance(row, dict) and row.get("driver")
    }
    priority_drivers = {
        str(row.get("driver"))
        for row in (plan.get("driver_priorities") or [])[:5]
        if isinstance(row, dict) and row.get("driver")
    }
    drivers = selected_drivers | priority_drivers
    if not drivers:
        return []

    source_norm = str(source_text or "").strip()
    seen: set[str] = {source_norm}
    candidates: list[tuple[str, str, dict]] = []

    def add(strategy: str, candidate: str, meta: dict | None, targeted: list[str]) -> None:
        normalized = str(candidate or "").strip()
        if not normalized or normalized in seen or normalized == source_norm:
            return
        if len(candidates) >= limit:
            return
        seen.add(normalized)
        candidates.append((
            strategy,
            candidate,
            {
                **(meta or {}),
                "formula_portfolio_candidate": True,
                "targeted_drivers": targeted,
                "formula_portfolio_plan": {
                    "score_before": plan.get("score_before"),
                    "target_score": plan.get("target_score"),
                    "required_gap": plan.get("required_gap"),
                    "positive_ai_burden": plan.get("positive_ai_burden"),
                    "human_anchor_suppression": plan.get("human_anchor_suppression"),
                    "selected_driver_portfolio": plan.get("selected_driver_portfolio"),
                },
            },
        ))

    def anchor_on_base(
        base_strategy: str,
        base_text: str,
        base_meta: dict | None,
        *,
        targeted: list[str],
        label: str,
    ) -> None:
        if len(candidates) >= limit:
            return
        anchor_variants = human_anchor_amplifier_candidates(base_text, report_dict, limit=1, deps=deps)
        for anchor_strategy, anchor_text, anchor_meta in anchor_variants[:1]:
            add(
                f"formula_portfolio_{label}_{base_strategy}_{anchor_strategy.replace('human_anchor_amplifier_', '')}",
                anchor_text,
                {
                    **(anchor_meta or {}),
                    "base_strategy": base_strategy,
                    "base_meta": base_meta or {},
                    "portfolio_operation": f"{label}_plus_human_anchor",
                },
                targeted,
            )

    topk_pool = list(topk_route_candidates or [])
    blocker_pool = list(blocker_operation_candidates or [])
    generic_pool = list(generic_assertion_candidates or [])
    pruning_pool = list(pruning_candidates or [])

    if "human_anchor_suppression" in drivers:
        for strategy, candidate, meta in human_anchor_amplifier_candidates(source_text, report_dict, limit=2, deps=deps):
            add(
                f"formula_portfolio_{strategy}",
                candidate,
                {
                    **(meta or {}),
                    "portfolio_operation": "human_anchor_suppression_gain",
                },
                ["human_anchor_suppression"],
            )

    if {"ai_likelihood", "topk_calibrated_risk", "human_anchor_suppression"} & drivers:
        for base_strategy, base_text, base_meta in topk_pool[:2]:
            anchor_on_base(
                base_strategy,
                base_text,
                base_meta,
                targeted=["ai_likelihood", "topk_calibrated_risk", "human_anchor_suppression"],
                label="route_anchor",
            )

    if {"semantic_uniformity", "patchwork_expansion", "ai_likelihood", "human_anchor_suppression"} & drivers:
        structural_pool = (blocker_pool[:2] + generic_pool[:2] + pruning_pool[:2])
        for base_strategy, base_text, base_meta in structural_pool:
            anchor_on_base(
                base_strategy,
                base_text,
                base_meta,
                targeted=["semantic_uniformity", "patchwork_expansion", "ai_likelihood", "human_anchor_suppression"],
                label="structure_anchor",
            )
            if len(candidates) >= limit:
                break

    if {"patchwork_expansion", "semantic_uniformity"} & drivers:
        for base_strategy, base_text, base_meta in pruning_pool[:2]:
            add(
                f"formula_portfolio_low_value_remove_{base_strategy}",
                base_text,
                {
                    **(base_meta or {}),
                    "portfolio_operation": "low_value_remove",
                },
                ["patchwork_expansion", "semantic_uniformity"],
            )

    return candidates[:limit]


def human_anchor_suppression_frontier(
    source_text: str,
    report_dict: dict | None,
    block_map: dict | None = None,
    *,
    deps: HumanAnchorCandidateDeps,
) -> dict:
    """Expose the live Human Anchor lever for formula convergence."""
    profile = deps.turnitin_like_ai_profile(report_dict)
    blockers = deps.blocker_scores(report_dict)
    contract = deps.human_anchor_driver_contract(report_dict, text=source_text)
    before = contract.get("before") if isinstance(contract.get("before"), dict) else {}
    suppression = float(profile.get("human_anchor_suppression") or 0.0)
    headroom = max(0.0, 45.0 - suppression)
    lived_detail_risk = float(before.get("lived_detail_risk", blockers.get("lived_detail_risk", 0.0)) or 0.0)
    domain_grounding = float(before.get("domain_grounding_strength", 0.0) or 0.0)
    rows = []
    for row in (block_map or {}).get("blocks") or []:
        if not isinstance(row, dict):
            continue
        if row.get("protected") or row.get("reference_section") or row.get("heading_like"):
            continue
        deficit = float(row.get("human_anchor_deficit") or 0.0)
        potential = float(row.get("suppression_gain_potential") or 0.0)
        if potential <= 0.0 and deficit <= 0.0:
            continue
        rows.append({
            "block_index": row.get("block_index"),
            "recommended_portfolio_action": row.get("recommended_portfolio_action"),
            "human_anchor_deficit": round(deficit, 3),
            "suppression_gain_potential": round(potential, 3),
            "weighted_drag": row.get("weighted_drag"),
            "remove_value_loss_risk": row.get("remove_value_loss_risk"),
            "preview": row.get("preview"),
        })
    rows.sort(
        key=lambda item: (
            float(item.get("suppression_gain_potential") or 0.0),
            float(item.get("human_anchor_deficit") or 0.0),
            float(item.get("weighted_drag") or 0.0),
        ),
        reverse=True,
    )
    return {
        "version": "human_anchor_suppression_frontier_v1",
        "human_anchor_suppression": round(suppression, 3),
        "suppression_headroom": round(headroom, 3),
        "lived_detail_risk": round(lived_detail_risk, 3),
        "domain_grounding_strength": round(domain_grounding, 3),
        "required_suppression_gain_to_target": round(
            min(headroom, max(0.0, float(profile.get("target_gap") or 0.0) + 3.0)),
            3,
        ),
        "driver_contract": contract,
        "candidate_blocks": rows[:8],
        "blocker": (
            "no_suppression_headroom"
            if headroom <= 0.0
            else "insufficient_lived_detail_density"
            if lived_detail_risk >= 55.0
            else "suppression_available_but_lived_detail_not_dominant"
        ),
    }


def anchor_sentence_for_paragraph(paragraph: str, variant: str = "process", *, deps: HumanAnchorCandidateDeps) -> str:
    terms = deps.ordered_concept_origin_terms(paragraph, limit=4)
    if len(terms) >= 2:
        first, second = terms[0], terms[1]
        if variant == "limitation":
            return f"This point should stay limited to {first} and {second}, not stretched into a wider claim."
        return f"The useful check is how {first} relates to {second} in the paragraph itself."
    if variant == "limitation":
        return "This point should be read as a limited judgement, not as a claim that every case will work the same way."
    return "The practical issue is how this would be checked in the actual work, not only how clear the statement sounds."


def append_anchor_sentence(paragraph: str, *, variant: str = "process", deps: HumanAnchorCandidateDeps) -> str:
    sentence = anchor_sentence_for_paragraph(paragraph, variant=variant, deps=deps)
    stripped = str(paragraph or "").strip()
    if not stripped:
        return sentence
    if sentence.lower() in stripped.lower():
        return stripped
    return f"{stripped} {sentence}"


def human_anchor_suppression_frontier_candidates(
    source_text: str,
    report_dict: dict | None,
    block_map: dict | None,
    *,
    limit: int = 4,
    deps: HumanAnchorCandidateDeps,
) -> list[tuple[str, str, dict]]:
    """Create formula candidates that move Human Anchor and AI burden together."""
    if not deps.env_flag("DRAFTPROOF_HUMAN_ANCHOR_SUPPRESSION_FRONTIER", True):
        return []
    frontier = human_anchor_suppression_frontier(source_text, report_dict, block_map, deps=deps)
    if float(frontier.get("suppression_headroom") or 0.0) <= 0.0:
        return []
    if (
        float(frontier.get("lived_detail_risk") or 0.0) < 55.0
        and float(frontier.get("human_anchor_suppression") or 0.0) >= 25.0
    ):
        return []
    paragraphs = deps.logical_paragraphs(source_text)
    if len(paragraphs) < 2:
        return []
    block_rows = [
        row for row in (block_map or {}).get("blocks") or []
        if isinstance(row, dict)
    ]
    by_index = {
        int(row.get("block_index")): row
        for row in block_rows
        if isinstance(row.get("block_index"), int)
    }
    targets = [
        row for row in frontier.get("candidate_blocks") or []
        if isinstance(row, dict)
        and isinstance(row.get("block_index"), int)
        and 0 <= int(row.get("block_index")) < len(paragraphs)
    ]
    if not targets:
        return []
    limit = max(1, int(limit or 1))
    source_norm = str(source_text or "").strip()
    candidates: list[tuple[str, str, dict]] = []
    seen = {source_norm}

    def add(strategy: str, next_paragraphs: list[str], meta: dict) -> None:
        if len(candidates) >= limit:
            return
        candidate = deps.join_logical_paragraphs(next_paragraphs)
        normalized = candidate.strip()
        if not normalized or normalized in seen or normalized == source_norm:
            return
        seen.add(normalized)
        candidates.append((
            strategy,
            candidate,
            {
                **meta,
                "human_anchor_suppression_frontier": True,
                "frontier": frontier,
            },
        ))

    for variant in ("process", "limitation"):
        changed: list[int] = []
        next_paragraphs = list(paragraphs)
        for target in targets[:3]:
            index = int(target["block_index"])
            block = by_index.get(index, {})
            if block.get("protected") or block.get("unique_core_claim") and variant == "process":
                continue
            replacement = append_anchor_sentence(next_paragraphs[index], variant=variant, deps=deps)
            if replacement != next_paragraphs[index]:
                next_paragraphs[index] = replacement
                changed.append(index)
            if len(changed) >= 2:
                break
        if changed:
            add(
                f"human_anchor_{variant}_patch",
                next_paragraphs,
                {
                    "operation": f"anchor_{variant}_reasoning_patch",
                    "portfolio_operation": "human_anchor_suppression_gain",
                    "paragraph_indexes": changed,
                    "targeted_drivers": ["human_anchor_suppression", "lived_detail_risk"],
                },
            )

    for target in targets[:3]:
        index = int(target["block_index"])
        block = by_index.get(index, {})
        if block.get("protected"):
            continue
        compressed = deps.compress_score_drag_paragraph(paragraphs[index], max_remove=2)
        replacement = append_anchor_sentence(compressed, variant="process", deps=deps)
        if replacement.strip() and replacement.strip() != paragraphs[index].strip():
            next_paragraphs = list(paragraphs)
            next_paragraphs[index] = replacement
            add(
                f"anchor_plus_texture_hybrid_p{index + 1}",
                next_paragraphs,
                {
                    "operation": "anchor_plus_texture_hybrid",
                    "portfolio_operation": "human_anchor_plus_texture_rebuild",
                    "paragraph_index": index,
                    "targeted_drivers": [
                        "human_anchor_suppression",
                        "ai_likelihood",
                        "semantic_uniformity",
                        "rewrite_smoothness",
                    ],
                },
            )

    removable = [
        row for row in block_rows
        if row.get("recommended_portfolio_action") == "remove_candidate"
        and row.get("remove_value_loss_risk") == "low"
        and not row.get("protected")
        and not row.get("unique_core_claim")
        and isinstance(row.get("block_index"), int)
    ]
    if removable:
        remove_index = int(removable[0]["block_index"])
        anchor_target = next(
            (
                int(row["block_index"])
                for row in targets
                if int(row["block_index"]) != remove_index
            ),
            0 if remove_index != 0 else min(1, len(paragraphs) - 1),
        )
        next_paragraphs = [
            paragraph for idx, paragraph in enumerate(paragraphs)
            if idx != remove_index
        ]
        adjusted_anchor_index = anchor_target - (1 if anchor_target > remove_index else 0)
        if 0 <= adjusted_anchor_index < len(next_paragraphs):
            next_paragraphs[adjusted_anchor_index] = append_anchor_sentence(
                next_paragraphs[adjusted_anchor_index],
                variant="limitation",
                deps=deps,
            )
            add(
                f"low_value_remove_anchor_p{remove_index + 1}",
                next_paragraphs,
                {
                    "operation": "low_value_block_remove_plus_anchor",
                    "portfolio_operation": "low_value_remove_plus_human_anchor",
                    "removed_paragraph_index": remove_index,
                    "anchor_paragraph_index": anchor_target,
                    "targeted_drivers": [
                        "human_anchor_suppression",
                        "patchwork_expansion",
                        "semantic_uniformity",
                    ],
                },
            )

    return candidates[:limit]
