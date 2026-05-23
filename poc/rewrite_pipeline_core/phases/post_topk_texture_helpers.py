"""Post-Top-k authorship/transformation texture helper functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import json
import re
import statistics


@dataclass(frozen=True)
class PostTopkTextureHelperDeps:
    logical_paragraphs: Callable[[str], list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]
    text_word_count: Callable[[str], int]
    clean_paragraph_component_candidate: Callable[..., tuple[str, str]]
    strict_ai_safe_band_status: Callable[[dict | None], dict]
    post_topk_convergence_candidates: Callable[..., list[tuple[str, str, dict]]]
    split_sentences: Callable[[str], list[str]]
    sentence_has_concrete_or_context: Callable[[str], bool]
    generic_assertion_sentence_score: Callable[[str], float]
    paragraph_role: Callable[..., str]
    detect_protected_spans: Callable[[str], Any]
    generic_assertion_protected_sentence_re: Any
    paragraph_citation_re: Any
    generic_assertion_terms_re: Any
    post_topk_template_opening_re: Any
    post_topk_low_value_paragraph_re: Any


def authorship_transformation_texture_patch_prompt(
    candidate_text: str,
    candidate_report: dict | None,
    *,
    deps: PostTopkTextureHelperDeps,
) -> str:
    driver_map = authorship_transformation_texture_driver_map(candidate_text, candidate_report, deps=deps)
    strict_status = deps.strict_ai_safe_band_status(candidate_report)
    patch_targets = []
    paragraphs = deps.logical_paragraphs(candidate_text)
    for row in driver_map.get("ranked_blocks") or []:
        index = row.get("paragraph_index")
        if not isinstance(index, int) or index < 0 or index >= len(paragraphs):
            continue
        if row.get("has_protected_anchor"):
            continue
        patch_targets.append({
            "paragraph_index": index,
            "role": row.get("role"),
            "authorship_driver_score": row.get("authorship_driver_score"),
            "transformation_driver_score": row.get("transformation_driver_score"),
            "generic_sentence_ratio": row.get("generic_sentence_ratio"),
            "paragraph": paragraphs[index],
        })
        if len(patch_targets) >= 5:
            break
    return (
        "DraftProof AUTHORSHIP_TRANSFORMATION_TEXTURE_CONTROLLER.\n"
        "The document already passed calibrated Top-k. Preserve that and attack only authorship/transformation texture.\n"
        "Return only valid JSON. No markdown.\n\n"
        "Hard controller rules:\n"
        "- topk_calibrated_risk must stay below 25\n"
        "- reduce ai_authorship or ai_transformation\n"
        "- do not increase review burden, weighted severity, or critical/high findings\n"
        "- no full-document rewrite; patch selected paragraphs only\n"
        "- no new facts, citations, names, numbers, dates, examples, or author evidence\n"
        "- do not polish the prose into a cleaner essay style\n\n"
        "Allowed operation types:\n"
        "- AUTHORSHIP_SUPPRESSION: break explanatory cadence in one paragraph\n"
        "- TRANSFORMATION_DETEMPLATE: collapse claim-explain-conclude symmetry\n"
        "- HYBRID_TEXTURE_COLLAPSE: combine small authorship/transformation reductions\n"
        "- LOW_VALUE_REMOVE: replace a low-value generic paragraph with an empty string only if meaning is duplicated nearby\n\n"
        "Current strict-safe status:\n"
        f"{json.dumps(strict_status, ensure_ascii=False)[:3000]}\n\n"
        "Texture driver map:\n"
        f"{json.dumps({k: driver_map.get(k) for k in ('authorship_drivers', 'transformation_drivers', 'generic_sentence_ratio')}, ensure_ascii=False)[:3500]}\n\n"
        "Patch targets:\n"
        f"{json.dumps(patch_targets, ensure_ascii=False)[:9000]}\n\n"
        "Return schema:\n"
        "{\n"
        "  \"candidates\": [\n"
        "    {\n"
        "      \"reason\": \"short reason\",\n"
        "      \"patches\": [\n"
        "        {\"operation_type\": \"AUTHORSHIP_SUPPRESSION\", \"target_paragraph_index\": 0, \"expected_driver\": \"ai_authorship\", \"replacement\": \"replacement paragraph\"}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Return at most 2 candidates. Each candidate may patch at most 2 paragraphs."
    )


def extract_post_topk_patch_candidates(response_text: str, *, max_candidates: int = 2) -> list[dict]:
    text = str(response_text or "").strip()
    if not text:
        return []
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return []
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    candidates = []
    for row in rows[:max(1, max_candidates)]:
        if not isinstance(row, dict):
            continue
        patches = row.get("patches")
        if not isinstance(patches, list):
            continue
        cleaned = []
        for patch in patches[:2]:
            if not isinstance(patch, dict):
                continue
            index = patch.get("target_paragraph_index")
            if not isinstance(index, int):
                index = patch.get("paragraph_index")
            replacement = str(patch.get("replacement") or "").strip()
            if not isinstance(index, int) or not replacement:
                continue
            operation_type = str(patch.get("operation_type") or row.get("operation_type") or "").strip()
            expected_driver = str(patch.get("expected_driver") or row.get("expected_driver") or "").strip()
            cleaned.append({
                "paragraph_index": index,
                "target_paragraph_index": index,
                "replacement": replacement,
                "operation_type": operation_type,
                "expected_driver": expected_driver,
            })
        if cleaned:
            candidates.append({"reason": row.get("reason"), "patches": cleaned})
    return candidates


def apply_post_topk_patches(
    text: str,
    patches: list[dict],
    *,
    deps: PostTopkTextureHelperDeps,
) -> tuple[str, list[dict]]:
    paragraphs = deps.logical_paragraphs(text)
    applied = []
    for patch in patches or []:
        index = patch.get("paragraph_index")
        replacement = str(patch.get("replacement") or "").strip()
        if not isinstance(index, int) or index < 0 or index >= len(paragraphs):
            continue
        cleaned, reason = deps.clean_paragraph_component_candidate(replacement, paragraphs[index])
        if reason or not cleaned:
            continue
        if cleaned.strip() == paragraphs[index].strip():
            continue
        paragraphs[index] = cleaned
        applied.append({
            "paragraph_index": index,
            "target_paragraph_index": index,
            "replacement_words": deps.text_word_count(cleaned),
            "operation_type": patch.get("operation_type"),
            "expected_driver": patch.get("expected_driver"),
        })
    if not applied:
        return text, []
    return deps.join_logical_paragraphs(paragraphs), applied


def opening_route_key(sentence: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", str(sentence or "").lower())
    return " ".join(words[:3])


def post_topk_sentence_contextual(sentence: str, *, deps: PostTopkTextureHelperDeps) -> bool:
    sentence = str(sentence or "").strip()
    if not sentence:
        return False
    return bool(
        deps.sentence_has_concrete_or_context(sentence)
        or deps.generic_assertion_protected_sentence_re.search(sentence)
        or deps.paragraph_citation_re.search(sentence)
    )


def post_topk_sentence_driver_score(sentence: str, *, deps: PostTopkTextureHelperDeps) -> float:
    sentence = str(sentence or "").strip()
    if not sentence:
        return 0.0
    score = deps.generic_assertion_sentence_score(sentence)
    if not post_topk_sentence_contextual(sentence, deps=deps):
        score += 5.0
    if deps.post_topk_template_opening_re.search(sentence):
        score += 4.0
    if len(deps.split_sentences(sentence)) == 1 and deps.text_word_count(sentence) >= 24:
        score += 1.5
    return round(score, 3)


def post_topk_driver_map(text: str, raw_json: dict | None, *, deps: PostTopkTextureHelperDeps) -> dict:
    paragraphs = deps.logical_paragraphs(text)
    rows = []
    generic_sentence_count = 0
    total_sentence_count = 0
    protected = deps.detect_protected_spans(text)

    def paragraph_has_protected(index: int) -> bool:
        before = deps.join_logical_paragraphs(paragraphs[:index])
        start = len(before) + (2 if before else 0)
        end = start + len(paragraphs[index])
        return any(span.start_char >= start and span.end_char <= end for span in protected)

    role_sequence = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = deps.split_sentences(paragraph)
        sentence_rows = []
        generic_scores = []
        contextual_count = 0
        for sentence_index, sentence in enumerate(sentences):
            contextual = post_topk_sentence_contextual(sentence, deps=deps)
            driver_score = post_topk_sentence_driver_score(sentence, deps=deps)
            total_sentence_count += 1
            if not contextual:
                generic_sentence_count += 1
            else:
                contextual_count += 1
            generic_scores.append(driver_score)
            sentence_rows.append({
                "sentence_index": sentence_index,
                "text": sentence,
                "word_count": deps.text_word_count(sentence),
                "contextual": contextual,
                "driver_score": driver_score,
                "protected": bool(deps.generic_assertion_protected_sentence_re.search(sentence)),
            })
        drivers = {
            "generic_assertion_hits": len(deps.generic_assertion_terms_re.findall(paragraph)),
            "concrete_anchor_hits": contextual_count,
            "source_gap": not bool(deps.paragraph_citation_re.search(paragraph)),
            "word_count": deps.text_word_count(paragraph),
        }
        role = deps.paragraph_role(paragraph, drivers, is_last=paragraph_index == len(paragraphs) - 1)
        role_sequence.append(role)
        generic_ratio = (
            sum(1 for row in sentence_rows if not row["contextual"]) / max(len(sentence_rows), 1)
        )
        low_value = bool(
            not paragraph_has_protected(paragraph_index)
            and contextual_count == 0
            and (
                generic_ratio >= 0.65
                or role in {"conclusion_template_risk", "generic_claim_heavy"}
                or deps.post_topk_low_value_paragraph_re.search(paragraph)
            )
        )
        rows.append({
            "paragraph_index": paragraph_index,
            "paragraph": paragraph,
            "role": role,
            "word_count": deps.text_word_count(paragraph),
            "sentence_count": len(sentences),
            "generic_sentence_ratio": round(generic_ratio, 3),
            "max_sentence_driver_score": max(generic_scores or [0.0]),
            "paragraph_driver_score": round(sum(generic_scores) + generic_ratio * 10.0, 3),
            "has_protected_anchor": paragraph_has_protected(paragraph_index),
            "low_value_generic_block": low_value,
            "sentences": sentence_rows,
        })
    repeated_role_runs = 0
    previous = None
    for role in role_sequence:
        if role and role == previous:
            repeated_role_runs += 1
        previous = role
    profile = deps.strict_ai_safe_band_status(raw_json).get("profile") if isinstance(raw_json, dict) else {}
    return {
        "kind": "post_topk_driver_map",
        "profile": profile or {},
        "paragraph_count": len(paragraphs),
        "sentence_count": total_sentence_count,
        "generic_sentence_count": generic_sentence_count,
        "generic_sentence_ratio": round(generic_sentence_count / max(total_sentence_count, 1), 3),
        "repeated_paragraph_role_runs": repeated_role_runs,
        "paragraphs": sorted(rows, key=lambda item: float(item.get("paragraph_driver_score") or 0.0), reverse=True),
    }


def authorship_transformation_texture_driver_map(
    text: str,
    raw_json: dict | None,
    *,
    deps: PostTopkTextureHelperDeps,
) -> dict:
    """Map the post-Top-k document to authorship and transformation texture drivers."""
    base = post_topk_driver_map(text, raw_json, deps=deps)
    paragraphs = base.get("paragraphs") or []
    opening_counts: dict[str, int] = {}
    sentence_lengths: list[int] = []
    transition_hits = 0
    transition_re = re.compile(
        r"^\s*(?:also|but|however|therefore|so|then|this|that|in\s+(?:addition|conclusion|summary)|"
        r"furthermore|moreover|additionally|overall)\b",
        re.I,
    )
    for row in paragraphs:
        for sentence_row in row.get("sentences") or []:
            sentence = sentence_row.get("text") or ""
            key = opening_route_key(sentence)
            if key:
                opening_counts[key] = opening_counts.get(key, 0) + 1
            sentence_lengths.append(int(sentence_row.get("word_count") or 0))
            if transition_re.search(sentence):
                transition_hits += 1

    mean_len = statistics.mean(sentence_lengths) if sentence_lengths else 0.0
    stdev_len = statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    length_uniformity = 0.0
    if mean_len > 0:
        length_uniformity = max(0.0, min(100.0, 100.0 - (stdev_len / mean_len * 100.0)))

    repeated_opening_hits = sum(count - 1 for count in opening_counts.values() if count > 1)
    transition_density = transition_hits / max(int(base.get("sentence_count") or 0), 1)
    role_counts: dict[str, int] = {}
    for row in paragraphs:
        role = str(row.get("role") or "")
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1

    ranked_blocks = []
    low_value_blocks = []
    for row in paragraphs:
        sentence_count = max(int(row.get("sentence_count") or 0), 1)
        generic_ratio = float(row.get("generic_sentence_ratio") or 0.0)
        role = str(row.get("role") or "")
        role_repeat = max(0, int(role_counts.get(role, 0)) - 1)
        repeated_openings_in_block = 0
        transition_hits_in_block = 0
        for sentence_row in row.get("sentences") or []:
            sentence = sentence_row.get("text") or ""
            key = opening_route_key(sentence)
            if key and opening_counts.get(key, 0) > 1:
                repeated_openings_in_block += 1
            if transition_re.search(sentence):
                transition_hits_in_block += 1
        authorship_score = (
            float(row.get("max_sentence_driver_score") or 0.0)
            + generic_ratio * 12.0
            + (transition_hits_in_block / sentence_count) * 5.0
            + repeated_openings_in_block * 1.5
            + (length_uniformity / 100.0) * 3.0
        )
        transformation_score = (
            float(row.get("paragraph_driver_score") or 0.0)
            + generic_ratio * 8.0
            + role_repeat * 2.0
            + (6.0 if role in {"generic_claim_heavy", "conclusion_template_risk", "source_summary_heavy"} else 0.0)
        )
        row_out = {
            "paragraph_index": row.get("paragraph_index"),
            "role": role,
            "word_count": row.get("word_count"),
            "sentence_count": row.get("sentence_count"),
            "has_protected_anchor": bool(row.get("has_protected_anchor")),
            "low_value_generic_block": bool(row.get("low_value_generic_block")),
            "generic_sentence_ratio": row.get("generic_sentence_ratio"),
            "authorship_driver_score": round(authorship_score, 3),
            "transformation_driver_score": round(transformation_score, 3),
            "texture_driver_score": round(authorship_score + transformation_score, 3),
            "repeated_opening_hits": repeated_openings_in_block,
            "transition_hits": transition_hits_in_block,
            "top_sentence_drivers": [
                {
                    "sentence_index": s.get("sentence_index"),
                    "driver_score": s.get("driver_score"),
                    "word_count": s.get("word_count"),
                    "contextual": s.get("contextual"),
                    "protected": s.get("protected"),
                }
                for s in sorted(
                    row.get("sentences") or [],
                    key=lambda item: float(item.get("driver_score") or 0.0),
                    reverse=True,
                )[:3]
            ],
        }
        ranked_blocks.append(row_out)
        if row_out["low_value_generic_block"]:
            low_value_blocks.append(row_out)

    ranked_blocks.sort(key=lambda item: float(item.get("texture_driver_score") or 0.0), reverse=True)
    profile = base.get("profile") or {}
    return {
        "kind": "authorship_transformation_texture_driver_map",
        "profile": profile,
        "authorship_drivers": {
            "ai_likelihood": profile.get("ai_likelihood"),
            "rewrite_smoothness": profile.get("rewrite_smoothness"),
            "repeated_opening_hits": repeated_opening_hits,
            "transition_density": round(transition_density, 3),
            "sentence_length_uniformity": round(length_uniformity, 3),
        },
        "transformation_drivers": {
            "ai_transformation": profile.get("ai_transformation"),
            "semantic_uniformity": profile.get("semantic_uniformity"),
            "discourse_regularity": profile.get("discourse_regularity"),
            "repeated_paragraph_role_runs": base.get("repeated_paragraph_role_runs"),
            "role_counts": role_counts,
        },
        "score_drag_blocks": low_value_blocks[:8],
        "generic_sentence_ratio": base.get("generic_sentence_ratio"),
        "generic_sentence_count": base.get("generic_sentence_count"),
        "sentence_count": base.get("sentence_count"),
        "paragraph_count": base.get("paragraph_count"),
        "ranked_blocks": ranked_blocks[:12],
        "paragraphs": paragraphs,
    }


def texture_candidate_family(operation: str | None) -> str:
    operation = str(operation or "").lower()
    if "authorship" in operation:
        return "AUTHORSHIP_SUPPRESSION"
    if "transformation" in operation or "template" in operation or "merge" in operation:
        return "TRANSFORMATION_DETEMPLATE"
    if "external_proxy" in operation or "generic_assertion_collapse" in operation:
        return "HYBRID_TEXTURE_COLLAPSE"
    if "removal" in operation or "remove" in operation:
        return "LOW_VALUE_REMOVE"
    return "HYBRID_TEXTURE_COLLAPSE"


def authorship_transformation_texture_candidates(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 12,
    deps: PostTopkTextureHelperDeps,
) -> list[tuple[str, str, dict]]:
    candidates = deps.post_topk_convergence_candidates(source_text, raw_json, limit=limit)
    mapped: list[tuple[str, str, dict]] = []
    for strategy, candidate, meta in candidates:
        operation = str((meta or {}).get("operation") or "")
        family = texture_candidate_family(operation)
        strategy_name = str(strategy or "texture_candidate")
        strategy_name = re.sub(r"^post_topk_", "texture_", strategy_name)
        if not strategy_name.startswith("texture_"):
            strategy_name = f"texture_{strategy_name}"
        mapped.append((
            strategy_name,
            candidate,
            {
                **(meta or {}),
                "authorship_transformation_texture_controller": True,
                "texture_candidate_family": family,
                "expected_driver": (
                    "ai_authorship" if family == "AUTHORSHIP_SUPPRESSION"
                    else "ai_transformation" if family == "TRANSFORMATION_DETEMPLATE"
                    else "ai_transformation" if family == "HYBRID_TEXTURE_COLLAPSE"
                    else "ai_transformation"
                ),
            },
        ))
    return mapped[:max(1, limit)]
