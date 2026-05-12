"""Post-Top-k convergence candidate builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math


@dataclass(frozen=True)
class PostTopkConvergenceDeps:
    env_flag: Callable[..., bool]
    strict_ai_safe_band_status: Callable[[dict | None], dict]
    safe_topk_calibrated_limit: Callable[[], float]
    logical_paragraphs: Callable[[str], list[str]]
    post_topk_driver_map: Callable[[str, dict | None], dict]
    text_word_count: Callable[[str], int]
    float_env: Callable[[str, float], float]
    join_logical_paragraphs: Callable[[list[str]], str]
    split_sentences: Callable[[str], list[str]]
    narrow_generic_claim_text: Callable[[str], str]
    compress_score_drag_paragraph: Callable[..., str]
    post_topk_template_opening_re: Any


def build_post_topk_convergence_candidates(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 10,
    deps: PostTopkConvergenceDeps | None = None,
) -> list[tuple[str, str, dict]]:
    """Build stronger post-Top-k candidates from document-wide driver movement.

    The Top-k rebuild already solved token-route risk. This phase is allowed to
    shorten or collapse generic material when the remaining strict blockers are
    authorship/transformation/proxy drivers.
    """
    if deps is None:
        raise ValueError("PostTopkConvergenceDeps is required")
    if not deps.env_flag("DRAFTPROOF_POST_TOPK_CONVERGENCE_OPTIMIZER", True):
        return []
    strict = deps.strict_ai_safe_band_status(raw_json)
    profile = strict.get("profile") or {}
    if float(profile.get("topk_calibrated_risk", 100.0)) >= deps.safe_topk_calibrated_limit():
        return []
    paragraphs = deps.logical_paragraphs(source_text)
    if len(paragraphs) < 2:
        return []
    driver_map = deps.post_topk_driver_map(source_text, raw_json)
    source_words = deps.text_word_count(source_text)
    # Strict-safe convergence is allowed to shorten low-value generic material.
    # Semantic drift/protected-anchor checks still gate the candidate after scan.
    min_words = max(30, int(source_words * deps.float_env("DRAFTPROOF_POST_TOPK_MIN_WORD_RATIO", 0.20)))
    candidates: list[tuple[str, str, dict]] = []
    seen: set[str] = {str(source_text or "").strip()}

    def add(strategy: str, next_paragraphs: list[str], meta: dict) -> None:
        cleaned_paragraphs = [paragraph.strip() for paragraph in next_paragraphs if paragraph and paragraph.strip()]
        candidate = deps.join_logical_paragraphs(cleaned_paragraphs)
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            return
        if deps.text_word_count(candidate) < min_words:
            return
        seen.add(normalized)
        candidates.append((
            strategy,
            candidate,
            {
                **meta,
                "post_topk_convergence": True,
                "post_topk_driver_map": {
                    "generic_sentence_ratio": driver_map.get("generic_sentence_ratio"),
                    "generic_sentence_count": driver_map.get("generic_sentence_count"),
                    "sentence_count": driver_map.get("sentence_count"),
                    "repeated_paragraph_role_runs": driver_map.get("repeated_paragraph_role_runs"),
                },
            },
        ))

    sentence_targets = []
    paragraph_rows = {
        int(row.get("paragraph_index", -1)): row
        for row in driver_map.get("paragraphs") or []
        if isinstance(row, dict)
    }
    for row in driver_map.get("paragraphs") or []:
        paragraph_index = int(row.get("paragraph_index", -1))
        if paragraph_index < 0 or paragraph_index >= len(paragraphs):
            continue
        for sentence_row in row.get("sentences") or []:
            if sentence_row.get("protected"):
                continue
            sentence_targets.append((
                float(sentence_row.get("driver_score") or 0.0),
                paragraph_index,
                int(sentence_row.get("sentence_index", 0) or 0),
                bool(sentence_row.get("contextual")),
                sentence_row.get("text") or "",
            ))
    sentence_targets.sort(reverse=True)

    removable_targets = [
        item for item in sentence_targets
        if item[0] >= deps.float_env("DRAFTPROOF_POST_TOPK_SENTENCE_DRIVER_MIN", 4.0)
        and not item[3]
    ]
    if not removable_targets:
        removable_targets = [
            item for item in sentence_targets
            if item[0] >= deps.float_env("DRAFTPROOF_POST_TOPK_CONTEXTUAL_SENTENCE_DRIVER_MIN", 7.5)
        ]

    for fraction in (0.18, 0.28, 0.40):
        if len(candidates) >= max(1, limit):
            break
        remove_count = max(1, int(math.ceil(len(removable_targets) * fraction)))
        remove_pairs = {
            (paragraph_index, sentence_index)
            for _score, paragraph_index, sentence_index, _contextual, _sentence in removable_targets[:remove_count]
        }
        if not remove_pairs:
            continue
        next_paragraphs = []
        removed = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            sentences = deps.split_sentences(paragraph)
            kept = []
            for sentence_index, sentence in enumerate(sentences):
                if (paragraph_index, sentence_index) in remove_pairs and len(sentences) - len([
                    pair for pair in remove_pairs if pair[0] == paragraph_index
                ]) >= 1:
                    removed.append({"paragraph_index": paragraph_index, "sentence_index": sentence_index})
                    continue
                kept.append(sentence)
            if kept:
                next_paragraphs.append(deps.narrow_generic_claim_text(" ".join(kept).strip()))
        add(
            f"post_topk_generic_assertion_collapse_{int(fraction * 100)}",
            next_paragraphs,
            {
                "operation": "generic_assertion_collapse",
                "removed_sentence_count": len(removed),
                "removed_sentences": removed[:20],
            },
        )

    authorship_rows = [
        row for row in driver_map.get("paragraphs") or []
        if not row.get("has_protected_anchor")
        and int(row.get("sentence_count") or 0) >= 2
        and float(row.get("paragraph_driver_score") or 0.0) >= 8.0
    ]
    for row in authorship_rows[:3]:
        if len(candidates) >= max(1, limit):
            break
        paragraph_index = int(row.get("paragraph_index", -1))
        if paragraph_index < 0 or paragraph_index >= len(paragraphs):
            continue
        sentences = deps.split_sentences(paragraphs[paragraph_index])
        sentence_rows = sorted(
            [
                sentence_row for sentence_row in row.get("sentences") or []
                if not sentence_row.get("protected")
            ],
            key=lambda item: float(item.get("driver_score") or 0.0),
            reverse=True,
        )
        if len(sentences) < 2 or not sentence_rows:
            continue
        remove_indexes = {
            int(sentence_rows[0].get("sentence_index", 0))
        }
        kept = [
            sentence for sentence_index, sentence in enumerate(sentences)
            if sentence_index not in remove_indexes
        ]
        if not kept:
            continue
        next_paragraphs = list(paragraphs)
        next_paragraphs[paragraph_index] = deps.narrow_generic_claim_text(" ".join(kept).strip())
        add(
            f"post_topk_authorship_suppression_candidate_p{paragraph_index + 1}",
            next_paragraphs,
            {
                "operation": "authorship_suppression_candidate",
                "paragraph_index": paragraph_index,
                "removed_sentence_indexes": sorted(remove_indexes),
                "paragraph_role": row.get("role"),
            },
        )

    high_drag_rows = [
        row for row in driver_map.get("paragraphs") or []
        if not row.get("has_protected_anchor")
        and float(row.get("generic_sentence_ratio") or 0.0) >= 0.60
        and int(row.get("sentence_count") or 0) >= 2
    ]
    for row in high_drag_rows[:3]:
        if len(candidates) >= max(1, limit):
            break
        paragraph_index = int(row.get("paragraph_index", -1))
        if paragraph_index < 0 or paragraph_index >= len(paragraphs):
            continue
        paragraph = paragraphs[paragraph_index]
        compressed = deps.compress_score_drag_paragraph(
            paragraph,
            max_remove=max(1, min(3, int(row.get("sentence_count") or 1) - 1)),
        )
        if compressed.strip() and compressed.strip() != paragraph.strip():
            next_paragraphs = list(paragraphs)
            next_paragraphs[paragraph_index] = compressed
            add(
                f"post_topk_structure_de_template_p{paragraph_index + 1}",
                next_paragraphs,
                {
                    "operation": "structure_de_template",
                    "paragraph_index": paragraph_index,
                    "paragraph_role": row.get("role"),
                },
            )

    symmetry_rows = [
        row for row in driver_map.get("paragraphs") or []
        if not row.get("has_protected_anchor")
        and int(row.get("sentence_count") or 0) >= 3
        and (
            float(row.get("generic_sentence_ratio") or 0.0) >= 0.34
            or row.get("role") in {"generic_claim_heavy", "conclusion_template_risk"}
        )
    ]
    for row in symmetry_rows[:2]:
        if len(candidates) >= max(1, limit):
            break
        paragraph_index = int(row.get("paragraph_index", -1))
        if paragraph_index < 0 or paragraph_index >= len(paragraphs):
            continue
        sentences = deps.split_sentences(paragraphs[paragraph_index])
        if len(sentences) < 3:
            continue
        compressed_sentences = []
        for sentence in sentences:
            narrowed = deps.narrow_generic_claim_text(sentence)
            if not deps.post_topk_template_opening_re.search(narrowed):
                compressed_sentences.append(narrowed)
        if len(sentences) >= 3:
            compressed_sentences = [sentences[0], sentences[-1]]
        next_paragraphs = list(paragraphs)
        next_paragraphs[paragraph_index] = " ".join(s.strip() for s in compressed_sentences if s.strip())
        add(
            f"post_topk_transformation_reduction_candidate_p{paragraph_index + 1}",
            next_paragraphs,
            {
                "operation": "transformation_reduction_candidate",
                "paragraph_index": paragraph_index,
                "paragraph_role": row.get("role"),
            },
        )

    low_value_rows = [
        row for row in driver_map.get("paragraphs") or []
        if row.get("low_value_generic_block")
        and not row.get("has_protected_anchor")
        and int(row.get("word_count") or 0) >= 12
    ]
    for remove_count in (1, 2, 3):
        if len(candidates) >= max(1, limit):
            break
        indexes = {
            int(row.get("paragraph_index", -1))
            for row in low_value_rows[:remove_count]
            if int(row.get("paragraph_index", -1)) >= 0
        }
        if not indexes or len(indexes) >= len(paragraphs):
            continue
        next_paragraphs = [
            paragraph for index, paragraph in enumerate(paragraphs)
            if index not in indexes
        ]
        add(
            f"post_topk_low_value_block_removal_candidate_{remove_count}",
            next_paragraphs,
            {
                "operation": "low_value_block_removal_candidate",
                "removed_paragraph_indexes": sorted(indexes),
            },
        )

    if low_value_rows and removable_targets and len(candidates) < max(1, limit):
        indexes = {
            int(row.get("paragraph_index", -1))
            for row in low_value_rows[:1]
            if int(row.get("paragraph_index", -1)) >= 0
        }
        remove_pairs = {
            (paragraph_index, sentence_index)
            for _score, paragraph_index, sentence_index, _contextual, _sentence in removable_targets[:2]
            if paragraph_index not in indexes
        }
        next_paragraphs = []
        removed_sentences = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            if paragraph_index in indexes:
                continue
            sentences = deps.split_sentences(paragraph)
            kept = []
            for sentence_index, sentence in enumerate(sentences):
                if (paragraph_index, sentence_index) in remove_pairs and len(sentences) > 1:
                    removed_sentences.append({"paragraph_index": paragraph_index, "sentence_index": sentence_index})
                    continue
                kept.append(sentence)
            if kept:
                next_paragraphs.append(deps.narrow_generic_claim_text(" ".join(kept).strip()))
        add(
            "post_topk_external_proxy_reduction_candidate",
            next_paragraphs,
            {
                "operation": "external_proxy_reduction_candidate",
                "removed_paragraph_indexes": sorted(indexes),
                "removed_sentences": removed_sentences,
            },
        )

    if len(candidates) < max(1, limit) and len(paragraphs) >= 4:
        next_paragraphs = []
        merged_indexes = []
        skip_next = False
        for index, paragraph in enumerate(paragraphs):
            if skip_next:
                skip_next = False
                continue
            row = paragraph_rows.get(index, {})
            next_row = paragraph_rows.get(index + 1, {})
            if (
                index + 1 < len(paragraphs)
                and not row.get("has_protected_anchor")
                and not next_row.get("has_protected_anchor")
                and int(row.get("sentence_count") or 0) <= 2
                and int(next_row.get("sentence_count") or 0) <= 2
                and (
                    float(row.get("generic_sentence_ratio") or 0.0) >= 0.50
                    or float(next_row.get("generic_sentence_ratio") or 0.0) >= 0.50
                )
            ):
                next_paragraphs.append(deps.narrow_generic_claim_text(f"{paragraph} {paragraphs[index + 1]}"))
                merged_indexes.append([index, index + 1])
                skip_next = True
            else:
                next_paragraphs.append(paragraph)
        if merged_indexes:
            add(
                "post_topk_paragraph_merge_de_template",
                next_paragraphs,
                {"operation": "paragraph_merge_de_template", "merged_paragraph_indexes": merged_indexes},
            )

    operation_priority = {
        "external_proxy_reduction_candidate": 0,
        "authorship_suppression_candidate": 1,
        "transformation_reduction_candidate": 2,
        "generic_assertion_collapse": 3,
        "structure_de_template": 4,
        "low_value_block_removal_candidate": 5,
        "paragraph_merge_de_template": 6,
    }
    candidates.sort(key=lambda row: operation_priority.get(str((row[2] or {}).get("operation") or ""), 99))
    return candidates[:max(1, limit)]
