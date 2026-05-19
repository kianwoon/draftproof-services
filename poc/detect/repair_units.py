"""Scanner-owned rewrite repair units.

This module turns structured scan evidence into downstream repair units. It
does not inspect content keywords; it uses sentence boundaries, numeric
predictability, localized blocker signals, and authorship-window scores already
produced by the scanner/report layer.
"""

from __future__ import annotations

import hashlib
from statistics import mean
from typing import Any


REPAIR_UNITS_SCHEMA_VERSION = "repair_units.v2"


def build_repair_units_v2(
    *,
    source_text: str,
    segments: list[dict[str, Any]],
    paragraph_rows: list[dict[str, Any]],
    blocker_radar: dict[str, Any] | None,
    authorship_window_profile: dict[str, Any] | None,
    rewrite_target_profile: dict[str, Any] | None = None,
    max_units: int = 8,
) -> dict[str, Any]:
    """Build scanner-owned units that a rewrite planner can execute against."""

    source = str(source_text or "")
    rows = _sentence_risk_rows(
        source_text=source,
        segments=segments,
        blocker_radar=blocker_radar or {},
        authorship_window_profile=authorship_window_profile or {},
    )
    eligible_rows = [
        row for row in rows
        if row["word_count"] > 0
        and _valid_bounds(source, row.get("start_char"), row.get("end_char"))
    ]
    selected = _selected_sentence_indexes(eligible_rows)
    clusters = _cluster_selected_rows(eligible_rows, selected)
    all_cluster_rows = [
        _cluster_contract_row(
            source=source,
            cluster=cluster,
            ordinal=ordinal,
            rewrite_target_profile=rewrite_target_profile or {},
        )
        for ordinal, cluster in enumerate(clusters, start=1)
    ]
    all_cluster_rows = [row for row in all_cluster_rows if row]
    all_cluster_rows.sort(
        key=lambda row: (
            float(row.get("impact_score") or 0.0),
            int(row.get("word_count") or 0),
            -int(row.get("start_sentence") or 0),
        ),
        reverse=True,
    )
    cluster_rows = all_cluster_rows[:max(1, int(max_units or 1))]
    repair_units = [
        _repair_unit_from_cluster(source=source, cluster=row, ordinal=ordinal)
        for ordinal, row in enumerate(cluster_rows, start=1)
    ]
    density_gate = _density_gate_from_clusters(eligible_rows, all_cluster_rows, top_clusters=cluster_rows)
    return {
        "schema_version": REPAIR_UNITS_SCHEMA_VERSION,
        "basis": "scanner_segments + blocker_radar + authorship_windows",
        "source_text_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "selection_policy": {
            "mode": "dynamic_numeric_sentence_risk_clusters",
            "content_keyword_matching": False,
            "sentence_count": len(eligible_rows),
            "selected_sentence_count": len(selected),
            "max_units": max(1, int(max_units or 1)),
        },
        "eligible_span_density_gate": density_gate,
        "repair_units": repair_units,
        "contract_checks": {
            "unit_count": len(repair_units),
            "all_units_have_valid_source_slice": all(
                unit.get("source_text") == source[int(unit.get("start_char") or 0):int(unit.get("end_char") or 0)]
                for unit in repair_units
            ),
            "all_units_start_on_clean_boundary": all(
                _clean_left_boundary(source, int(unit.get("start_char") or 0))
                for unit in repair_units
            ),
            "all_units_end_on_clean_boundary": all(
                _clean_right_boundary(source, int(unit.get("end_char") or 0))
                for unit in repair_units
            ),
            "all_units_have_sentence_ids": all(bool(unit.get("sentence_ids")) for unit in repair_units),
        },
        "paragraph_context": {
            "paragraph_count": len(paragraph_rows or []),
        },
    }


def _sentence_risk_rows(
    *,
    source_text: str,
    segments: list[dict[str, Any]],
    blocker_radar: dict[str, Any],
    authorship_window_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    blocker_by_sentence = _localized_blocker_scores(blocker_radar)
    paragraph_scores = _paragraph_window_scores(authorship_window_profile)
    rows: list[dict[str, Any]] = []
    aligned_segments = _aligned_segments(source_text, segments or [])
    for index, segment in enumerate(aligned_segments):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "")
        paragraph_id = str(segment.get("paragraph_id") or "")
        predictability = segment.get("predictability") if isinstance(segment.get("predictability"), dict) else {}
        signals = segment.get("signals") if isinstance(segment.get("signals"), list) else []
        signal_rows = [signal for signal in signals if isinstance(signal, dict)]
        signal_score = max((_score01(signal.get("score"), percent=True) for signal in signal_rows), default=0.0)
        blocker_score = blocker_by_sentence.get(str(segment.get("sentence_id") or ""), 0.0)
        paragraph_score = paragraph_scores.get(paragraph_id, 0.0)
        pred_score = _score01(predictability.get("score"))
        top10 = _score01(predictability.get("top10_ratio"))
        top50 = _score01(predictability.get("top50_ratio"))
        risk_score = min(
            100.0,
            100.0 * (
                pred_score * 0.36
                + top10 * 0.22
                + top50 * 0.14
                + signal_score * 0.16
                + blocker_score * 0.06
                + paragraph_score * 0.06
            ),
        )
        rows.append({
            "sentence_index": int(segment.get("sentence_index") if isinstance(segment.get("sentence_index"), int) else index),
            "sentence_id": str(segment.get("sentence_id") or f"s{index + 1:03d}"),
            "paragraph_id": paragraph_id,
            "start_char": _int_or_none(segment.get("start_char")),
            "end_char": _int_or_none(segment.get("end_char")),
            "text": text,
            "word_count": _word_count(text),
            "risk_score": round(risk_score, 3),
            "risk_components": {
                "predictability_score": round(pred_score, 4),
                "top10_ratio": round(top10, 4),
                "top50_ratio": round(top50, 4),
                "localized_blocker_score": round(blocker_score, 4),
                "paragraph_ai_assistance_score": round(paragraph_score, 4),
                "max_segment_signal_score": round(signal_score, 4),
            },
            "signal_keys": _signal_keys(signal_rows),
            "predictable_token_spans": _string_list(predictability.get("predictable_token_spans"), limit=8),
        })
    return rows


def _selected_sentence_indexes(rows: list[dict[str, Any]]) -> set[int]:
    if not rows:
        return set()
    scores = [float(row.get("risk_score") or 0.0) for row in rows]
    avg = mean(scores)
    ordered = sorted(rows, key=lambda row: float(row.get("risk_score") or 0.0), reverse=True)
    sentence_count = len(rows)
    dynamic_limit = max(1, min(sentence_count, round(sentence_count ** 0.5) + sentence_count // 5))
    floor_index = max(0, min(len(ordered) - 1, dynamic_limit - 1))
    dynamic_floor = min(avg, float(ordered[floor_index].get("risk_score") or 0.0))
    return {
        int(row["sentence_index"])
        for row in ordered[:dynamic_limit]
        if float(row.get("risk_score") or 0.0) >= dynamic_floor
    }


def _aligned_segments(source_text: str, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = str(source_text or "")
    aligned: list[dict[str, Any]] = []
    cursor = 0
    for segment in sorted(
        [row for row in segments if isinstance(row, dict)],
        key=lambda row: (
            _int_or_none(row.get("sentence_index")) if _int_or_none(row.get("sentence_index")) is not None else 10**9,
            _int_or_none(row.get("start_char")) if _int_or_none(row.get("start_char")) is not None else 10**9,
        ),
    ):
        sentence = str(segment.get("text") or "")
        start = _int_or_none(segment.get("start_char"))
        end = _int_or_none(segment.get("end_char"))
        if sentence:
            located = source.find(sentence, cursor)
            if located < 0 and start is not None:
                located = source.find(sentence, max(0, start - 240))
            if located >= 0:
                start = located
                end = located + len(sentence)
                cursor = end
        aligned.append({
            **segment,
            "start_char": start,
            "end_char": end,
        })
    return aligned


def _cluster_selected_rows(rows: list[dict[str, Any]], selected: set[int]) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    active: list[dict[str, Any]] = []
    previous_index: int | None = None
    previous_paragraph = ""
    for row in sorted(rows, key=lambda item: int(item.get("sentence_index") or 0)):
        index = int(row.get("sentence_index") or 0)
        paragraph = str(row.get("paragraph_id") or "")
        if index not in selected:
            if active:
                clusters.append(active)
                active = []
            previous_index = None
            previous_paragraph = ""
            continue
        adjacent = previous_index is not None and index == previous_index + 1 and paragraph == previous_paragraph
        if not active or adjacent:
            active.append(row)
        else:
            clusters.append(active)
            active = [row]
        previous_index = index
        previous_paragraph = paragraph
    if active:
        clusters.append(active)
    return clusters


def _cluster_contract_row(
    *,
    source: str,
    cluster: list[dict[str, Any]],
    ordinal: int,
    rewrite_target_profile: dict[str, Any],
) -> dict[str, Any] | None:
    if not cluster:
        return None
    start = min(_int_or_none(row.get("start_char")) or 0 for row in cluster)
    end = max(_int_or_none(row.get("end_char")) or 0 for row in cluster)
    if not _valid_bounds(source, start, end):
        return None
    sentence_ids = [str(row.get("sentence_id") or "") for row in cluster if str(row.get("sentence_id") or "")]
    paragraph_ids = sorted({str(row.get("paragraph_id") or "") for row in cluster if str(row.get("paragraph_id") or "")})
    total_words = sum(int(row.get("word_count") or 0) for row in cluster)
    risk_total = sum(float(row.get("risk_score") or 0.0) for row in cluster)
    drivers = _dominant_drivers(cluster, rewrite_target_profile)
    source_text = source[start:end]
    return {
        "cluster_id": f"scanner_cluster_{ordinal:03d}",
        "start_sentence": int(cluster[0].get("sentence_index") or 0),
        "end_sentence": int(cluster[-1].get("sentence_index") or 0),
        "sentence_ids": sentence_ids,
        "paragraph_ids": paragraph_ids,
        "start_char": start,
        "end_char": end,
        "sentence_count": len(cluster),
        "word_count": _word_count(source_text) or total_words,
        "risk_score": round(risk_total, 3),
        "average_sentence_risk": round(risk_total / max(1, len(cluster)), 3),
        "impact_score": round(risk_total * max(1, total_words), 3),
        "preview": " ".join(str(row.get("text") or "") for row in cluster)[:320],
        "dominant_drivers": drivers,
        "recommended_scope": "cluster_route_replacement",
        "source": "scanner.repair_units_v2",
    }


def _repair_unit_from_cluster(*, source: str, cluster: dict[str, Any], ordinal: int) -> dict[str, Any]:
    start = int(cluster.get("start_char") or 0)
    end = int(cluster.get("end_char") or start)
    source_text = source[start:end]
    return {
        "unit_id": f"ru{ordinal:03d}",
        "unit_type": "density_cluster",
        "source": "scanner.repair_units_v2",
        "cluster_id": cluster.get("cluster_id"),
        "start_char": start,
        "end_char": end,
        "start_sentence": cluster.get("start_sentence"),
        "end_sentence": cluster.get("end_sentence"),
        "sentence_ids": list(cluster.get("sentence_ids") or []),
        "paragraph_ids": list(cluster.get("paragraph_ids") or []),
        "word_count": _word_count(source_text),
        "source_text": source_text,
        "source_excerpt": source_text[:420],
        "before_context": source[max(0, start - 260):start],
        "after_context": source[end:min(len(source), end + 260)],
        "dominant_drivers": list(cluster.get("dominant_drivers") or []),
        "recommended_scope": cluster.get("recommended_scope"),
        "prompt_contract": {
            "task": "cluster_route_replacement",
            "preserve_unit_meaning": True,
            "preserve_sentence_boundary_integrity": True,
            "avoid_full_document_rewrite": True,
            "validator_owns_score_truth": True,
        },
    }


def _density_gate_from_clusters(
    rows: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    *,
    top_clusters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    eligible_word_count = sum(int(row.get("word_count") or 0) for row in rows)
    unsafe_word_count = sum(int(cluster.get("word_count") or 0) for cluster in clusters)
    visible_clusters = top_clusters if isinstance(top_clusters, list) else clusters
    return {
        "version": "scanner_repair_units_density_v2",
        "safe": not clusters,
        "source": "scanner.repair_units_v2",
        "eligible_sentence_count": len(rows),
        "eligible_word_count": eligible_word_count,
        "unsafe_sentence_count": sum(int(cluster.get("sentence_count") or 0) for cluster in clusters),
        "unsafe_word_count": unsafe_word_count,
        "unsafe_eligible_word_ratio": round((unsafe_word_count / max(1, eligible_word_count)) * 100.0, 3),
        "longest_unsafe_span_words": max([int(cluster.get("word_count") or 0) for cluster in clusters] or [0]),
        "unsafe_cluster_count": len(clusters),
        "top_unsafe_clusters": [
            {
                "start_sentence": cluster.get("start_sentence"),
                "end_sentence": cluster.get("end_sentence"),
                "start_char": cluster.get("start_char"),
                "end_char": cluster.get("end_char"),
                "sentence_ids": cluster.get("sentence_ids"),
                "paragraph_ids": cluster.get("paragraph_ids"),
                "sentence_count": cluster.get("sentence_count"),
                "word_count": cluster.get("word_count"),
                "risk_score": cluster.get("risk_score"),
                "average_sentence_risk": cluster.get("average_sentence_risk"),
                "impact_score": cluster.get("impact_score"),
                "preview": cluster.get("preview"),
                "dominant_drivers": cluster.get("dominant_drivers"),
                "source": cluster.get("source"),
            }
            for cluster in visible_clusters
        ],
        "recommended_actions": [
            "target_scanner_owned_repair_unit",
            "replace_cluster_route",
            "rescan_full_document_before_accepting",
        ] if clusters else ["preserve_scanner_safe_output"],
    }


def _dominant_drivers(cluster: list[dict[str, Any]], rewrite_target_profile: dict[str, Any]) -> list[dict[str, Any]]:
    totals: dict[str, float] = {}
    for row in cluster:
        components = row.get("risk_components") if isinstance(row.get("risk_components"), dict) else {}
        for key, value in components.items():
            numeric = _float_or_zero(value)
            if numeric > 0:
                totals[key] = totals.get(key, 0.0) + numeric
        for key in row.get("signal_keys") or []:
            totals[str(key)] = totals.get(str(key), 0.0) + 0.5
    driver_summary = rewrite_target_profile.get("driver_summary") if isinstance(rewrite_target_profile.get("driver_summary"), dict) else {}
    for key, value in driver_summary.items():
        if key in totals:
            totals[key] += _float_or_zero(value) * 0.05
    rows = [
        {"key": key, "score": round(value / max(1, len(cluster)), 3)}
        for key, value in totals.items()
    ]
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:6]


def _localized_blocker_scores(blocker_radar: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for blocker in blocker_radar.get("blockers") or []:
        if not isinstance(blocker, dict):
            continue
        sentence_ids = blocker.get("sentence_ids") if isinstance(blocker.get("sentence_ids"), list) else []
        if not sentence_ids:
            continue
        score = _score01(blocker.get("score"), percent=True)
        for sentence_id in sentence_ids:
            key = str(sentence_id or "")
            if key:
                result[key] = max(result.get(key, 0.0), score)
    return result


def _paragraph_window_scores(authorship_window_profile: dict[str, Any]) -> dict[str, float]:
    rows = authorship_window_profile.get("windows") if isinstance(authorship_window_profile.get("windows"), list) else []
    result: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        paragraph_id = str(row.get("paragraph_id") or "")
        if paragraph_id:
            result[paragraph_id] = max(result.get(paragraph_id, 0.0), _score01(row.get("ai_assistance_score")))
    return result


def _signal_keys(signals: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for signal in signals:
        key = str(signal.get("key") or "").strip()
        if key and key not in keys:
            keys.append(key)
        if len(keys) >= 8:
            break
    return keys


def _score01(value: Any, *, percent: bool = False) -> float:
    numeric = _float_or_zero(value)
    if percent or numeric > 1.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))


def _float_or_zero(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _valid_bounds(source: str, start: Any, end: Any) -> bool:
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    return 0 <= start < end <= len(source)


def _clean_left_boundary(source: str, start: int) -> bool:
    if start <= 0:
        return True
    if start >= len(source):
        return True
    previous = _previous_non_space(source, start)
    return previous is None or source[previous] in ".!?\n"


def _clean_right_boundary(source: str, end: int) -> bool:
    if end >= len(source):
        return True
    previous = _previous_non_space(source, end)
    if previous is None:
        return True
    if source[previous] in ".!?":
        return True
    next_index = _next_non_space(source, end)
    return next_index is None or source[next_index] == "\n"


def _previous_non_space(source: str, end: int) -> int | None:
    index = min(len(source), max(0, end)) - 1
    while index >= 0:
        if not source[index].isspace():
            return index
        index -= 1
    return None


def _next_non_space(source: str, start: int) -> int | None:
    index = max(0, min(len(source), start))
    while index < len(source):
        if not source[index].isspace():
            return index
        index += 1
    return None


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    rows: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            rows.append(text)
        if len(rows) >= limit:
            break
    return rows
