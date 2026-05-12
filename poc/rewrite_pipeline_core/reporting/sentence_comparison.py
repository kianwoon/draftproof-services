"""Sentence comparison helpers for rewrite reports."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def _detail_value(detail: dict, *keys, default=0):
    """Read the first present metric key from a sentence detail dict."""
    for key in keys:
        value = detail.get(key)
        if value is not None:
            return value
    return default


def _scan_scope_summary(report_dict: dict) -> dict:
    """Small diagnostic summary of how much text the detector scored."""
    pred = (report_dict or {}).get("predictability") or {}
    if not isinstance(pred, dict):
        return {}
    score_derivation = pred.get("score_derivation") or {}
    sentences = pred.get("sentences") or []
    all_sentences = pred.get("all_sentences") or []
    scope = {
        "predictability_scored_sentences": len(sentences),
    }
    if all_sentences:
        scope["predictability_total_sentences"] = len(all_sentences)
    included = score_derivation.get("included_sentence_count")
    if included is not None:
        scope["predictability_included_sentence_count"] = included
    raw_mean = score_derivation.get("raw_mean")
    if isinstance(raw_mean, (int, float)):
        scope["predictability_raw_mean"] = round(float(raw_mean), 4)
    return scope


def _sentence_detail_lookup(details: list) -> dict:
    """Map sentence text to metric details, preserving the first occurrence."""
    lookup = {}
    for d in details or []:
        sentence = (d.get("sentence") or "").strip()
        if sentence and sentence not in lookup:
            lookup[sentence] = d
    return lookup


def _comparison_sentences(text: str) -> list[str]:
    """Split comparison text without merging headings into adjacent prose."""
    rows: list[str] = []
    for block in str(text or "").splitlines():
        block = block.strip()
        if not block:
            continue
        rows.extend(
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", block)
            if s.strip()
        )
    return rows


def _build_aligned_sentence_comparison(mp) -> list:
    """Build before/after sentence rows using text alignment, not index pairing."""
    if not mp:
        return []

    original_sentences = _comparison_sentences(mp.original_text or "")
    final_sentences = _comparison_sentences(mp.final_text or "")
    if not original_sentences and not final_sentences:
        return []

    orig_details = (mp.original_metrics.sentence_details if mp.original_metrics else []) or []
    final_details = (mp.final_metrics.sentence_details if mp.final_metrics else []) or []
    orig_lookup = _sentence_detail_lookup(orig_details)
    final_lookup = _sentence_detail_lookup(final_details)

    rows = []
    matcher = SequenceMatcher(a=original_sentences, b=final_sentences, autojunk=False)
    row_index = 1
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for o_sent, n_sent in zip(original_sentences[i1:i2], final_sentences[j1:j2]):
                o = orig_lookup.get(o_sent, {})
                n = final_lookup.get(n_sent, {})
                rows.append({
                    "index": row_index,
                    "orig_tier": _detail_value(o, "label", "risk_label", default="?"),
                    "orig_risk": _detail_value(o, "risk", "predictability_risk"),
                    "orig_top10": _detail_value(o, "top10_ratio", "top_10_ratio"),
                    "orig_sentence": o_sent,
                    "new_tier": _detail_value(n, "label", "risk_label", default="?"),
                    "new_risk": _detail_value(n, "risk", "predictability_risk"),
                    "new_top10": _detail_value(n, "top10_ratio", "top_10_ratio"),
                    "new_sentence": n_sent,
                })
                row_index += 1
            continue

        old_block = original_sentences[i1:i2]
        new_block = final_sentences[j1:j2]
        if max(len(old_block), len(new_block), 0) <= 2:
            o_sent = " ".join(old_block).strip()
            n_sent = " ".join(new_block).strip()
            o = orig_lookup.get(old_block[0], {}) if old_block else {}
            n = final_lookup.get(new_block[0], {}) if new_block else {}
            rows.append({
                "index": row_index,
                "orig_tier": _detail_value(o, "label", "risk_label", default="?"),
                "orig_risk": _detail_value(o, "risk", "predictability_risk"),
                "orig_top10": _detail_value(o, "top10_ratio", "top_10_ratio"),
                "orig_sentence": o_sent,
                "new_tier": _detail_value(n, "label", "risk_label", default="?"),
                "new_risk": _detail_value(n, "risk", "predictability_risk"),
                "new_top10": _detail_value(n, "top10_ratio", "top_10_ratio"),
                "new_sentence": n_sent,
            })
            row_index += 1
            continue
        block_len = max(len(old_block), len(new_block), 1)
        for offset in range(block_len):
            o_sent = old_block[offset].strip() if offset < len(old_block) else ""
            n_sent = new_block[offset].strip() if offset < len(new_block) else ""
            o = orig_lookup.get(o_sent, {}) if o_sent else {}
            n = final_lookup.get(n_sent, {}) if n_sent else {}
            rows.append({
                "index": row_index,
                "orig_tier": _detail_value(o, "label", "risk_label", default="?"),
                "orig_risk": _detail_value(o, "risk", "predictability_risk"),
                "orig_top10": _detail_value(o, "top10_ratio", "top_10_ratio"),
                "orig_sentence": o_sent,
                "new_tier": _detail_value(n, "label", "risk_label", default="?"),
                "new_risk": _detail_value(n, "risk", "predictability_risk"),
                "new_top10": _detail_value(n, "top10_ratio", "top_10_ratio"),
                "new_sentence": n_sent,
            })
            row_index += 1
    return rows


def sanitize_text(text: str) -> str:
    """Fix mojibake and normalize Unicode in text before processing."""
    text = text.replace('â€™', "'").replace('â€˜', "'")
    text = text.replace('â€œ', '"').replace('â€\x9d', '"')
    text = text.replace('â€"', ' -- ').replace('â€"', '-')
    text = text.replace('â€¦', '...')
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('—', ' -- ').replace('–', '-')
    text = text.replace('…', '...')
    text = text.replace(' ', ' ')
    text = re.sub(r'  +', ' ', text)
    return text
