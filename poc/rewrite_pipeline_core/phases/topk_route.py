"""Top-k route optimizer helpers and prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import json
import math
import re

from detect.topk_calibration import calibrate_topk_risk


@dataclass(frozen=True)
class TopkRouteDeps:
    text_word_count: Callable[[str], int]
    float_env: Callable[[str, float], float]
    split_sentences: Callable[[str], list[str]]
    safe_topk_calibrated_limit: Callable[[], float]
    protected_anchor_brief_for_prompt: Callable[..., list[dict]]
    contribution_scores: Callable[[dict | None], dict]


def topk_optimizer_sentence_limit(text: str, *, deps: TopkRouteDeps) -> int:
    words = deps.text_word_count(text)
    if words <= 700:
        return int(deps.float_env("DRAFTPROOF_TOPK_ROUTE_SHORT_SENTENCES", 8.0))
    if words <= 1800:
        return int(deps.float_env("DRAFTPROOF_TOPK_ROUTE_MEDIUM_SENTENCES", 12.0))
    return int(deps.float_env("DRAFTPROOF_TOPK_ROUTE_LONG_SENTENCES", 20.0))


def topk_repair_map(text: str, report_dict: dict | None, *, limit: int | None = None, deps: TopkRouteDeps) -> dict:
    """Rank sentence/token routes that directly drive top-k saturation."""
    limit = max(1, int(limit or topk_optimizer_sentence_limit(text, deps=deps)))
    predictability = (report_dict or {}).get("predictability") or {}
    source_rows = predictability.get("all_sentences") or predictability.get("sentences") or []
    rows: list[dict] = []
    for index, item in enumerate(source_rows):
        if not isinstance(item, dict):
            continue
        sentence = str(item.get("sentence") or item.get("text") or "").strip()
        if not sentence:
            continue
        top10 = float(item.get("top10_ratio") or item.get("top_10_ratio") or item.get("top10") or 0.0)
        top50 = float(item.get("top50_ratio") or item.get("top_50_ratio") or item.get("top50") or 0.0)
        risk = float(item.get("predictability_risk") or item.get("risk") or item.get("score") or 0.0)
        spans = [
            str(span).strip()
            for span in (item.get("predictable_token_spans") or [])
            if str(span).strip()
        ][:6]
        tokens = [
            token for token in (item.get("top_predicted_tokens") or [])
            if isinstance(token, dict)
        ][:10]
        generic_opening = bool(re.search(
            r"^(?:The|This|These|It|Another|One of|In addition|At the same time|Despite|However|In conclusion|Overall)\b",
            sentence,
            re.I,
        ))
        rows.append({
            "sentence_id": item.get("sentence_id") or f"s{index + 1:03d}",
            "sentence_index": index,
            "paragraph_id": item.get("paragraph_id") or "",
            "sentence": sentence,
            "top10_ratio": round(top10, 4),
            "top50_ratio": round(top50, 4),
            "predictability_risk": round(risk, 4),
            "predictable_token_spans": spans,
            "top_predicted_tokens": tokens,
            "drivers": {
                "generic_opening": generic_opening,
                "high_top10": top10 >= 0.65,
                "span_count": len(spans),
            },
            "route_score": round(top10 * 0.55 + top50 * 0.20 + risk * 0.20 + (0.05 if generic_opening else 0.0), 4),
        })
    rows.sort(key=lambda row: row["route_score"], reverse=True)
    selected = rows[:limit]
    ai_components = ((report_dict or {}).get("ai_risk_badge") or {}).get("ai_components") or {}
    raw_topk = ai_components.get("topk_pattern_raw", ai_components.get("topk_pattern"))
    calibrated_topk = ai_components.get("topk_calibrated_risk")
    if not isinstance(calibrated_topk, (int, float)):
        calibrated_topk = calibrate_topk_risk(raw_topk, eligible_sentence_count=3).get("topk_calibrated_risk")
    return {
        "enabled": True,
        "limit": limit,
        "saturated": float(calibrated_topk or 0.0) >= deps.float_env(
            "DRAFTPROOF_AI_FOOTPRINT_ACTIVE_TOPK_THRESHOLD",
            90.0,
        ),
        "topk_pattern": raw_topk,
        "topk_pattern_raw": ai_components.get("topk_pattern_raw", raw_topk),
        "topk_calibrated_risk": calibrated_topk,
        "topk_safe_band": ai_components.get("topk_safe_band"),
        "targets": selected,
        "target_sentence_ids": [row.get("sentence_id") for row in selected],
    }


def deterministic_topk_route_sentence(sentence: str) -> tuple[str, list[str]]:
    """Make local route changes that attack predictability without adding facts."""
    original = str(sentence or "").strip()
    candidate = original
    operations: list[str] = []

    strong_routes = [
        (
            r"^One of the biggest strengths of (?P<subject>.+?) is (?P<claim>[^.]+)\.$",
            lambda m: f"{m.group('claim').strip().capitalize()} is where {m.group('subject').strip()} still carries weight.",
            "ranked_claim_route_strong",
        ),
        (
            r"^(?P<subject>The [^.]{3,80}?) has one of the largest (?P<asset>[^.]{3,80}?) in the world and is home to many (?P<group>[^.]+)\.$",
            lambda m: (
                f"Large {m.group('asset').strip()}, many {m.group('group').strip()}, and global reach: "
                f"that is part of {m.group('subject').strip().lower()}'s position."
            ),
            "largest_asset_route",
        ),
        (
            r"^(?P<subject>.+?) was founded in (?P<year>\d{4}) after (?P<event>[^.]+)\.$",
            lambda m: (
                f"{m.group('year')} matters here: {m.group('event').strip()}, "
                f"and {m.group('subject').strip()} began from that break."
            ),
            "founding_date_route",
        ),
        (
            r"^Millions of people from different countries moved to (?P<place>.+?) in search of better opportunities and a new life\.$",
            lambda m: (
                f"Better work, safety, a new life: those hopes brought millions of people from different countries "
                f"to {m.group('place').strip()}."
            ),
            "migration_motive_route",
        ),
        (
            r"^In addition to (?P<context>.+?), (?P<subject>.+?) has a strong cultural influence\.$",
            lambda m: (
                f"Culture is another route of influence for {m.group('subject').strip()}, "
                f"beyond {m.group('context').strip()}."
            ),
            "culture_influence_route",
        ),
        (
            r"^The country was built on ideas such as (?P<ideas>[^.]+)\.$",
            lambda m: (
                f"Ideas such as {m.group('ideas').strip()} sat near the centre of how the country described itself."
            ),
            "idea_list_route",
        ),
        (
            r"^At the same time, (?P<subject>.+?) has also created challenges related to (?P<issues>[^.]+)\.$",
            lambda m: f"Still, {m.group('subject').strip()} also brings harder questions: {m.group('issues').strip()}.",
            "challenge_list_route",
        ),
        (
            r"^The (?P<movement>[^.]{3,80} Movement) was an important period that aimed to (?P<aim>[^.]+)\.$",
            lambda m: f"The {m.group('movement').strip()} had a direct aim: {m.group('aim').strip()}.",
            "movement_aim_route",
        ),
    ]
    for pattern, replacement, op in strong_routes:
        match = re.match(pattern, candidate, flags=re.I)
        if match:
            updated = replacement(match)
            if updated and updated != candidate:
                candidate = updated
                operations.append(op)
                break

    replacements = [
        (r"^This becomes clear when\b", r"The gap shows up when", "this_opening_route"),
        (r"^The challenge is more than\b", r"The challenge is not just", "challenge_opening_route"),
        (r"^The challenge in ([^.]+?) does not only appear\b", r"In \1, the issue does not only appear", "challenge_scope_route"),
        (r"^When ([A-Za-z][^.]{2,80}?)\b", r"Once \1", "when_route"),
        (r"^A demonstration reveals\b", r"A demonstration can show", "demonstration_route"),
        (r"^(.{4,80}?) should not be misunderstood as\b", r"\1 is not", "misunderstood_as_route"),
        (r"^(.{4,80}?) limits remain\.$", r"\1 limits remain. That part is hard to ignore.", "short_followup_route"),
        (r"^This connects with\b", r"That links back to", "this_opening_route"),
        (r"^This creates\b", r"That creates", "this_opening_route"),
        (r"^This does not\b", r"It does not", "this_opening_route"),
        (r"^Instead, it comes from\b", r"Instead, the control comes from", "instead_route"),
        (r"^([A-Z][A-Za-z ]{2,60}) should not depend on\b", r"\1 cannot be left to", "dependency_route"),
        (r"^The ([A-Z][A-Za-z ]{2,60}) is often described as ", r"\1 is often described this way: ", "opening_route"),
        (r"^It has shaped ", r"Its influence reaches into ", "pronoun_opening_route"),
        (r"^The country has ", r"Inside the country, there is ", "country_opening_route"),
        (r"^One of the biggest strengths of ([^,]+) is ", r"For \1, one clear strength is ", "ranked_claim_route"),
        (r"^Another important feature of ([^,]+) is ", r"Another part of \1 is ", "feature_route"),
        (r"^In addition to ", r"Beyond ", "connector_remove"),
        (r"^At the same time, ", r"Still, ", "connector_shorten"),
        (r"^Despite its success, ", r"That success has limits. ", "fragment_route"),
        (r"^However, ", r"But ", "connector_plain"),
        (r"^In conclusion, ", r"Taken together, ", "conclusion_route"),
        (r"\bis known for\b", "is often linked with", "bland_verb_route"),
        (r"\bplays a major role in\b", "matters in", "formula_route"),
        (r"\bhas a strong influence\b", "carries influence", "bland_verb_route"),
        (r"\bhas become one of the\b", "now works as one of the", "formula_route"),
        (r"\bThis has led to\b", "That leaves", "formula_route"),
        (r"\bThis is why\b", "That is where", "formula_route"),
    ]
    for pattern, replacement, op in replacements:
        updated = re.sub(pattern, replacement, candidate, count=1, flags=re.I)
        if updated != candidate:
            candidate = updated
            operations.append(op)
    if candidate == original:
        because_match = re.search(r"\s+because\s+", candidate, flags=re.I)
        if because_match and len(candidate.split()) >= 18:
            left = candidate[:because_match.start()].strip()
            right = candidate[because_match.end():].strip()
            if left and right:
                candidate = f"{left}. Because {right[0].lower() + right[1:] if len(right) > 1 else right}"
                operations.append("because_route_split")
    if candidate == original:
        but_match = re.search(r"\s+but\s+", candidate, flags=re.I)
        if but_match and len(candidate.split()) >= 16:
            left = candidate[:but_match.start()].strip()
            right = candidate[but_match.end():].strip()
            if left and right:
                candidate = f"{left}. But {right[0].lower() + right[1:] if len(right) > 1 else right}"
                operations.append("contrast_route_split")
    if candidate == original:
        comma_parts = candidate.split(",", 1)
        if len(comma_parts) == 2 and 8 <= len(candidate.split()) <= 32:
            candidate = f"{comma_parts[1].strip()} {comma_parts[0].strip().lower()}."
            candidate = re.sub(r"\.\.$", ".", candidate)
            operations.append("clause_route_flip")
    candidate = re.sub(r"\s{2,}", " ", candidate).strip()
    return (candidate, operations) if candidate != original else (original, [])


def splice_sentences_by_text(text: str, replacements: dict[str, str]) -> str:
    candidate = str(text or "")
    for original, replacement in replacements.items():
        if original and replacement and original != replacement and original in candidate:
            candidate = candidate.replace(original, replacement, 1)
    return candidate


def remove_sentences_by_text(text: str, sentences: list[str]) -> str:
    candidate = str(text or "")
    for sentence in sentences or []:
        target = str(sentence or "").strip()
        if not target or target not in candidate:
            continue
        candidate = candidate.replace(target, "", 1)
    candidate = re.sub(r"[ \t]{2,}", " ", candidate)
    candidate = re.sub(r"\n{3,}", "\n\n", candidate)
    candidate = re.sub(r"(?m)^[ \t]+", "", candidate)
    return candidate.strip()


def topk_low_value_removal_allowed(sentence: str, row: dict | None = None, *, deps: TopkRouteDeps) -> bool:
    """Last-resort top-k pruning for generic high-predictability sentences."""
    value = str(sentence or "").strip()
    if len(value.split()) < 10:
        return False
    if re.search(r"\b\d{4}\b|https?://|www\.|\[[^\]]+\]|\([^)]*\d{4}[^)]*\)", value):
        return False
    top10 = float((row or {}).get("top10_ratio") or 0.0)
    if top10 < deps.float_env("DRAFTPROOF_TOPK_ROUTE_REMOVAL_MIN_TOP10", 0.66):
        return False
    lower = value.lower()
    generic_route = bool(re.search(
        r"^(?:this|these|it|another|one of|in addition|at the same time|despite|however|overall|taken together|critics argue|many people|the country)\b",
        value,
        re.I,
    ))
    generic_phrase = any(
        phrase in lower
        for phrase in (
            "in many ways",
            "significant influence",
            "important role",
            "major role",
            "complex and influential",
            "different languages and traditions",
            "not equally shared",
            "side by side",
        )
    )
    return generic_route or generic_phrase


def topk_route_optimizer_candidates(
    text: str,
    report_dict: dict | None,
    *,
    limit: int | None = None,
    deps: TopkRouteDeps,
) -> list[tuple[str, str, dict]]:
    repair_map = topk_repair_map(text, report_dict, limit=limit, deps=deps)
    topk_value = float(repair_map.get("topk_calibrated_risk") or 0.0)
    if topk_value < deps.safe_topk_calibrated_limit():
        return []
    expanded_limit = int(deps.float_env(
        "DRAFTPROOF_TOPK_ROUTE_EXPANDED_SENTENCES",
        max(float(repair_map.get("limit") or 1), min(48.0, float(len(deps.split_sentences(text)) or 1))),
    ))
    expanded_map = topk_repair_map(text, report_dict, limit=expanded_limit, deps=deps)
    targets = repair_map.get("targets") or []
    candidate_rows: list[tuple[str, str, dict]] = []
    batches = [(targets, 0.35, "small"), (targets, 0.65, "medium"), (targets, 1.0, "full")]
    expanded_targets = [
        row for row in (expanded_map.get("targets") or [])
        if float(row.get("top10_ratio") or 0.0) >= deps.float_env("DRAFTPROOF_TOPK_ROUTE_EXPANDED_MIN_TOP10", 0.45)
    ]
    if len(expanded_targets) > len(targets):
        batches.append((expanded_targets, 1.0, "expanded"))
    for batch_targets, fraction, label in batches:
        take = max(1, min(len(batch_targets), math.ceil(len(batch_targets) * fraction)))
        replacements: dict[str, str] = {}
        operations: list[dict] = []
        for target in batch_targets[:take]:
            sentence = str(target.get("sentence") or "")
            if not sentence:
                continue
            replacement, ops = deterministic_topk_route_sentence(sentence)
            if ops and replacement != sentence:
                replacements[sentence] = replacement
                operations.append({
                    "sentence_id": target.get("sentence_id"),
                    "sentence_index": target.get("sentence_index"),
                    "operations": ops,
                    "top10_ratio": target.get("top10_ratio"),
                })
        candidate = splice_sentences_by_text(text, replacements)
        if candidate != text and operations:
            candidate_rows.append((
                f"topk_route_optimizer_{label}",
                candidate,
                {
                    "topk_route_optimizer": True,
                    "stage": "deterministic_route_edits",
                    "target_count": take,
                    "applied_count": len(operations),
                    "operations": operations,
                },
            ))
    removable_targets = [
        row for row in expanded_targets
        if topk_low_value_removal_allowed(str(row.get("sentence") or ""), row, deps=deps)
    ]
    max_remove = max(0, int(deps.float_env("DRAFTPROOF_TOPK_ROUTE_REMOVAL_MAX_SENTENCES", 3.0)))
    if max_remove > 0 and removable_targets:
        for take in range(1, min(max_remove, len(removable_targets)) + 1):
            removed_sentences = [str(row.get("sentence") or "") for row in removable_targets[:take]]
            candidate = remove_sentences_by_text(text, removed_sentences)
            if candidate and candidate != text:
                candidate_rows.append((
                    f"topk_route_optimizer_remove_low_value_{take}",
                    candidate,
                    {
                        "topk_route_optimizer": True,
                        "stage": "low_value_topk_sentence_removal",
                        "removed_count": take,
                        "removed_sentence_ids": [row.get("sentence_id") for row in removable_targets[:take]],
                        "removed_top10": [row.get("top10_ratio") for row in removable_targets[:take]],
                    },
                ))
    return candidate_rows


def topk_masked_route_prompt(
    text: str,
    report_dict: dict | None,
    *,
    candidate_count: int = 2,
    deps: TopkRouteDeps,
) -> str:
    repair_map = topk_repair_map(text, report_dict, deps=deps)
    payload = {
        "topk_pattern": repair_map.get("topk_pattern"),
        "target_sentence_ids": repair_map.get("target_sentence_ids"),
        "targets": repair_map.get("targets"),
        "protected_anchors": deps.protected_anchor_brief_for_prompt(text),
    }
    return (
        "DraftProof TOPK_ROUTE_OPTIMIZER.\n"
        "Repair only the listed high top-k sentences. Return JSON patches, not a full document.\n\n"
        "Goal:\n"
        "- reduce top_10_ratio / topk_pattern by changing sentence routes\n"
        "- preserve facts, anchors, dates, names, and core claims\n"
        "- use detector-first texture: mild roughness, fragments, clause movement, less predictable openings\n"
        "- break repeated claim -> explanation -> implication routes\n"
        "- mix rhythm: short sentence, longer explanation, small follow-up where useful\n\n"
        "Preferred operations:\n"
        "- replace predictable openings with concrete route changes\n"
        "- remove generic connectors instead of replacing them with polished connectors\n"
        "- split over-smooth sentences when meaning remains intact\n"
        "- move a clause to the front only when it lowers the predictable opening\n\n"
        "Forbidden:\n"
        "- no new facts, citations, statistics, examples, or personal experience\n"
        "- no full-document rewrite\n"
        "- no anchor mutation\n"
        "- no smoother academic polish\n\n"
        "- do not use Furthermore, Moreover, Additionally, In conclusion, It is important to note, This demonstrates, This underscores, plays a crucial role, significant impact\n\n"
        "Return valid JSON only:\n"
        "{\n"
        '  "candidates": [\n'
        '    {"patches": [{"sentence_id": "s001", "original_sentence": "...", "replacement_sentence": "..."}]}\n'
        "  ]\n"
        "}\n\n"
        f"Return exactly {max(1, int(candidate_count or 1))} candidates.\n"
        f"REPAIR MAP:\n{json.dumps(payload, ensure_ascii=False)[:12000]}"
    )


def topk_safe_band_snapshot_prompt(text: str, report_dict: dict | None, *, deps: TopkRouteDeps) -> str:
    ai_components = (((report_dict or {}).get("ai_risk_badge") or {}).get("ai_components") or {})
    contribution = deps.contribution_scores(report_dict)
    source_words = deps.text_word_count(text)
    min_words = max(260, int(source_words * 0.72))
    max_words = max(min_words + 80, int(source_words * 1.08))
    if source_words <= 700:
        block_guidance = "5 to 8 normal paragraphs"
    elif source_words <= 1800:
        block_guidance = "7 to 12 normal paragraphs"
    else:
        block_guidance = "section-preserving paragraphs; do not collapse the document into a short summary"
    current_signals = json.dumps({
        "raw_topk": ai_components.get("topk_pattern_raw", ai_components.get("topk_pattern")),
        "topk_calibrated_risk": ai_components.get("topk_calibrated_risk"),
        "ai_transformation": contribution.get("ai_transformation"),
        "human_contribution": contribution.get("human"),
        "source_word_count": source_words,
        "target_word_range": [min_words, max_words],
    }, ensure_ascii=False)
    return (
        "DraftProof TOPK_SAFE_BAND_REBUILD.\n"
        "The current document is saturated on GPT-2 Top-k token-route predictability. "
        "A normal smooth essay will fail, but compressed fragment prose will also fail. "
        "Rebuild the submission as ordinary paragraphs with less predictable sentence routes.\n\n"
        "Objective:\n"
        "- lower calibrated Top-k risk below 25 after scanning\n"
        "- preserve the topic and main claims\n"
        "- reduce smooth textbook cadence\n"
        "- do not copy source sentence structure\n\n"
        "Current signals:\n"
        f"{current_signals}\n\n"
        "Required texture:\n"
        f"- write {block_guidance}\n"
        f"- target {min_words} to {max_words} words unless preserving meaning needs a little more\n"
        "- keep full-document coverage; do not turn the submission into a short country profile or bullet-like digest\n"
        "- use concrete anchors where they naturally belong: date, place, named event, law, institution, company, person, technology, object, or movement\n"
        "- include at least 10 prose sentences of 8-18 words so the scanner has eligible prose\n"
        "- use uneven rhythm: short sentence, longer sentence, plain follow-up where useful\n"
        "- use direct plain turns where useful: That sounds tidy. It is not. The cleaner story leaves parts out.\n"
        "- do not repeat the same sentence opening or subject-verb route\n"
        "- use stable public facts already implied by the source topic; no invented statistics\n"
        "- avoid generic openings: The topic is, It is important, One of the, In conclusion, Overall\n"
        "- avoid smooth claim -> explanation -> implication paragraphs\n"
        "- avoid colon-heavy compressed fragments and promotional list tone\n"
        "- avoid poetic metaphor, quirky phrasing, slang, and dramatic compressed phrases\n"
        "- use plain concrete wording even when varying sentence routes\n"
        "- keep sentence spacing clean; every sentence must have a space after punctuation\n\n"
        "Sentence texture examples:\n"
        "Original: The organisation has a strong public influence.\n"
        "Route: Its influence appears through several channels. Some are visible, others are easier to miss.\n"
        "Original: Despite its success, the project also faces serious issues.\n"
        "Route: The success is real. That sounds tidy, but it leaves out cost, access, and trust.\n"
        "Original: Technology and innovation continue to shape the future.\n"
        "Route: Technology still shapes the work. Not every change is simple or fair.\n\n"
        "Return only the rewritten prose. No explanation.\n\n"
        "SOURCE DOCUMENT:\n"
        f"{str(text or '')[:12000]}"
    )


def topk_plain_spoken_snapshot_prompt(text: str, report_dict: dict | None, *, deps: TopkRouteDeps) -> str:
    """Fallback rebuild when ordinary Top-k snapshot remains saturated."""
    ai_components = (((report_dict or {}).get("ai_risk_badge") or {}).get("ai_components") or {})
    source_words = deps.text_word_count(text)
    min_words = max(240, int(source_words * 0.70))
    max_words = max(min_words + 80, int(source_words * 1.05))
    current_signals = json.dumps({
        "raw_topk": ai_components.get("topk_pattern_raw", ai_components.get("topk_pattern")),
        "topk_calibrated_risk": ai_components.get("topk_calibrated_risk"),
        "source_word_count": source_words,
        "target_word_range": [min_words, max_words],
    }, ensure_ascii=False)
    return (
        "DraftProof PLAIN_SPOKEN_TOPK_REBUILD.\n"
        "The first Top-k rebuild may still be too predictable. Rebuild the document again, "
        "but do it in plain, ordinary prose. Do not use poetic, quirky, metaphorical, or slogan-like phrasing.\n\n"
        "Goal:\n"
        "- lower calibrated Top-k risk by changing sentence routes and paragraph order\n"
        "- preserve the same topic, core claims, names, dates, and examples\n"
        "- keep the writing readable and natural, not compressed or decorative\n\n"
        "Current signals:\n"
        f"{current_signals}\n\n"
        "How to write:\n"
        f"- target {min_words} to {max_words} words\n"
        "- use plain everyday nouns and verbs\n"
        "- use mixed sentence jobs: fact, limit, example, consequence, doubt, correction\n"
        "- make some paragraphs short and some fuller, but keep normal prose\n"
        "- narrow broad claims instead of praising or dramatizing them\n"
        "- use direct turns such as: The difficulty is, This is not always true, One limit is, The result is uneven\n"
        "- do not repeat the same opening route such as Its influence, The country, or This shows\n"
        "- keep at least 10 eligible prose sentences of 8-18 words\n\n"
        "Forbidden:\n"
        "- no metaphors such as route, cogs, strands, muscle, tapestry, shadow, ground zero, sharp corner\n"
        "- no quirky slang or colloquial shortcuts such as folks, cash, grit, stateside, hatched\n"
        "- no colon-heavy list fragments\n"
        "- no missing spaces after punctuation\n"
        "- no generic admiration list of strengths\n"
        "- no new statistics or invented citations\n\n"
        "Plain route examples:\n"
        "Original: The topic is influential in public life.\n"
        "Rewrite: Its influence reaches beyond one setting. The clearest channels depend on the specific case.\n"
        "Original: The topic has many strengths but also many challenges.\n"
        "Rewrite: The strengths are visible. That is not the full picture. Cost, access, and trust still matter.\n"
        "Original: Technology changes modern work.\n"
        "Rewrite: Technology affects daily routines. It changes how people communicate, decide, and check information.\n\n"
        "Return only the rewritten prose. No explanation.\n\n"
        "SOURCE DOCUMENT:\n"
        f"{str(text or '')[:12000]}"
    )


def topk_safe_band_sentence_patch_prompt(candidate_text: str, candidate_report: dict | None, *, deps: TopkRouteDeps) -> str:
    rows = []
    for row in (topk_repair_map(candidate_text, candidate_report, limit=14, deps=deps).get("targets") or []):
        sentence = str(row.get("sentence") or "").strip()
        if not sentence:
            continue
        rows.append({
            "sentence_id": row.get("sentence_id"),
            "sentence": sentence,
            "top10_ratio": row.get("top10_ratio"),
            "predictable_token_spans": row.get("predictable_token_spans") or [],
        })
    return (
        "DraftProof TOPK_SAFE_BAND_SENTENCE_PATCH.\n"
        "Patch these high Top-k sentences after the snapshot rebuild. Preserve meaning, but use lower-predictability routing.\n\n"
        "Allowed:\n"
        "- plain uneven sentence routes: one short sentence, one specific follow-up, one limited contrast\n"
        "- direct skeptical turns: That sounds tidy. It is not. The cleaner version misses something.\n"
        "- occasional plain fragment only when it reads naturally, not as a headline\n"
        "- tiny stable details only when directly implied by the sentence or public context\n"
        "- replace smooth textbook phrasing with less predictable but ordinary wording\n\n"
        "Forbidden:\n"
        "- generic essay polish\n"
        "- poetic metaphor, quirky phrasing, slang, or compressed headline style\n"
        "- colon-heavy fragments and dramatic list rhythm\n"
        "- new statistics or citations\n"
        "- changing the document topic\n\n"
        "Patch coverage:\n"
        "- each candidate must include one patch for every listed sentence unless a sentence cannot be found exactly\n"
        "- do not return a one-sentence patch when many high-risk sentences are listed\n"
        "- candidate 1 should use direct plain contrast; candidate 2 should use plain sentence splitting and clause movement\n\n"
        "Examples:\n"
        "Original: The topic has a strong influence.\n"
        "Replacement: The influence is visible in several places. The exact routes depend on the case.\n"
        "Original: Despite its success, the project also faces serious issues.\n"
        "Replacement: The strengths are clear. That is not the whole story. Cost, access, and trust remain problems.\n"
        "Original: The system was built on several important ideas.\n"
        "Replacement: Those ideas mattered from the start. People still argue over how they should work.\n\n"
        "Return valid JSON only:\n"
        '{"candidates":[{"patches":[{"original_sentence":"...","replacement_sentence":"..."}]}]}\n'
        "Return 2 candidates with different sentence routes. Each candidate should patch all listed sentences. Do not repeat the same openings across replacements.\n\n"
        f"SENTENCES:\n{json.dumps(rows, ensure_ascii=False)[:12000]}"
    )
