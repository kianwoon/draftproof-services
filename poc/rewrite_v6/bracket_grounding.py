"""Bracket-grounding -- the last-stage treatment for generic (un-grounded) sentences.

For each generic sentence the showcase selector finds, ask the model (qwen) for a rewrite, then a
deterministic GATE (`_is_genuine_improvement`) -- NOT qwen's self-report -- decides what ships:
  - replacement is genuinely better -> GREEN span ('improved')  the replacement is used
  - otherwise                        -> AMBER span ('kept')      the ORIGINAL is kept (qwen text dropped)

Green requires ALL of: (1) NOT fabricated -- introduces no new number and no concreteness/anchor signal
the original lacked (qwen has no source, so any added specific is invented); (2) grammar/quality clean;
(3) not worse on a before/after `scan_text` A/B (finding_count + sentence-shape risk). The generator
(qwen) and the checker are SEPARATE entities, and the A/B is a symmetric regression check (reject-if-
worse), NOT the scanner's grounding oracle -- so green is not the self-congratulatory circular gate that
the prior `_is_grounded` version was. The (original, replacement, decision, scan delta) per candidate is
captured into `diag` for audit.

Consequence (honest): qwen cannot raise grounding without inventing specifics, which the fabrication
check rejects -- so green mostly reflects faithful grammar/structural gains; real grounding still comes
from the USER editing the amber sentences. Content-agnostic selection (reuses the scanner's
structural-concreteness check via _generic_candidates). MUTATES rewritten_text -- the caller MUST
re-scan the shipped text after this stage so the report's scores describe the bytes the user receives.
Never raises; on disable / no candidates / any failure the text is returned unchanged.

allow-hardcode: the `rules` below are the model coaching PROMPT (human-reviewed guidance), not a
detect/allow/scoring word-list in code logic.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


def bracket_grounding_enabled() -> bool:
    """Feature flag. Code default OFF; production enables it via the worker entrypoint. Set
    DRAFTPROOF_V6_BRACKET_GROUNDING=1 to enable, =0 to disable."""
    return os.environ.get("DRAFTPROOF_V6_BRACKET_GROUNDING", "0").strip().lower() in {"1", "true", "yes", "on"}


def _max_sentences() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_BRACKET_MAX_SENTENCES", "").strip()
    try:
        v = int(raw)
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return 8


def _max_tokens() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_BRACKET_MAX_TOKENS", "").strip()
    try:
        v = int(raw)
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return 8000


# allow-hardcode: _SYSTEM and the `rules` below are the model coaching PROMPT (human-reviewed
# guidance), not a detect/allow/scoring word-list in code logic. The few illustrative filler words are
# examples shown to the model, never matched against the input.
# IMPORTANT: this prompt MUST stay aligned with `_is_genuine_improvement` -- it asks ONLY for faithful
# improvement with NO new facts, because the gate rejects any invented number/name/specific. Asking
# qwen to "add a concrete anchor" (the old prompt) produced exactly what the gate throws away -> all amber.
_SYSTEM = (
    "You are a precise line editor. You rewrite over-generic sentences to read more like natural human "
    "writing, WITHOUT adding any new fact, number, name, date, place, example, or event -- you have no "
    "source, so inventing specifics is forbidden. Lower AI-pattern risk only through FAITHFUL structure: "
    "make the sentence shorter and more direct, cut hedging and filler, start with a concrete subject "
    "(not 'It/This/They/There' + a modal), and avoid long comma/semicolon clause chains. Keep the meaning, "
    "stance, and every existing fact. If you cannot improve a sentence without inventing something, leave "
    "it out. Return only valid JSON."
)


def _build_prompt(batch: list[tuple[int, str]], feedback: dict[int, str] | None, n: int) -> str:
    # allow-hardcode: _SYSTEM and these `rules` are model coaching PROMPT guidance (human-reviewed), not
    # a detect/allow/scoring word-list in code logic. The illustrative hedge/opener words are examples
    # shown to the model, never matched against the input.
    sentences: list[dict[str, Any]] = []
    for i, s in batch:
        row: dict[str, Any] = {"i": i, "text": s}
        if feedback and feedback.get(i):
            row["previous_attempt_feedback"] = feedback[i]
        sentences.append(row)
    payload = {
        "task": "tighten_without_inventing",
        "rules": [
            "Do NOT add any new fact, number, name, date, place, example, statistic, or event. You have "
            "no source; inventing specifics is forbidden and will be rejected.",
            "Lower AI-pattern risk by FAITHFUL wording changes ONLY: cut hedging/qualifier words (e.g. "
            "may, might, could, often, usually, generally, seems, appears, relatively, somewhat); replace "
            "a predictable opener like 'It/This/They/There' + a modal with the concrete subject; trim "
            "redundant filler. Keep it a COMPLETE sentence with its subject and verb.",
            "Keep the meaning and EVERY detail of the original -- do not drop the subject, articles, "
            "clauses, or specifics, and do NOT write telegraphic / note / headline style. Preserve roughly "
            "the same length (a little shorter is fine; do not delete content to look shorter). Return "
            "clean, natural, complete sentences.",
            f"Return up to {n} distinct alternatives per sentence in 'alternatives'. If you cannot improve "
            "a sentence without inventing a specific, return an empty 'alternatives' list -- the original is kept.",
            "If 'previous_attempt_feedback' is present, your earlier rewrite did NOT lower the score -- "
            "apply the specific fix it names with a bolder (still faithful) structural change.",
        ],
        "sentences": sentences,
        "output_schema": {"results": [{"i": 0, "alternatives": ["faithful tighter rewrite with NO new facts", "..."]}]},
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False)


# Risk tolerance for the before/after A/B (mirrors write.py::_has_meaningful_movement's `>= -2.0`):
# a replacement may raise mean sentence-shape risk by at most this much and still count as "not worse".
_AB_RISK_TOLERANCE = 2.0
# Fidelity floor: a green rewrite must keep the meaning (content overlap) and not collapse into a
# telegraphic fragment. Prevents gaming the brevity-rewarding structural signal by deleting content.
_CONTENT_COVERAGE_FLOOR = 0.72
_WORD_RETENTION_FLOOR = 0.75


def _min_risk_improvement() -> float:
    """Green must be a GENUINE improvement, not merely not-worse: a finding removed OR at least this
    much drop in mean sentence-shape risk. Faithful-but-flat rewords (synonym swaps) -> amber, so
    green reliably means 'better'. Env-tunable for calibration."""
    raw = os.environ.get("DRAFTPROOF_V6_BRACKET_MIN_RISK_IMPROVEMENT", "").strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def _variants() -> int:
    """Number of faithful alternatives requested per candidate (best-of-N). Env-tunable."""
    raw = os.environ.get("DRAFTPROOF_V6_BRACKET_VARIANTS", "").strip()
    try:
        return max(1, min(8, int(raw)))
    except (TypeError, ValueError):
        return 5


def _feedback_rounds() -> int:
    """Extra amber->green retry rounds with targeted feedback (0 = single pass). Env-tunable."""
    raw = os.environ.get("DRAFTPROOF_V6_BRACKET_FEEDBACK_ROUNDS", "").strip()
    try:
        return max(0, min(5, int(raw)))
    except (TypeError, ValueError):
        return 3


def _lever_feedback(sentence: str, attempted: bool) -> str:
    """Targeted, content-agnostic hint built from the measured gap (reuses structural_metrics._HEDGES).
    Names the faithful levers the rewrite still needs to pull -- never suggests adding facts."""
    import re
    hints: list[str] = []
    words = sentence.split()
    if len(words) > 18:
        hints.append(f"shorten from {len(words)} words to under 18")
    try:
        from .structural_metrics import _HEDGES
        hedges = sorted({w for w in re.findall(r"[a-z']+", sentence.casefold()) if w in _HEDGES})
    except Exception:
        hedges = []
    if hedges:
        hints.append("remove hedging words: " + ", ".join(hedges))
    if re.match(r"\s*(it|this|they|there)\b", sentence.lower()):
        hints.append("replace the predictable opener; start with a concrete subject")
    if attempted:
        hints.append("your previous rewrite did not lower the risk -- make a bolder faithful structural change (no new facts)")
    return "; ".join(hints) or "make a bolder faithful structural change (shorten, de-hedge, vary the opening) -- add no new facts"


# allow-hardcode: a CLOSED linguistic class (cardinal/ordinal number words), not a domain/detect list.
# Used to catch INVENTED numbers written as words ("forty percent") that the digit regex misses.
_WORD_NUMBERS = frozenset((
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
    "million billion dozen first second third fourth fifth sixth seventh eighth ninth tenth"
).split())


def _specific_tokens(sentence: str) -> set[str]:
    """The verifiable SPECIFICS in a sentence: digit numbers/percentages, number-words, and named
    entities / acronyms (capitalised mid-sentence tokens). These cannot be produced by faithfully
    rephrasing existing generic words -- so a specific present in the replacement but NOT the original
    is INVENTED (fabrication). Deliberately ignores generic STRUCTURE (first-person 'in my X',
    'when/such as/if' anchors, synonym swaps), which faithful rephrasing legitimately introduces and
    which previously caused false 'fabricated' rejections."""
    import re
    s = str(sentence or "").strip()
    toks: set[str] = set(re.findall(r"\d[\d,\.]*%?", s))                       # 47, 12%, 2020
    toks |= {w for w in re.findall(r"[a-z]+", s.lower()) if w in _WORD_NUMBERS}  # forty, ten, third
    caps = re.findall(r"\b[A-Z][A-Za-z]{2,}\b", s)                            # Chicago, Discord, STEM
    if caps:  # drop the sentence's first word (capitalised by position, not an entity)
        first = re.findall(r"[A-Za-z]+", s)[:1]
        if first and caps and caps[0] == first[0]:
            caps = caps[1:]
    toks |= set(caps)
    return toks


def _is_genuine_improvement(original: str, replacement: str) -> tuple[bool, dict[str, Any]]:
    """Decide GREEN (use replacement) vs AMBER (keep original) by a deterministic, generator-independent
    comparison -- never qwen's self-report. Returns (accept, audit_detail). Green requires ALL of:
      (1) NOT fabricated: introduces no new SPECIFIC token (number, number-word, named entity/acronym)
          vs the original. Generic reframes ('in my X', 'when/such as') and synonym swaps are NOT
          fabrication (token-based check -- avoids the old anchor-pattern false positives);
      (2) grammar/quality clean: no hard (non-advisory) integrity blocker, no fragment/trace sentence;
      (3) not worse on the structural A/B: scan_text finding_count and mean_sentence_shape_risk do not
          regress vs the original -- the same scores write.py::_has_meaningful_movement trusts.
    Fails CLOSED to amber when a check cannot run (never ship an unverified replacement)."""
    detail: dict[str, Any] = {"original": original, "replacement": replacement}
    repl = (replacement or "").strip()
    if not repl or repl == (original or "").strip():
        detail["decision"] = "no_change"
        return False, detail

    # (1) fabrication: a verifiable SPECIFIC (number, number-word, named entity/acronym) present in the
    # replacement but NOT the original is invented. Generic reframes / synonym swaps are NOT fabrication.
    added_specifics = sorted(_specific_tokens(repl) - _specific_tokens(original))
    fabricated = bool(added_specifics)
    detail.update({"added_specifics": added_specifics, "fabricated": fabricated})
    if fabricated:
        detail["decision"] = "fabricated"
        return False, detail

    # (2) grammar / quality
    try:
        from .integrity_guard import ADVISORY_BLOCKERS, candidate_integrity_blockers
        from .prose_quality import has_fragment_or_trace_sentences
        hard = [b for b in candidate_integrity_blockers(repl) if b not in ADVISORY_BLOCKERS]
        grammar_ok = not hard and not has_fragment_or_trace_sentences(repl)
        detail["grammar_blockers"] = hard
    except Exception:
        grammar_ok = True  # selection already passed; import failure must not block every candidate
    detail["grammar_ok"] = grammar_ok
    if not grammar_ok:
        detail["decision"] = "grammar_regression"
        return False, detail

    # (2b) FIDELITY floor: reject telegraphic / content-dropping rewrites. The structural-risk signal
    # rewards brevity, so "shorten" can be GAMED by deleting the subject, articles, and details
    # (note/headline style). Reuse the topk-repair content-coverage check + a word-retention band so a
    # green keeps the meaning and reads as a complete sentence, not a fragment.
    try:
        from .highlight_topk_repair import _content_coverage
        cov = float(_content_coverage(original, repl))
    except Exception:
        cov = 1.0
    ow, rw = len(original.split()), len(repl.split())
    detail["content_coverage"] = round(cov, 3)
    detail["word_retention"] = round(rw / ow, 3) if ow else 1.0
    if cov < _CONTENT_COVERAGE_FLOOR or (ow and rw < _WORD_RETENTION_FLOOR * ow):
        detail["decision"] = "telegraphic_or_content_loss"
        return False, detail

    # (3) before/after A/B on structural scores -- fail CLOSED if the scan can't run
    try:
        from .scan import scan_text
        before = scan_text(original).scores
        after = scan_text(repl).scores
        finding_drop = before.get("finding_count", 0) - after.get("finding_count", 0)
        risk_drop = before.get("mean_sentence_shape_risk", 0.0) - after.get("mean_sentence_shape_risk", 0.0)
        detail["scan_before"] = {"finding_count": before.get("finding_count"),
                                 "risk": round(before.get("mean_sentence_shape_risk", 0.0), 3)}
        detail["scan_after"] = {"finding_count": after.get("finding_count"),
                                "risk": round(after.get("mean_sentence_shape_risk", 0.0), 3)}
        detail["finding_drop"] = finding_drop
        detail["risk_drop"] = round(risk_drop, 3)
    except Exception:
        detail["decision"] = "scan_unavailable"
        return False, detail
    if finding_drop < 0 or risk_drop < -_AB_RISK_TOLERANCE:
        detail["decision"] = "score_regression"
        return False, detail
    # Green must be a GENUINE improvement, not merely not-worse: a finding removed or a real drop in
    # sentence-shape risk. Faithful-but-flat rewords (synonym swaps, neutral restructures) -> amber.
    if not (finding_drop >= 1 or risk_drop >= _min_risk_improvement()):
        detail["decision"] = "no_improvement"
        return False, detail

    detail["decision"] = "accepted"
    return True, detail


def apply_bracket_grounding(
    text: str,
    *,
    gateway: Any,
    fallback_gateway: Any = None,
    cancellation_check: Callable[[], None] | None = None,
    diag: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Ground the generic sentences. Returns (clean_text, spans) -- NO literal brackets in the text.
    spans = [{"start","end","kind"}] over clean_text, kind 'improved' (qwen generated a better version,
    rendered GREEN) or 'kept' (qwen could not improve, original kept, rendered AMBER).
    (text, []) on disable / no candidates / failure."""
    original = str(text or "")
    if not bracket_grounding_enabled() or not original.strip():
        return original, []
    if cancellation_check:
        cancellation_check()
    try:
        from .predictability_showcase import _generic_candidates
        from .json_io import parse_json

        candidates = _generic_candidates(original, _max_sentences())
        if not candidates:
            return original, []

        n_variants = _variants()

        def _request_alternatives(batch: list[tuple[int, str]], feedback: dict[int, str]) -> dict[int, list[str]]:
            """One qwen call -> {i: [faithful alternative, ...]} (best-of-N). {} on failure -> all amber."""
            try:
                response = gateway.chat(
                    _build_prompt(batch, feedback, n_variants),
                    system=_SYSTEM,
                    temperature=0.5,
                    top_p=0.9,
                    max_tokens=_max_tokens(),
                    response_format={"type": "json_object"},
                    app_label="BracketGrounding",
                )
                raw = getattr(response, "raw_content", "") or getattr(response, "content", "") or ""
                data = parse_json(raw)
            except Exception:
                logger.warning("bracket_grounding qwen call failed; falling back to amber-kept spans", exc_info=True)
                return {}
            out: dict[int, list[str]] = {}
            for r in ((data.get("results") or []) if isinstance(data, dict) else []):
                if not isinstance(r, dict) or "i" not in r:
                    continue
                alts = r.get("alternatives")
                if isinstance(alts, list):
                    vals = [str(a).strip() for a in alts if str(a or "").strip()]
                else:  # tolerate the older single-string 'improved' shape
                    one = str(r.get("improved") or "").strip()
                    vals = [one] if one else []
                seen: set[str] = set()
                uniq: list[str] = []
                for v in vals:
                    if v not in seen:
                        seen.add(v); uniq.append(v)
                out[r["i"]] = uniq
            return out

        def _best_passing(sentence: str, alts: list[str]) -> tuple[str, dict[str, Any], bool]:
            """Gate each alternative; return (chosen_text, detail, accepted) for the best PASSING one
            (ranked by finding/risk drop), else (original, last_detail, False)."""
            best: tuple[tuple[float, float], str, dict[str, Any]] | None = None
            attempts: list[dict[str, Any]] = []
            for alt in alts:
                accept, detail = _is_genuine_improvement(sentence, alt)
                attempts.append(detail)
                if accept:
                    rank = (float(detail.get("finding_drop", 0) or 0), float(detail.get("risk_drop", 0.0) or 0.0))
                    if best is None or rank > best[0]:
                        best = (rank, alt, detail)
            if best is not None:
                return best[1], best[2], True
            last = attempts[-1] if attempts else {"decision": "no_change", "original": sentence, "replacement": ""}
            return sentence, last, False

        # best-of-N generation with an amber->green feedback retry loop (mirrors writer_feedback +
        # pipeline.py:370-381): regenerate the still-amber candidates with the measured gap as feedback.
        decisions: dict[int, tuple[bool, str, dict[str, Any]]] = {}
        pending = [(i, s) for i, s in enumerate(candidates) if original.find(s) >= 0]
        feedback: dict[int, str] = {}
        for round_idx in range(_feedback_rounds() + 1):
            if not pending:
                break
            alts_by_i = _request_alternatives(pending, feedback)
            still: list[tuple[int, str]] = []
            for i, sentence in pending:
                chosen, detail, accepted = _best_passing(sentence, alts_by_i.get(i, []))
                detail["round"] = round_idx
                decisions[i] = (accepted, chosen, detail)
                if not accepted:
                    still.append((i, sentence))
                    feedback[i] = _lever_feedback(sentence, attempted=bool(alts_by_i.get(i)))
            pending = still

        # locate each candidate; the deterministic gate (NOT qwen's self-report) decided green vs amber.
        located = []
        audit: list[dict[str, Any]] = []
        for i, sentence in enumerate(candidates):
            idx = original.find(sentence)
            if idx < 0:
                continue  # not locatable verbatim (e.g. spans a line break) -> leave untouched
            accepted, chosen, detail = decisions.get(i, (False, sentence, {"decision": "not_located", "original": sentence}))
            audit.append(detail)
            if accepted:
                located.append((idx, idx + len(sentence), chosen, "improved"))
            else:
                located.append((idx, idx + len(sentence), sentence, "kept"))  # keep ORIGINAL, drop qwen text
        located.sort()
        if diag is not None:
            diag["candidates"] = audit
            diag["variants"] = n_variants
            diag["feedback_rounds"] = _feedback_rounds()

        # rebuild CLEAN text, tracking each replacement's offset span in the new text
        out: list[str] = []
        spans: list[dict[str, Any]] = []
        cursor = 0
        pos = 0
        for start_o, end_o, replacement, kind in located:
            if start_o < cursor:
                continue  # overlapping / duplicate match -> skip
            gap = original[cursor:start_o]
            out.append(gap); pos += len(gap)
            span_start = pos
            out.append(replacement); pos += len(replacement)
            spans.append({"start": span_start, "end": pos, "kind": kind})
            cursor = end_o
        out.append(original[cursor:])
        new_text = "".join(out)
        logger.info("bracket_grounding: candidates=%d spans=%d", len(candidates), len(spans))
        return new_text, spans
    except Exception:
        logger.warning("bracket_grounding failed", exc_info=True)
        return original, []
