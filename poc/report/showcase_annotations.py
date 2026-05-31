"""Teaching annotations for the rewrite showcase.

Lays additive, explanatory notes on top of the original -> rewritten worked example so the user
LEARNS the grounding/anchoring technique and sees where to supply their own voice and real
specifics. It is a TEACHING layer: it explains the change, it does NOT modify the text to evade any
detector (that would be a humanizer, which DraftProof does not build). The user studies the
annotated example, then writes their own version.

Each change yields one or more (technique, why) notes plus a "your turn" prompt that makes explicit
the illustrative specifics are placeholders for the user's real ones.
"""
from __future__ import annotations

import re

_FIRST_PERSON = re.compile(
    r"\b(in my|when i\b|i have|i've|i saw|i watched|i keep|i notice|my students|my classroom|"
    r"in my experience|i taught|i lead|i grade)\b", re.I)
_NUMBER = re.compile(r"\b\d[\d,.%/–-]*\b")
_PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|of|for|and|the))*\s+[A-Z][a-zA-Z]+\b")
_HEDGE = re.compile(r"\b(many|some|often|generally|various|numerous|a lot of|things|stuff|people)\b", re.I)


def annotate_change(original: str, rewritten: str) -> dict:
    """Teaching note for one original -> rewritten change. Returns {techniques:[(label, why)], your_turn}."""
    o, r = (original or "").strip(), (rewritten or "").strip()
    ol = o.lower()
    techniques: list[tuple[str, str]] = []

    new_numbers = [n for n in _NUMBER.findall(r) if n not in o][:3]
    new_proper = [e for e in _PROPER.findall(r) if e.lower() not in ol][:3]
    if new_numbers or new_proper:
        anchors = ", ".join(f"“{a}”" for a in (new_numbers + new_proper)[:3])
        techniques.append((
            "Grounded with a concrete anchor",
            f"The rewrite drops in a specific detail ({anchors}) where the original stayed general — "
            f"a concrete particular is what makes a claim read as real, not generic."))

    if _FIRST_PERSON.search(r) and not _FIRST_PERSON.search(ol):
        techniques.append((
            "Anchored in lived experience",
            "It reframes the abstract claim as something witnessed first-hand. First-hand specifics "
            "are hard to fake and read as genuinely yours."))

    if _HEDGE.search(ol) and not _HEDGE.search(r.lower()):
        techniques.append((
            "Cut the vague hedge",
            "Words like ‘many/often/things’ signal a claim with no anchor. The rewrite names the "
            "specific instead."))

    if not techniques:
        techniques.append((
            "Made the claim more specific",
            "The rewrite trades a broad statement for a more particular one — narrower, more owned."))

    return {
        "techniques": techniques,
        "your_turn": (
            "These specifics are illustrative — they show the SHAPE of a good anchor. Swap them for "
            "YOUR real ones: your example, your number, your moment. That is the part that makes the "
            "writing genuinely yours (and the part no tool can do for you)."),
    }


def annotate_comparison(pairs) -> list[dict]:
    """pairs: iterable of (original, rewritten[, changed]). Annotates only the changed ones."""
    out = []
    for p in pairs:
        original = p.get("original") if isinstance(p, dict) else p[0]
        rewritten = p.get("rewritten") if isinstance(p, dict) else p[1]
        changed = (p.get("changed") if isinstance(p, dict) else (p[2] if len(p) > 2 else True))
        if str(changed).lower() in ("false", "0", "none", ""):
            continue
        if not original or not rewritten or original.strip() == rewritten.strip():
            continue
        out.append({"original": original, "rewritten": rewritten, **annotate_change(original, rewritten)})
    return out
