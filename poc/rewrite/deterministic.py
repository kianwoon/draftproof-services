"""Deterministic rewrite engine — pattern-level fixes without LLM.

Replaces known AI-signature patterns with less formulaic alternatives.
Zero risk of introducing new AI patterns since no LLM is involved.

Runs BEFORE the LLM pass in the rewrite pipeline. Findings that can be
fixed deterministically are removed from the LLM queue.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DeterministicFix:
    """Record of a single deterministic replacement."""
    original: str
    replacement: str
    finding_type: str
    subtype: str
    sentence_idx: int


@dataclass
class DeterministicResult:
    """Result of deterministic rewrite pass."""
    text: str
    fixes: List[DeterministicFix] = field(default_factory=list)
    findings_addressed: List[str] = field(default_factory=list)


# ── Replacement lookup tables ────────────────────────────────────────
# Maps flagged phrase → list of alternatives (chosen at random).
# These are curated to sound human but NOT sound AI-generated.

GENERIC_PHRASE_REPLACEMENTS: Dict[str, List[str]] = {
    # Academic filler — DELETE by default (empty string = remove entirely)
    "it is worth mentioning": [""],
    "it should be noted that": [""],
    "in recent years": ["lately", "over the past few years"],
    "has gained significant attention": ["has drawn interest", "is getting noticed"],
    "has been widely studied": ["has seen considerable research", "researchers have explored this"],
    "increasingly important": ["more important", "hard to ignore"],
    "a growing body of research": ["several studies", "recent findings"],
    "the literature suggests": ["studies show", "findings suggest"],
    "further research is needed": ["more work remains", "this needs further study"],
    "sheds light on": ["clarifies", "reveals"],
    "paves the way for": ["makes possible", "enables"],
    "in conclusion": [""],
    "to summarize": [""],
    "in summary": [""],
    "it is important to note": [""],
    "it is important to note that": [""],

    # Transitions — DELETE. These are flagged as AI patterns. Removing them
    # is safer than replacing with "also"/"so" which are MORE predictable.
    "furthermore": [""],
    "in addition": [""],
    "moreover": [""],
    "additionally": [""],
    "consequently": [""],
    "therefore": [""],
    "thus": [""],
    "hence": [""],
    "accordingly": [""],
    "nevertheless": ["still", "yet"],
    "nonetheless": ["still", "yet"],
    "notwithstanding": [""],
    "as a result": [""],
    "in other words": [""],
    "for example": [""],
    "for instance": [""],
    "in particular": [""],
    "specifically": [""],
    "notably": [""],

    # Hedges — DELETE filler hedges
    "it can be argued": ["one could argue"],
    "it is suggested": [""],
    "it is widely believed": ["many believe"],
    "research suggests": ["findings suggest"],
    "studies indicate": ["findings suggest"],
    "it is generally accepted": ["most agree"],
    "there is evidence": ["evidence shows"],
    "it has been shown": ["studies show"],

    # Conclusions / wrap-ups
    "overall, this demonstrates": ["this shows", "this makes clear"],
    "overall, this shows": ["this shows", "this illustrates"],
    "this highlights the importance": ["this shows why", "this underlines why"],
    "this demonstrates the importance": ["this shows why", "this makes clear why"],
    "this underscores": ["this shows", "this reinforces"],
    "this emphasizes": ["this shows", "this drives home"],
    "this plays a crucial role": ["this matters greatly", "this is central to"],
    "plays a vital role": ["is essential for", "matters greatly for"],
    "plays an important role": ["matters for", "is key to", "contributes to"],
    "in today's world": ["today", "nowadays", "right now"],
    "in the modern world": ["today", "now", "in our time"],

    # Business / tech filler
    "has transformed the way": ["changed how", "reshaped", "altered how"],
    "in today's fast-paced world": ["today", "now"],
    "enhancing efficiency": ["working more efficiently", "improving speed"],
    "reducing costs": ["cutting costs", "saving money"],
    "better decision-making": ["smarter decisions", "better choices"],
    "unlock new opportunities": ["open new doors", "find new possibilities"],
    "drive innovation": ["push boundaries", "spark new ideas"],
    "significant impact": ["real impact", "major effect"],
    "rapidly evolving": ["changing fast", "shifting quickly"],
    "seamless experience": ["smooth experience", "frictionless process"],
    "leveraging cutting-edge technology": ["using the latest technology", "working with new tech"],
    "revolutionizing the industry": ["changing the industry", "reshaping the field"],
    "game-changing solution": ["real solution", "practical answer"],
    "paradigm shift": ["fundamental change", "major shift"],
    "disruptive innovation": ["breakthrough", "radical change"],
    "at the forefront of": ["leading", "at the cutting edge of"],
    "stay ahead of the curve": ["stay competitive", "keep up"],
    "next-generation": ["new", "latest", "modern"],
    "best-in-class": ["top", "leading", "excellent"],
}

# Subtype-specific sentence-level rewrite patterns
# These handle formulaic sentence structures, not just phrases
SUBTYPE_SENTENCE_FIXES = {
    "formulaic_conclusion": {
        "patterns": [
            (r"\bin conclusion,?\s*", ""),
            (r"\bto summarize,?\s*", "In short, "),
            (r"\bto sum up,?\s*", "Briefly, "),
            (r"\bin summary,?\s*", "In short, "),
            (r"\bultimately,?\s*", ""),
            (r"\boverall,?\s*", ""),
            (r"\bperhaps\s+(?:the\s+)?most\s+\w+\s+is\s+that", "What stands out most is that"),
        ],
    },
    "broad_education_claim": {
        "patterns": [
            (r"plays an important role in", "matters in"),
            (r"has transformed the way", "changed how"),
            (r"increasingly important in today'?s?", "important in today's"),
            (r"rapidly evolving", "changing"),
            (r"has gained significant attention", "has drawn interest"),
        ],
    },
}


def _pick_alternative(alternatives: List[str], rng=None) -> str:
    """Pick an alternative from the list."""
    import random
    if not alternatives:
        return ""
    if rng:
        return rng.choice(alternatives)
    return alternatives[hash(id(alternatives)) % len(alternatives)]


def fix_generic_phrases(text: str, phrases: List[str], rng=None) -> List[DeterministicFix]:
    """Identify generic phrase replacements (does NOT mutate text).

    Args:
        text: The document text (used for matching only).
        phrases: List of generic phrases detected by the scanner.
        rng: Optional random state for reproducibility.

    Returns:
        List of fixes to apply.
    """
    fixes = []

    for phrase in phrases:
        # Case-insensitive search
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            continue

        replacements = GENERIC_PHRASE_REPLACEMENTS.get(phrase.lower())
        if not replacements:
            continue

        alt = _pick_alternative(replacements, rng)

        if alt == "":
            fixes.append(DeterministicFix(
                original=match.group(),
                replacement="(removed)",
                finding_type="generic_phrase",
                subtype="filler",
                sentence_idx=-1,
            ))
        else:
            original = match.group()
            if original[0].isupper() and alt[0].islower():
                alt = alt[0].upper() + alt[1:]
            fixes.append(DeterministicFix(
                original=original,
                replacement=alt,
                finding_type="generic_phrase",
                subtype="filler",
                sentence_idx=-1,
            ))

    return fixes


def fix_subtype_patterns(text: str, subtype: str, rng=None) -> List[DeterministicFix]:
    """Identify deterministic fixes for a predictability subtype (does NOT mutate text).

    Args:
        text: The document text (used for matching only).
        subtype: The predictability subtype (e.g. 'formulaic_conclusion').
        rng: Optional random state for reproducibility.

    Returns:
        List of fixes to apply.
    """
    config = SUBTYPE_SENTENCE_FIXES.get(subtype)
    if not config:
        return []

    fixes = []

    for pattern, replacement in config["patterns"]:
        regex = re.compile(pattern, re.IGNORECASE)
        match = regex.search(text)
        if not match:
            continue

        original = match.group()

        # Skip if replacement already there
        if replacement and replacement.lower() in text[match.start()-5:match.end()+5].lower():
            continue

        fixes.append(DeterministicFix(
            original=original,
            replacement=replacement if replacement else "(removed)",
            finding_type="predictability",
            subtype=subtype,
            sentence_idx=-1,
        ))

    return fixes


def run_deterministic(
    text: str,
    findings: list,
    rng=None,
) -> DeterministicResult:
    """Run all deterministic fixes on the text.

    Processes findings in priority order:
    1. generic_phrase findings (exact phrase replacement)
    2. predictability findings by subtype (pattern-level fixes)

    Args:
        text: The document text.
        findings: List of DetectResult Finding objects from the detect scan.
        rng: Optional random state for reproducibility.

    Returns:
        DeterministicResult with the fixed text and list of changes.
    """
    all_fixes = []
    findings_addressed = []
    current = text

    # Phase 1: generic_phrase — these have exact flagged phrases
    generic_phrases = []
    for f in findings:
        if getattr(f, 'finding_type', '') == 'generic_phrase' and f.evidence:
            generic_phrases.append(f.evidence)

    if generic_phrases:
        before = current
        fixes = fix_generic_phrases(current, generic_phrases, rng)
        if fixes:
            # fix_generic_phrases mutates internally, but we re-apply
            # the replacements ourselves for correctness
            for fix in fixes:
                repl = "" if fix.replacement == "(removed)" else fix.replacement
                pattern = re.compile(re.escape(fix.original), re.IGNORECASE)
                current = pattern.sub(repl, current, count=1)
                # Clean up double spaces left by deletions
                current = re.sub(r'  +', ' ', current).strip()
                # Fix orphaned punctuation after deletion
                current = re.sub(r'\s+([.,;:])', r'\1', current)
            all_fixes.extend(fixes)
            findings_addressed.extend(["generic_phrase"] * len(fixes))

    # Phase 2: predictability subtypes — pattern-level fixes
    subtypes_seen = set()
    for f in findings:
        if getattr(f, 'finding_type', '') not in ('high_predictability', 'medium_predictability'):
            continue
        subtype = ""
        meta = getattr(f, 'metadata', None)
        if meta:
            subtype = meta.get('subtype', '')
        if not subtype or subtype in subtypes_seen or subtype == 'statistical_predictability':
            continue
        subtypes_seen.add(subtype)

        fixes = fix_subtype_patterns(current, subtype, rng)
        if fixes:
            for fix in fixes:
                repl = "" if fix.replacement == "(removed)" else fix.replacement
                pattern = re.compile(re.escape(fix.original), re.IGNORECASE)
                current = pattern.sub(repl, current, count=1)
                current = re.sub(r'  +', ' ', current).strip()
                current = re.sub(r'\s+([.,;:])', r'\1', current)
            all_fixes.extend(fixes)
            findings_addressed.extend([f"predictability/{subtype}"] * len(fixes))

    return DeterministicResult(
        text=current,
        fixes=all_fixes,
        findings_addressed=findings_addressed,
    )
