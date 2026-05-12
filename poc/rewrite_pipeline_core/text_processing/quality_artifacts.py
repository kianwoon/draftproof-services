from __future__ import annotations

import re


_SYNTHETIC_ANCHOR_RE = re.compile(
    r"\b(?:In my chair|During consultation|For example|In my notes|"
    r"During sectioning|Specifically|In my experience|During the cut|"
    r"In practice|For this task|During the practical work|In assessment|"
    r"When the task is underway|During feedback):",
    re.I,
)
_DANGLING_FRAGMENT_JOIN_RE = re.compile(
    r"\b(?:can|could|should|would|will|may|might|must|to|and|but|or|"
    r"while|because|if|before|after|adjust)\s+"
    r"(?:With only|People gain|A competent|The standard|Conclusion|"
    r"Introduction|This review|In this section|The issue)\b",
    re.I,
)
_GENERIC_PRAISE_TERMS_RE = re.compile(
    r"\b(?:strengths?|power(?:ful)?|influential|influence|global|worldwide|"
    r"remarkable|major|prestige|respected|successful|success|icons?|iconic|"
    r"innovation|innovative|entrepreneurship|thrives?|opportunit(?:y|ies)|"
    r"diversity|cultural|fame|famous|leadership|stability|"
    r"fresh starts?|creativity|self-expression)\b",
    re.I,
)
_GENERIC_PRAISE_PHRASES_RE = re.compile(
    r"\b(?:one of the (?:biggest|largest|most|strongest)|"
    r"(?:known|famous|recognized|respected) for|"
    r"(?:shaped|shapes) (?:the )?(?:modern )?world|"
    r"(?:economic|cultural|global) influence|(?:economic|military) power|"
    r"(?:global|major) (?:companies|icons|force|forces|alliances)|"
    r"attract(?:s|ing)? (?:people|participants|visitors|users) (?:from|worldwide)|"
    r"symbolize(?:s)? national identity|forces molding world history)\b",
    re.I,
)
_LOW_FRICTION_CONTRAST_RE = re.compile(
    r"\b(?:however|although|despite|yet|but|critics?|challenge|problem|"
    r"uneven|inequality|limitation|risk|concern|struggle|tension)\b",
    re.I,
)
_STYLIZED_TEXTURE_TERMS_RE = re.compile(
    r"\b(?:carv(?:e|ed|ing)|route|routes|cogs?|knobs?|sprint(?:ed|ing)?|"
    r"stunn(?:ed|ing)|mirrors?|sprawling|strands?|tapestr(?:y|ies)|"
    r"towering|churn(?:s|ed|ing)?|hatched|ground zero|sharp corner|"
    r"shadow|shadows|muscle|heavyweight|fuse|fuel(?:ed|s)?|"
    r"pull(?:s|ed|ing)? strings?|levers?|edge|echo(?:es|ed|ing)?|"
    r"stitched|quilt|thorny|cracks?|fault lines?|roadblocks?)\b",
    re.I,
)
_COLLOQUIAL_TEXTURE_TERMS_RE = re.compile(
    r"\b(?:folks?|cash|grit|hungry to|stateside|wins|fresh chances?|"
    r"biggest|loudest|kicks off|snapped from|staring down)\b",
    re.I,
)
_ABSTRACT_LIST_TERMS_RE = re.compile(
    r"\b(?:freedom|democracy|individual rights|opportunity|change|"
    r"influence|power|culture|innovation|diversity|stability|"
    r"identity|self-expression|values?|progress|responsibility)\b",
    re.I,
)
def _normalize_known_heading_boundaries(text: str) -> tuple[str, list[str]]:
    """Separate common document headings that were flattened into prose."""
    if not isinstance(text, str) or not text:
        return text, []
    repaired = text
    repairs: list[str] = []
    heading_re = (
        r"Introduction|Conclusion|References|Bibliography|Abstract|Background|"
        r"Discussion|Methodology|Method|Methods|Results|Findings|Appendix"
    )
    next_text = re.sub(
        rf"\A(\s*[^\n.!?]{{12,180}}?)\s+({heading_re})\s+(?=[A-Z])",
        r"\1\n\n\2\n\n",
        repaired,
        flags=re.I,
        count=1,
    )
    if next_text != repaired:
        repaired = next_text
        repairs.append("split_title_from_heading")

    next_text = re.sub(
        rf"(?<=[.!?])\s+({heading_re})\s+(?=[A-Z])",
        r"\n\n\1\n\n",
        repaired,
        flags=re.I,
    )
    if next_text != repaired:
        repaired = next_text
        repairs.append("split_sentence_before_heading")

    next_text = re.sub(
        rf"(?m)^({heading_re})\s+(?=[A-Z])",
        r"\1\n\n",
        repaired,
        flags=re.I,
    )
    if next_text != repaired:
        repaired = next_text
        repairs.append("split_merged_heading")

    return repaired, repairs


def _strip_reference_like_lines_for_quality(text: str) -> str:
    """Remove bibliography/reference lines before repetition quality checks."""
    if not isinstance(text, str) or not text:
        return ""
    kept = []
    in_reference_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if re.match(r"^(?:references|reference list|bibliography)\s*$", line, re.I):
            in_reference_section = True
            continue
        if in_reference_section:
            # Keep prose if a later section heading starts, otherwise ignore
            # the reference block. Long publisher names repeat naturally there.
            if re.match(r"^[A-Z][A-Za-z0-9 ,/&-]{2,70}$", line) and not re.search(r"\(\d{4}\)|https?://", line):
                in_reference_section = False
            else:
                continue
        if re.search(r"https?://|doi\.org|\(\d{4}\)", line, re.I) and len(line.split()) >= 8:
            continue
        kept.append(raw_line)
    return "\n".join(kept)


def _repeated_long_sequence_reason(text: str, window: int = 8) -> str:
    body_text = _strip_reference_like_lines_for_quality(text)
    tokens = re.findall(r"[A-Za-z0-9']+", str(body_text or "").lower())
    if len(tokens) < window * 3:
        return ""
    seen: dict[tuple[str, ...], int] = {}
    for index in range(0, len(tokens) - window + 1):
        gram = tuple(tokens[index:index + window])
        if len(set(gram)) <= 3:
            continue
        if gram in seen and index - seen[gram] > window:
            return "repeated_long_sequence:" + " ".join(gram[:6])
        seen[gram] = index
    return ""


def _repeated_sentence_opening_reason(text: str) -> str:
    body_text = _strip_reference_like_lines_for_quality(text)
    sentences = [
        re.sub(r"\s+", " ", sentence.strip())
        for sentence in re.split(r"(?<=[.!?])\s+", str(body_text or ""))
        if len(sentence.split()) >= 4
    ]
    if len(sentences) < 8:
        return ""
    counts: dict[str, int] = {}
    for sentence in sentences:
        words = re.findall(r"[A-Za-z']+", sentence.lower())
        if len(words) < 3:
            continue
        opening = " ".join(words[:3])
        counts[opening] = counts.get(opening, 0) + 1
    if not counts:
        return ""
    opening, count = max(counts.items(), key=lambda item: item[1])
    if count >= 3 and count / max(1, len(sentences)) >= 0.12:
        return f"repeated_sentence_opening:{opening}"
    return ""


def _external_detector_style_artifact_reason(text: str) -> str:
    """Detect generic promotional/list tone that external detectors flag."""
    if not isinstance(text, str) or not text.strip():
        return ""
    body = _strip_reference_like_lines_for_quality(text)
    missing_space_count = len(re.findall(r"(?<=[a-z0-9)\]])[.!?](?=[A-Z0-9])", body))
    if missing_space_count >= 3:
        return f"missing_sentence_spacing_artifact:{missing_space_count}"
    if len(re.findall(r"[A-Za-z']+", body)) < 80:
        return ""
    sentences = [
        re.sub(r"\s+", " ", sentence.strip())
        for sentence in re.split(r"(?<=[.!?])\s+", body)
        if len(sentence.split()) >= 3
    ]
    if len(sentences) < 8:
        return ""
    word_count = max(1, len(re.findall(r"[A-Za-z']+", body)))
    praise_hits = len(_GENERIC_PRAISE_TERMS_RE.findall(body))
    praise_phrase_hits = len(_GENERIC_PRAISE_PHRASES_RE.findall(body))
    contrast_hits = len(_LOW_FRICTION_CONTRAST_RE.findall(body))
    stylized_hits = len(_STYLIZED_TEXTURE_TERMS_RE.findall(body))
    colloquial_hits = len(_COLLOQUIAL_TEXTURE_TERMS_RE.findall(body))
    abstract_hits = len(_ABSTRACT_LIST_TERMS_RE.findall(body))
    short_fragment_count = sum(1 for sentence in sentences if len(sentence.split()) <= 9)
    colon_fragment_count = sum(1 for sentence in sentences if ":" in sentence and len(sentence.split()) <= 14)
    dash_sentence_count = sum(1 for sentence in sentences if " -- " in sentence or " - " in sentence)
    praise_density = praise_hits / word_count
    phrase_density = praise_phrase_hits / max(1, len(sentences))
    short_fragment_ratio = short_fragment_count / max(1, len(sentences))
    stylized_density = (stylized_hits + colloquial_hits) / word_count
    abstract_density = abstract_hits / word_count

    if (
        praise_hits >= 12
        and praise_density >= 0.035
        and phrase_density >= 0.12
        and contrast_hits < max(4, praise_hits // 4)
    ):
        return "generic_admiration_tone"
    if (
        praise_hits >= 10
        and short_fragment_ratio >= 0.45
        and colon_fragment_count >= 3
    ):
        return "compressed_promotional_fragment_style"
    if (
        stylized_hits + colloquial_hits >= 10
        and stylized_density >= 0.018
        and (dash_sentence_count >= 2 or colon_fragment_count >= 2 or abstract_density >= 0.025)
    ):
        return "over_stylized_metaphorical_texture"
    return ""


_SYNTHETIC_META_ANCHOR_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?m)(?:^|[.!?]\s+)When this is applied in practice,\s+", "when_this_is_applied_in_practice"),
    (r"(?m)(?:^|[.!?]\s+)In this case,\s+", "in_this_case_prefix"),
    (r"(?m)(?:^|[.!?]\s+)During review,\s+", "during_review_prefix"),
    (r"(?m)(?:^|[.!?]\s+)I would narrow the point this way:\s+", "i_would_narrow_prefix"),
    (r"(?m)(?:^|[.!?]\s+)When the process is checked,\s+", "process_checked_prefix"),
    (r"\bOne example is the underlying claim that\b", "underlying_claim_frame"),
)


def _synthetic_meta_anchor_artifact_reason(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    body = _strip_reference_like_lines_for_quality(text)
    for pattern, label in _SYNTHETIC_META_ANCHOR_PATTERNS:
        if re.search(pattern, body, flags=re.I):
            return f"synthetic_meta_anchor_artifact:{label}"
    return ""


def _neutralize_external_detector_style_artifacts(text: str) -> tuple[str, list[str]]:
    """Repair promotional/list-like phrasing without adding new facts."""
    if not isinstance(text, str) or not text.strip():
        return "", []
    updated = text
    repairs: list[str] = []
    replacements: list[tuple[str, str, str]] = [
        (
            r"\b([^.!?]{0,140}?)\s+carve(?:s|d)? innovation'?s edge\.",
            r"\1 are linked to technology and business.",
            "innovation_edge_neutral",
        ),
        (
            r"\bEntrepreneurship thrives in ([^.!?]{3,80})\.",
            r"Business creation is visible in \1.",
            "entrepreneurship_thrives_neutral",
        ),
        (
            r"\bGlobal companies wield economic influence worldwide\.",
            "Large companies operate across many markets.",
            "global_companies_neutral",
        ),
        (
            r"\b([^.!?]{0,100}?)\s+ships culture worldwide(?:\s*--\s*[^.!?]+)?\.",
            r"\1 circulates culture internationally.",
            "ships_culture_neutral",
        ),
        (
            r"\b([^.!?]{0,100}?):\s*global fame magnets\.",
            r"\1 are known internationally.",
            "fame_magnets_neutral",
        ),
        (
            r"\b([^.!?]{0,140}?):\s*forces molding world history\.",
            r"\1 are part of wider public influence.",
            "forces_molding_neutral",
        ),
    ]
    for pattern, replacement, name in replacements:
        next_text = re.sub(pattern, replacement, updated, flags=re.I)
        if next_text != updated:
            updated = next_text
            repairs.append(name)
    updated = re.sub(r"\bglobal fame magnets\b", "people with international recognition", updated, flags=re.I)
    updated = re.sub(r"\b(?:giants|heavyweight ring|throwing heavy punches)\b", "large organizations", updated, flags=re.I)
    updated = re.sub(r"\bships culture worldwide\b", "circulates culture internationally", updated, flags=re.I)
    updated = re.sub(r"\btrumpet(?:s|ed)?\b", "argue for", updated, flags=re.I)
    neutral_terms = [
        (r"\bcarved a sharp route\b", "developed quickly"),
        (r"\bcut(?:s)? a sharp corner\b", "marks a clear point"),
        (r"\bpull(?:s|ed|ing)? strings?\b", "has influence"),
        (r"\btech cogs?\b", "technology systems"),
        (r"\bsprint(?:ed|ing)? to power\b", "rapid growth"),
        (r"\bstunned many\b", "was unusually fast"),
        (r"\bmirrors? a sprawling mix\b", "includes a wide range"),
        (r"\bunder wins\b", "alongside its strengths"),
        (r"\bground zero for\b", "an important base for"),
        (r"\btowering corporations\b", "large corporations"),
        (r"\bsocial tapestry\b", "society"),
        (r"\bthrows? a vast shadow\b", "has wide influence"),
        (r"\bchurns? out\b", "produces"),
        (r"\bhatched stateside\b", "from the same country"),
        (r"\bleans human quirkiness\b", "sounds less mechanically polished"),
        (r"\bpower'?s knobs\b", "centres of power"),
        (r"\bstrands\b", "areas"),
        (r"\bfresh chances\b", "new opportunities"),
        (r"\bfolks\b", "people"),
        (r"\bcash\b", "money"),
        (r"\bgrit\b", "effort"),
        (r"\bglobal fame magnets\b", "people with international recognition"),
        (r"\b(?:giants|heavyweight ring|throwing heavy punches)\b", "large organizations"),
        (r"\bships culture worldwide\b", "circulates culture internationally"),
        (r"\btrumpet(?:s|ed)?\b", "argue for"),
    ]
    for pattern, replacement in neutral_terms:
        updated = re.sub(pattern, replacement, updated, flags=re.I)
    generic_plain_terms = [
        (r"\bremarkable strengths\b", "strengths"),
        (r"\bprestige\b", "reputation"),
        (r"\bsymbolize national identity\b", "are part of public culture"),
        (r"\becho calls for\b", "show interest in"),
    ]
    for pattern, replacement in generic_plain_terms:
        updated = re.sub(pattern, replacement, updated, flags=re.I)
    updated = re.sub(r"[ \t]+", " ", updated)
    updated = re.sub(r"\n{3,}", "\n\n", updated).strip()
    if updated != text:
        repairs.append("external_detector_style_neutralized")
    return updated, repairs
