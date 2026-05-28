from __future__ import annotations

import re


def candidate_integrity_blockers(text: str) -> list[str]:
    blockers: list[str] = []
    visible = _normalize(text)
    if _broken_citation_shape(visible):
        blockers.append("broken_citation_shape")
    if _citation_name_wrapper(visible):
        blockers.append("citation_name_wrapper")
    if _dangling_article_predicate(visible):
        blockers.append("dangling_article_predicate")
    if _subject_verb_agreement_error(visible):
        blockers.append("malformed_subject_verb_agreement")
    if _split_negation_fragment(visible):
        blockers.append("split_negation_fragment")
    if _malformed_verb_complement(visible):
        blockers.append("malformed_verb_complement")
    if _unsupported_learner_blame_shape(visible):
        blockers.append("unsupported_learner_blame_shape")
    if _bare_instruction_fragment(visible):
        blockers.append("bare_instruction_fragment")
    if _citation_report_sentence(visible):
        blockers.append("citation_report_sentence")
    if _stranded_thereby_citation(visible):
        blockers.append("stranded_thereby_citation")
    if _planner_language_leakage(visible):
        blockers.append("planner_language_leakage")
    if _external_narrator_reporting_chain(visible):
        blockers.append("external_narrator_reporting_chain")
    if _malformed_serial_verb_chain(visible):
        blockers.append("malformed_serial_verb_chain")
    if _malformed_nominal_stack(visible):
        blockers.append("malformed_nominal_stack")
    if _malformed_learning_predicate(visible):
        blockers.append("malformed_learning_predicate")
    if _malformed_telegraphic_predicate(visible):
        blockers.append("malformed_telegraphic_predicate")
    if _unnatural_completion_phrase(visible):
        blockers.append("unnatural_completion_phrase")
    if _dangling_consequence_tail(visible):
        blockers.append("dangling_consequence_tail")
    if _dangling_additive_tail(visible):
        blockers.append("dangling_additive_tail")
    if _standalone_additive_fragment(visible):
        blockers.append("standalone_additive_fragment")
    if _misplaced_channel_in_challenge(visible):
        blockers.append("misplaced_channel_in_challenge")
    if _malformed_parallel_connector_list(visible):
        blockers.append("malformed_parallel_connector_list")
    if _malformed_parallel_verb_tail(visible):
        blockers.append("malformed_parallel_verb_tail")
    if _redundant_trust_phrase(visible):
        blockers.append("redundant_trust_phrase")
    if _keyword_dump_sequence(visible):
        blockers.append("keyword_dump_sequence")
    if _lost_serial_punctuation(visible):
        blockers.append("lost_serial_punctuation")
    if _capitalized_common_noun_mid_sentence(visible):
        blockers.append("capitalized_common_noun_mid_sentence")
    if _repeated_platform_catalogue(visible):
        blockers.append("repeated_platform_catalogue")
    if _repeated_subject_start(visible):
        blockers.append("repeated_subject_start")
    if _vague_unintroduced_reliance(visible):
        blockers.append("vague_unintroduced_reliance")
    if _malformed_tool_student_relation(visible):
        blockers.append("malformed_tool_student_relation")
    if _tool_practise_skills_predicate(visible):
        blockers.append("tool_practise_skills_predicate")
    return blockers


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _broken_citation_shape(text: str) -> bool:
    if text.count("(") != text.count(")"):
        return True
    return bool(
        re.search(r"\([A-Z][A-Za-z]+(?:\s+et\s+al)?\.\s+[A-Z]|\([A-Z][A-Za-z]+(?:\s+et\s+al)?\.\s+(?:They|It|This|The)\b", text)
        or re.search(r"\([A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?\s+(?:19|20)\d{2}[a-z]?(?:\s*[;)])", text)
    )


def _citation_name_wrapper(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:from|by|with|according to)\s+[A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?\s*\((?:19|20)\d{2}[a-z]?\)\s+"
            r"(?:aligns?|supports?|shows?|confirms?|proves?|highlights?|explains?)\b",
            text,
        )
    )


def _dangling_article_predicate(text: str) -> bool:
    return bool(re.search(r"(?:^|[.!?]\s+)The\s+(?:involves?|created|creates|helps?|shows?|supports?|requires?|guides?)\b", text, flags=re.I))


def _subject_verb_agreement_error(text: str) -> bool:
    plural_subject = r"(?:I|we|they|students|learners|teachers|educators|schools|teams|classes|people)"
    singular_s_verb = r"(?:carries|involves|creates|requires|supports|guides|explains|shows|makes|helps)"
    return bool(re.search(rf"\b{plural_subject}\s+(?:also\s+)?{singular_s_verb}\b", text, flags=re.I))


def _split_negation_fragment(text: str) -> bool:
    return bool(re.search(r"\b(?:not|no)\.\s+(?:They|It|This|The|Students|Learners)\b", text, flags=re.I))


def _malformed_verb_complement(text: str) -> bool:
    return bool(re.search(r"\b(?:learned|learn|guide|guides|guided|question)\s+(?:accepting|compare|develop|apply|create|created|creating)\b", text, flags=re.I))


def _unsupported_learner_blame_shape(text: str) -> bool:
    human_group = r"(?:learners|students|clients|people|participants)"
    return bool(
        re.search(rf"\b{human_group}\s+(?:become|became|are|were)\s+difficult\b", text, flags=re.I)
        or re.search(rf"\b{human_group}\s+are\s+unwilling\s+to\s+learn\s+because\b", text, flags=re.I)
    )


def _bare_instruction_fragment(text: str) -> bool:
    return bool(
        re.search(
            r"(?:^|[.!?]\s+)(?:Help|Promote|Enable|Teach|Apply|Use|Encourage)\s+"
            r"(?:them|students|learners|clients|inclusive|practical|the)\b",
            text,
        )
    )


def _citation_report_sentence(text: str) -> bool:
    return bool(
        re.search(r"(?:^|[.!?]\s+)The same outcome was observed in later studies\s*\([^)]+\)\.?", text)
        or re.search(r"(?:^|[.!?]\s+)Further evidence supports this finding\s*\([^)]+\)\.?", text)
    )


def _stranded_thereby_citation(text: str) -> bool:
    return bool(re.search(r"\bthereby\s*\([^)]+\)", text, flags=re.I))


def _planner_language_leakage(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:coverage\s+beat|source\s+slot|source_sentence_id|writer_execution_contract|"
            r"construction\s+recipe|route\s+question|active\s+variant|planner\s+decision|"
            r"relationship\s+sees|beat\s+plan|coverage\s+capsule|source\s+sentence)\b",
            text,
            flags=re.I,
        )
        or re.search(
            r"(?:^|[.!?]\s+)(?:guide|compare|develop|apply|focus|think|words)\s+"
            r"(?:prompts?|occurs?|expands?|shapes?|relationship|route|function)\b",
            text,
            flags=re.I,
        )
        or re.search(
            r"(?:^|[.!?]\s+)(?:shift|teacher|good\s+teacher)\s+"
            r"(?:has|is|helps?)\b",
            text,
            flags=re.I,
        )
        or re.search(r"\b(?:source\s+groups|education\s+today\s+emphasizes\s+the\s+words|words\s+students\s+learn)\b", text, flags=re.I)
    )


def _external_narrator_reporting_chain(text: str) -> bool:
    report_verbs = (
        "adds",
        "compares",
        "concludes",
        "decides",
        "emphasizes",
        "explains",
        "finds",
        "mentions",
        "notes",
        "observes",
        "points out",
        "records",
        "reads",
        "sees",
        "states",
        "struggles",
        "stresses",
    )
    verb_pattern = "|".join(re.escape(verb) for verb in report_verbs)
    reporting_sentences = re.findall(
        rf"(?:^|[.!?]\s+|,\s*(?:and\s+)?(?:finally|then|therefore),?\s*)"
        rf"(?:the\s+writer|the\s+author|he|she|they)\s+(?:{verb_pattern})\s+that\b",
        text,
        flags=re.I,
    )
    if len(reporting_sentences) >= 2:
        return True
    return bool(
        re.search(
            rf"(?:^|[.!?]\s+|,\s*(?:and\s+)?(?:finally|then|therefore),?\s*)"
            rf"(?:the\s+writer|the\s+author)\s+(?:{verb_pattern})\s+that\b",
            text,
            flags=re.I,
        )
        and re.search(
            rf"(?:^|[.!?]\s+|,\s*(?:and\s+)?(?:finally|then|therefore),?\s*)"
            rf"(?:he|she|they)\s+(?:{verb_pattern})\s+that\b",
            text,
            flags=re.I,
        )
    )


def _malformed_serial_verb_chain(text: str) -> bool:
    return bool(
        re.search(r"\bthink\s+deeply\s+solve\s+problems\b", text, flags=re.I)
        or re.search(r"\b(?:analyse|analyze)\s+adapt\s+communicate\b", text, flags=re.I)
    )


def _malformed_nominal_stack(text: str) -> bool:
    return bool(re.search(r"\busefulness\s+ethics\s+trustworthiness\b", text, flags=re.I))


def _malformed_learning_predicate(text: str) -> bool:
    nonhuman_learning_actor = (
        r"(?:ai\s+tools|apps?|courses?|platforms?|resources?|search\s+engines?|"
        r"social\s+media|peer\s+communities|websites?|videos?|tutorials?)"
    )
    return bool(
        re.search(rf"\b{nonhuman_learning_actor}(?:\s+(?:and|or)\s+{nonhuman_learning_actor})?\s+are\s+learning\b", text, flags=re.I)
        or re.search(rf"\b[A-Z][^.!?]{{0,80}}\b{nonhuman_learning_actor}[^.!?]{{0,80}}\s+are\s+learning\b", text, flags=re.I)
    )


def _malformed_telegraphic_predicate(text: str) -> bool:
    return bool(
        re.search(r"(?:^|[.!?]\s+)Created\s+\w+\s+learning\s+environment\s+is\s+present\b", text, flags=re.I)
        or re.search(r"(?:^|[.!?]\s+)Real\s+challenge\s+is\s+knowing\s+\w+\s+\w+\s+\w+", text, flags=re.I)
    )


def _unnatural_completion_phrase(text: str) -> bool:
    return bool(re.search(r"\b(?:complete|completes|completed)\s+the\s+(?:assessment|challenge|picture|process|criteria)\b", text, flags=re.I))


def _dangling_consequence_tail(text: str) -> bool:
    return bool(re.search(r"\b(?:and|where|because)\s+(?:consequently|ultimately|therefore)\s*[.!?]", text, flags=re.I))


def _dangling_additive_tail(text: str) -> bool:
    return bool(re.search(r"\b(?:and|or)\s+(?:additionally|also|further|moreover)\s*[.!?]", text, flags=re.I))


def _standalone_additive_fragment(text: str) -> bool:
    return any(
        re.match(r"^(?:as\s+well\s+as|along\s+with|together\s+with)\b", sentence, flags=re.I)
        for sentence in _sentences(text)
    )


def _misplaced_channel_in_challenge(text: str) -> bool:
    channel = r"(?:online courses|ai tools|search engines|social media|peer communities|youtube|tiktok)"
    return bool(
        re.search(rf"\b(?:accurate|useful|ethical|trustworthy|worth trusting)\s+and\s+{channel}\b", text, flags=re.I)
    )


def _malformed_parallel_connector_list(text: str) -> bool:
    mixed_connector = r"\b(?:as\s+well\s+as|along\s+with|together\s+with)\s+[^.!?]{0,120}\band\s+[^.!?]{0,80}\band\s+[^.!?]{0,80}[.!?]"
    platform = r"(?:youtube|tiktok|online courses|ai tools|search engines|social media|peer communities)"
    repeated_final_and = rf"\b{platform}\s+and\s+{platform}\s+and\s+{platform}\b"
    return any(
        _platform_hit_count(sentence) >= 4
        and (re.search(mixed_connector, sentence, flags=re.I) or re.search(repeated_final_and, sentence, flags=re.I))
        for sentence in _sentences(text)
    )


def _malformed_parallel_verb_tail(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:help|helps|helping)\s+\w+\s+\w+\s+[^.!?]{0,80}\band\s+[^.!?]{0,80},\s*(?:improve|improving|practise|practising|practice|practicing)\b",
            text,
            flags=re.I,
        )
    )


def _redundant_trust_phrase(text: str) -> bool:
    return bool(re.search(r"\btrustworthy\s+and\s+worth\s+trusting\b|\bworth\s+trusting\s+and\s+trustworthy\b", text, flags=re.I))


def _keyword_dump_sequence(text: str) -> bool:
    for sentence in _sentences(text):
        hits = _platform_hit_count(sentence)
        if hits >= 4 and sentence.count(",") == 0 and sentence.casefold().count(" and ") <= 1:
            return True
    return False


def _lost_serial_punctuation(text: str) -> bool:
    for sentence in _sentences(text):
        lowered = sentence.casefold()
        if re.search(r"\baccurate\s+useful\s+ethical\s+(?:and\s+)?worth\s+trusting\b", lowered):
            return True
        if re.search(r"\baccurate\s+useful\s+ethical\s+trustworthy\b", lowered):
            return True
        if _platform_hit_count(sentence) >= 4 and sentence.count(",") == 0:
            return True
    return False


def _capitalized_common_noun_mid_sentence(text: str) -> bool:
    common = (
        "Access",
        "Assessment",
        "Challenge",
        "Education",
        "Information",
        "Knowledge",
        "Learning",
        "Process",
        "System",
        "Technology",
    )
    noun_pattern = "|".join(common)
    return bool(re.search(rf"\b(?:and|but|because|while|where|when|that)\s+(?:{noun_pattern})\b", text))


def _repeated_platform_catalogue(text: str) -> bool:
    previous_hits = 0
    for sentence in _sentences(text):
        hits = _platform_hit_count(sentence)
        if previous_hits >= 4 and hits >= 4:
            return True
        previous_hits = hits
    return False


def _repeated_subject_start(text: str) -> bool:
    starts: dict[str, int] = {}
    for sentence in _sentences(text):
        match = re.match(r"^(AI tools|The situation|The same tools)\b", sentence, flags=re.I)
        if not match:
            continue
        key = match.group(1).casefold()
        starts[key] = starts.get(key, 0) + 1
    return any(count >= 2 for count in starts.values())


def _vague_unintroduced_reliance(text: str) -> bool:
    lowered = text.casefold()
    marker = "a danger also emerges from this reliance"
    if marker not in lowered:
        return False
    before = lowered.split(marker, 1)[0]
    return not re.search(r"\b(?:rely|relies|reliance|dependent|dependence|depending)\b", before)


def _malformed_tool_student_relation(text: str) -> bool:
    return bool(re.search(r"\bAI tools and students belong together\b", text, flags=re.I))


def _tool_practise_skills_predicate(text: str) -> bool:
    return bool(re.search(r"\b(?:AI tools|The same tools|tools)\s+(?:also\s+)?(?:improve|improves)[^.!?]{0,80}\bpracti[cs]e\s+skills\b", text, flags=re.I))


def _platform_hit_count(sentence: str) -> int:
    lowered = sentence.casefold()
    terms = (
        "youtube",
        "tiktok",
        "online courses",
        "ai tools",
        "search engines",
        "social media",
        "peer communities",
    )
    return sum(1 for term in terms if term in lowered)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "")) if part.strip()]
