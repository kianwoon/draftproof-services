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
    if _malformed_negation_order(visible):
        blockers.append("malformed_negation_order")
    if _malformed_modal_do_negation(visible):
        blockers.append("malformed_modal_do_negation")
    if _missing_verb_after_negation_scope(visible):
        blockers.append("missing_verb_after_negation_scope")
    if _malformed_verb_complement(visible):
        blockers.append("malformed_verb_complement")
    if _malformed_with_finite_clause(visible):
        blockers.append("malformed_with_finite_clause")
    if _unsupported_actor_blame_shape(visible):
        blockers.append("unsupported_actor_blame_shape")
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
    if _malformed_nonhuman_activity_predicate(visible):
        blockers.append("malformed_nonhuman_activity_predicate")
    if _malformed_telegraphic_predicate(visible):
        blockers.append("malformed_telegraphic_predicate")
    if _unnatural_completion_phrase(visible):
        blockers.append("unnatural_completion_phrase")
    if _dangling_consequence_tail(visible):
        blockers.append("dangling_consequence_tail")
    if _dangling_additive_tail(visible):
        blockers.append("dangling_additive_tail")
    if _dangling_terminal_and_tail(visible):
        blockers.append("dangling_terminal_and_tail")
    if _standalone_additive_fragment(visible):
        blockers.append("standalone_additive_fragment")
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
    if _vague_unintroduced_reliance(visible):
        blockers.append("vague_unintroduced_reliance")
    if _malformed_tool_actor_relation(visible):
        blockers.append("malformed_tool_actor_relation")
    if _malformed_tool_skill_predicate(visible):
        blockers.append("malformed_tool_skill_predicate")
    if _demonstrative_agreement_error(visible):
        blockers.append("demonstrative_agreement_error")
    if _semantic_anchor_corruption(visible):
        blockers.append("semantic_anchor_corruption")
    if _sentence_starts_with_conjunction(visible):
        blockers.append("sentence_starts_with_conjunction")
    if _stranded_prepositional_fragment(visible):
        blockers.append("stranded_prepositional_fragment")
    if _malformed_connector_fragment(visible):
        blockers.append("malformed_connector_fragment")
    if _malformed_contrast_pair(visible):
        blockers.append("malformed_contrast_pair")
    if _malformed_additive_predicate(visible):
        blockers.append("malformed_additive_predicate")
    if _proxy_context_adjective_stack(visible):
        blockers.append("proxy_context_adjective_stack")
    if _generic_role_inflation(visible):
        blockers.append("generic_role_inflation")
    if _unsupported_evidence_tail(visible):
        blockers.append("unsupported_evidence_tail")
    if _awkward_modal_double_hedge(visible):
        blockers.append("awkward_modal_double_hedge")
    if _vague_danger_opener(visible):
        blockers.append("vague_danger_opener")
    if _duplicated_assessment_consequence(visible):
        blockers.append("duplicated_assessment_consequence")
    if _premature_assessment_consequence(visible):
        blockers.append("premature_assessment_consequence")
    return blockers


def candidate_integrity_warnings(text: str) -> list[str]:
    warnings: list[str] = []
    visible = _normalize(text)
    if _repeated_platform_catalogue(visible):
        warnings.append("repeated_platform_catalogue")
    if _repeated_subject_start(visible):
        warnings.append("repeated_subject_start")
    if _dangling_modifier_sentence_start(visible):
        warnings.append("dangling_modifier_sentence_start")
    if _transition_label_final_consequence(visible):
        warnings.append("transition_label_final_consequence")
    if _compressed_final_consequence_list(visible):
        warnings.append("compressed_final_consequence_list")
    return warnings


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
    plural_subject = r"(?:I|we|they|people|(?!this\b)[A-Za-z][A-Za-z'’-]*s)"
    singular_s_verb = r"(?:carries|involves|creates|requires|supports|guides|explains|shows|makes|helps)"
    return bool(re.search(rf"\b{plural_subject}\s+(?:also\s+)?{singular_s_verb}\b", text, flags=re.I))


def _demonstrative_agreement_error(text: str) -> bool:
    return bool(re.search(r"\bthis\s+(?:encourage|support|reduce|raise|create|make|show|reflect)\b", text, flags=re.I))


def _dangling_modifier_sentence_start(text: str) -> bool:
    return bool(re.search(r"(?:^|[.!?]\s+)No\s+longer\s+(?:merely|just)\b", text, flags=re.I))


def _semantic_anchor_corruption(text: str) -> bool:
    return bool(
        re.search(r"\b(?:words|word)\s+(?:[a-z]+(?:\s+[a-z]+){0,2})\s+(?:know|use|mean|represent)\b", text, flags=re.I)
        or re.search(r"\b(?:the\s+)?words?\s+[\"“'][^\"”']{3,80}[\"”']", text, flags=re.I)
        or re.search(r"\bfocus(?:es)?\s+on\s+(?:the\s+)?words?\s+[\"“'][^\"”']{3,80}[\"”']", text, flags=re.I)
        or re.search(r"\bwords\s+of\s+[a-z]{4,}\b", text, flags=re.I)
        or re.search(r"\blonger\s+streams\s+of\s+content\b", text, flags=re.I)
        or re.search(r"\b(?:model|framework|system)\b[^.!?]{0,80}\b(?:has\s+become|is|became|seems|looks)?\s*longer\s+than\s+before\b", text, flags=re.I)
        or re.search(r"\b(?:model|framework|system)\b[^.!?]{0,80}\b(?:has\s+become|became|is)\s+longer\b", text, flags=re.I)
    )


def _sentence_starts_with_conjunction(text: str) -> bool:
    return bool(re.search(r"(?:^|[.!?]\s+)(?:And|But|Or)\s+\w+", text))


def _stranded_prepositional_fragment(text: str) -> bool:
    return bool(re.search(r"[;:]\s+in\s+real\s+situations\s*[.!?]", text, flags=re.I))


def _malformed_connector_fragment(text: str) -> bool:
    return bool(
        re.search(r"(?:^|[.!?]\s+|[,;]\s*)and\s+in\s+many\s+[a-z][a-z'’-]*s\b", text, flags=re.I)
        or re.search(r"\band\s+which\s+\w+", text, flags=re.I)
        or re.search(r"[;,.]\s*(?:instead|as\s+a\s+result)\s*[.!?]", text, flags=re.I)
        or re.search(r"\band\s+as\s+a\s+result\b", text, flags=re.I)
    )


def _malformed_contrast_pair(text: str) -> bool:
    return bool(re.search(r"\bdoes\s+not\s+only\b[^.?!]*[.?!]\s+Yet\s+it\s+also\b", text, flags=re.I))


def _malformed_additive_predicate(text: str) -> bool:
    return bool(re.search(r"(?:^|[.!?]\s+)It\s+also\s+(?:on|to|for)\b", text, flags=re.I))


def _proxy_context_adjective_stack(text: str) -> bool:
    return bool(re.search(r"\b(?:[a-z][a-z'’-]*[-\s]+){2,}[a-z][a-z'’-]*-rich\s+\w+", text, flags=re.I))


def _generic_role_inflation(text: str) -> bool:
    return bool(
        re.search(
            r"\b[a-z][a-z'’-]*s?\s+(?:now\s+)?(?:serve|act)\s+as\s+"
            r"(?:[a-z][a-z'’-]*(?:ors?|ers?|ists?))(?:\s+and\s+[a-z][a-z'’-]*(?:ors?|ers?|ists?))*\b",
            text,
            flags=re.I,
        )
        or re.search(r"\bas\s+[a-z][a-z'’-]*s\s+(?:adapt\s+to\s+evolving\s+demands|respond\s+to\s+new\s+challenges)\b", text, flags=re.I)
    )


def _duplicated_assessment_consequence(text: str) -> bool:
    lowered = text.casefold()
    assessment_mentions = len(re.findall(r"\bassessment\b", lowered))
    difficult_mentions = len(re.findall(r"\b(?:difficult|harder|hard)\b", lowered))
    return assessment_mentions >= 2 and difficult_mentions >= 2


def _premature_assessment_consequence(text: str) -> bool:
    lowered = text.casefold()
    assessment_at = _first_index(lowered, ("assessment",))
    if assessment_at < 0:
        return False
    risk_at = _first_index(lowered, ("dependent", "dependence", "copy", "submit", "polished", "real ability"))
    return risk_at >= 0 and assessment_at < risk_at


def _transition_label_final_consequence(text: str) -> bool:
    sentences = _sentences(text)
    if not sentences:
        return False
    final = sentences[-1].strip()
    if not re.match(r"^(?:therefore|consequently|thus|as\s+a\s+result)\b", final, flags=re.I):
        return False
    abstract_count = len(re.findall(r"\b[A-Za-z][A-Za-z'’-]*(?:ity|ness|tion|ment|ance|ence)\b", final, flags=re.I))
    list_count = final.count(",") + len(re.findall(r"\band\b", final, flags=re.I))
    return abstract_count >= 2 or list_count >= 2


def _compressed_final_consequence_list(text: str) -> bool:
    sentences = _sentences(text)
    if not sentences:
        return False
    final = sentences[-1].strip()
    list_count = final.count(",") + len(re.findall(r"\band\b", final, flags=re.I))
    abstract_count = len(re.findall(r"\b[A-Za-z][A-Za-z'’-]*(?:ity|ness|tion|ment|ance|ence)\b", final, flags=re.I))
    consequence_verb = re.search(r"\b(?:raise|raises|raising|undermine|undermines|undermining|compromise|compromises|compromising|threaten|threatens|threatening)\b", final, flags=re.I)
    return bool(consequence_verb and list_count >= 2 and abstract_count >= 2)


def _first_index(text: str, terms: tuple[str, ...]) -> int:
    positions = [text.find(term) for term in terms if text.find(term) >= 0]
    return min(positions) if positions else -1


def _split_negation_fragment(text: str) -> bool:
    return bool(re.search(r"\b(?:not|no)\.\s+(?:They|It|This|The|[A-Z][A-Za-z'’-]*)\b", text))


def _malformed_negation_order(text: str) -> bool:
    return bool(
        re.search(r"\b(?:is|are|was|were|be|being|been)\s+longer\s+not\b", text, flags=re.I)
        or re.search(r"\b(?:am|are|is|was|were|be|being|been)\s+(?:do|does|did)\s+not\b", text, flags=re.I)
        or re.search(r"\blonger\s+not\s+fully\s+reflecting\b", text, flags=re.I)
        or re.search(r"\bnot\s+longer\s+(?:appropriate|suitable|adequate|sufficient|reflecting|reflects?)\b", text, flags=re.I)
        or _bare_not_always_without_auxiliary(text)
    )


def _malformed_modal_do_negation(text: str) -> bool:
    return bool(re.search(r"\b(?:may|might|could|should|would|can)\s+do\s+not\s+\w+", text, flags=re.I))


def _bare_not_always_without_auxiliary(text: str) -> bool:
    for match in re.finditer(r"\bnot\s+always\b", str(text or ""), flags=re.I):
        prefix_words = re.findall(r"[A-Za-z][A-Za-z'’-]*", str(text or "")[:match.start()])
        if not prefix_words:
            return True
        if prefix_words[-1].casefold() not in {"do", "does", "did", "can", "could", "may", "might", "must", "should", "will", "would", "is", "are", "was", "were"}:
            return True
    return False


def _missing_verb_after_negation_scope(text: str) -> bool:
    return bool(re.search(r"\b(?:do|does|did)\s+not\s+always\s+(?:how|what|when|where|why|whether)\b", str(text or ""), flags=re.I))


def _malformed_verb_complement(text: str) -> bool:
    malformed = re.search(
        r"\b(?!after\b|before\b|by\b|during\b|for\b|from\b|through\b|when\b|while\b|with\b|without\b)"
        r"[A-Za-z][A-Za-z'’-]*ed\s+[a-z]{4,}ing\b",
        text,
        flags=re.I,
    )
    progressive = re.search(r"\b(?:am|are|is|was|were|be|being|been)\s+[a-z]{4,}ing\b", text, flags=re.I)
    return bool(malformed and not progressive)


def _malformed_with_finite_clause(text: str) -> bool:
    return bool(
        re.search(
            r"\bwith\s+(?:I|we|they|[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,2})\s+"
            r"(?:received|practiced|practised|proved|learned|used|created|made|became|were|was|are|is)\b",
            text,
            flags=re.I,
        )
    )


def _unsupported_actor_blame_shape(text: str) -> bool:
    human_group = r"(?:people|participants|[A-Za-z][A-Za-z'’-]*(?:ers|ents|ants|ists|ors|ors?))"
    return bool(
        re.search(rf"\b{human_group}\s+(?:become|became|are|were)\s+difficult\b", text, flags=re.I)
        or re.search(rf"\b{human_group}\s+are\s+unwilling\s+to\s+[a-z][a-z'’-]{{2,}}\s+because\b", text, flags=re.I)
    )


def _bare_instruction_fragment(text: str) -> bool:
    return bool(
        re.search(
            r"(?:^|[.!?]\s+)(?:Help|Promote|Enable|Teach|Apply|Use|Encourage)\s+"
            r"(?:them|the|[a-z][a-z'’-]*(?:s|al|ive|ing|ical)?)\b",
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
        or re.search(r"\b(?:source\s+groups|source\s+slots?|source\s+beats?|words\s+[a-z][a-z'’-]*s?\s+learn)\b", text, flags=re.I)
        or re.search(r"\b(?:provides?\s+evidence|bridge\s+shows|author-?proxy|planner)\b", text, flags=re.I)
    )


def _unsupported_evidence_tail(text: str) -> bool:
    return bool(
        re.search(r"(?:^|[.!?]\s+)(?:Research|Evidence|Studies?)\b[^.!?]{0,120}\b(?:supports?|shows?|proves?|confirms?)\b", text, flags=re.I)
        or re.search(r"\bsupports?\s+this\s+(?:observation|claim|point|argument)\b", text, flags=re.I)
    )


def _awkward_modal_double_hedge(text: str) -> bool:
    return bool(re.search(r"\b(?:risk|danger)\s+\w+\s+if\b[^.!?]{0,80}\b(?:may|might|could)\b", text, flags=re.I))


def _vague_danger_opener(text: str) -> bool:
    return bool(
        re.search(
            r"(?:^|[.!?]\s+)(?:But\s+|Yet\s+|However,\s+)?(?:there\s+is\s+|a\s+)?danger\s+(?:also\s+)?(?:appears?|emerges?|exists?|is)\b",
            text,
            flags=re.I,
        )
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
    for sentence in _sentences(text):
        if _has_bare_verb_run(sentence):
            return True
    return False


def _malformed_nominal_stack(text: str) -> bool:
    for sentence in _sentences(text):
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", sentence)
        run = 0
        for word in words:
            if _abstract_noun_like(word):
                run += 1
                if run >= 3:
                    return True
            else:
                run = 0
    return False


def _malformed_nonhuman_activity_predicate(text: str) -> bool:
    nonhuman_learning_actor = (
        r"(?:tools?|apps?|courses?|platforms?|resources?|engines?|"
        r"media|communities|websites?|videos?|tutorials?)"
    )
    return bool(
        re.search(rf"\b{nonhuman_learning_actor}(?:\s+(?:and|or)\s+{nonhuman_learning_actor})?\s+are\s+learning\b", text, flags=re.I)
        or re.search(rf"\b[A-Z][^.!?]{{0,80}}\b{nonhuman_learning_actor}[^.!?]{{0,80}}\s+are\s+learning\b", text, flags=re.I)
    )


def _malformed_telegraphic_predicate(text: str) -> bool:
    return bool(
        re.search(r"(?:^|[.!?]\s+)Created\s+\w+(?:\s+\w+){1,4}\s+is\s+present\b", text, flags=re.I)
        or re.search(r"(?:^|[.!?]\s+)(?:Real\s+)?[A-Z]?[a-z]+\s+challenge\s+is\s+knowing\s+\w+\s+\w+\s+\w+", text, flags=re.I)
        or re.search(r"(?:^|[.!?]\s+)A\s+\w+\s+no\s+longer\s+just\s+\w+", text, flags=re.I)
        or re.search(r"\bthis\s+importance\s+is\s+not\s+less\b", text, flags=re.I)
    )


def _unnatural_completion_phrase(text: str) -> bool:
    return bool(re.search(r"\b(?:complete|completes|completed)\s+the\s+(?:assessment|challenge|picture|process|criteria)\b", text, flags=re.I))


def _dangling_consequence_tail(text: str) -> bool:
    return bool(re.search(r"\b(?:and|where|because)\s+(?:consequently|ultimately|therefore)\s*[.!?]", text, flags=re.I))


def _dangling_additive_tail(text: str) -> bool:
    return bool(
        re.search(r"\b(?:and|or)\s+(?:additionally|also|further|moreover)\s*[.!?]", text, flags=re.I)
        or any(_has_dangling_participle_tail(sentence) for sentence in _sentences(text))
    )


def _dangling_terminal_and_tail(text: str) -> bool:
    for sentence in _sentences(text):
        lowered = sentence.casefold()
        if not re.search(r"\b(?:what|which|whether)\s+(?:is|are|can\s+be)\b", lowered):
            continue
        if sentence.count(",") < 2:
            continue
        if re.search(
            r"\b(?:worth|safe|fair|valid|useful|ethical|accurate|original|responsible)\s+\w+\s+and\s+[a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,2}\s*[.!?]?$",
            lowered,
        ):
            return True
    return False


def _has_dangling_participle_tail(sentence: str) -> bool:
    match = re.search(
        r"\band\s+([a-z][a-z'’-]*ed)\s+([a-z][a-z'’-]*)\s*$",
        str(sentence or "").strip(),
        flags=re.I,
    )
    if not match:
        return False
    prefix = sentence[: match.start()].strip()
    if re.search(r"\b(?:is|are|was|were|be|been|being|get|gets|got)\b[^.!?]{0,80}$", prefix, flags=re.I):
        return False
    return bool(re.match(r"^(?:well|properly|clearly|carefully|effectively|fully)$", match.group(2), flags=re.I))


def _standalone_additive_fragment(text: str) -> bool:
    return any(
        re.match(r"^(?:as\s+well\s+as|along\s+with|together\s+with)\b", sentence, flags=re.I)
        for sentence in _sentences(text)
    )


def _malformed_parallel_connector_list(text: str) -> bool:
    mixed_connector = r"\b(?:as\s+well\s+as|along\s+with|together\s+with)\s+[^.!?]{0,120}\band\s+[^.!?]{0,80}\band\s+[^.!?]{0,80}[.!?]"
    repeated_final_and = r"\b(?:[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,2})\s+and\s+(?:[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,2})\s+and\s+(?:[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,2})\b"
    return any(
        _catalogue_item_count(sentence) >= 4
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
        hits = _catalogue_item_count(sentence)
        if hits >= 4 and sentence.count(",") == 0 and sentence.casefold().count(" and ") <= 1:
            return True
    return False


def _lost_serial_punctuation(text: str) -> bool:
    for sentence in _sentences(text):
        if re.search(
            r"\b(?:what|which|that|whether)\s+(?:is|are|can\s+be)\s+(?:[a-z]{4,}\s+){3,}(?:and\s+)?[a-z]{4,}\b",
            sentence,
            flags=re.I,
        ) and sentence.count(",") == 0:
            return True
        if _catalogue_item_count(sentence) >= 4 and sentence.count(",") == 0:
            return True
    return False


def _capitalized_common_noun_mid_sentence(text: str) -> bool:
    return bool(re.search(r"\b(?:and|but|because|while|where|when|that)\s+[A-Z][a-z]{3,}\s+(?:is|are|was|were|has|have|becomes?|makes?)\b", text))


def _repeated_platform_catalogue(text: str) -> bool:
    previous_hits = 0
    for sentence in _sentences(text):
        hits = _catalogue_item_count(sentence)
        if previous_hits >= 4 and hits >= 4:
            return True
        previous_hits = hits
    return False


def _repeated_subject_start(text: str) -> bool:
    starts: dict[str, int] = {}
    for sentence in _sentences(text):
        match = re.match(
            r"^([A-Za-z][A-Za-z]*(?:\s+[A-Za-z][A-Za-z]*){0,2})\s+"
            r"(?:can|could|may|might|must|should|also|is|are|has|have|helps?|improves?|supports?|makes?|raises?|creates?)\b",
            sentence,
            flags=re.I,
        )
        if not match:
            continue
        key = match.group(1).casefold()
        starts[key] = starts.get(key, 0) + 1
    return any(count >= 2 for count in starts.values())


def _vague_unintroduced_reliance(text: str) -> bool:
    lowered = text.casefold()
    match = re.search(
        r"\b(?:a\s+)?(?:danger|risk)\s+(?:also\s+)?(?:emerges|appears|exists|arises)\s+from\s+this\s+reliance\b",
        lowered,
    )
    if not match:
        return False
    before = lowered[:match.start()]
    return not re.search(r"\b(?:rely|relies|reliance|dependent|dependence|depending)\b", before)


def _malformed_tool_actor_relation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:tools?|systems?|platforms?|software|technology|applications?|programs?|services?)\s+"
            r"and\s+(?:people|teams?|users?|[a-z][a-z'’-]*(?:ers|ents|ants|ists|ors))\s+belong\s+together\b",
            text,
            flags=re.I,
        )
    )


def _malformed_tool_skill_predicate(text: str) -> bool:
    tool_subject = r"(?:the\s+same\s+)?(?:tools?|systems?|platforms?|software|technology|applications?|programs?|services?)"
    for sentence in _sentences(text):
        if re.search(
            rf"\b{tool_subject}\s+(?:also\s+)?(?:improve|improves)\s+[a-z][a-z'’-]+\s+and\s+[a-z][a-z'’-]*\s+skills\b",
            sentence,
            flags=re.I,
        ):
            return True
        if re.search(
            rf"\b{tool_subject}\s+(?:also\s+)?(?:improve|improves)[^.!?]{{0,50}}\band\s+[a-z][a-z'’-]*\s+skills\b",
            sentence,
            flags=re.I,
        ) and not re.search(r"\b(?:allow|let|help)\s+(?:people|teams?|users?|[a-z][a-z'’-]*(?:ers|ents|ants|ists|ors))\s+to?\s*[a-z][a-z'’-]*\s+skills\b", sentence, flags=re.I):
            return True
    return False


def _has_bare_verb_run(sentence: str) -> bool:
    if _repeated_auxiliary_stack(sentence):
        return True
    segments = [part for part in re.split(r"[,;:]", str(sentence or "")) if part.strip()]
    return any(_segment_has_bare_verb_run(segment) for segment in segments)


def _repeated_auxiliary_stack(sentence: str) -> bool:
    auxiliaries = (
        "am",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "was",
        "were",
        "will",
        "would",
    )
    pattern = "|".join(auxiliaries)
    return bool(re.search(rf"\b({pattern})\s+\1\s+(?:not|never|no)\b", sentence, flags=re.I))


def _segment_has_bare_verb_run(segment: str) -> bool:
    return bool(
        re.search(
            r"\b[a-z][a-z'’-]{3,}\s+and\s+[a-z][a-z'’-]{3,}\s+and\s+[a-z][a-z'’-]{3,}\b",
            str(segment or ""),
            flags=re.I,
        )
    )


def _abstract_noun_like(word: str) -> bool:
    value = str(word or "").casefold()
    return len(value) >= 5 and (
        value.endswith(("ness", "ics", "ity", "ship", "ment", "tion", "sion", "ance", "ence", "ability", "ibility"))
    )


def _catalogue_item_count(sentence: str) -> int:
    text = str(sentence or "")
    comma_items = [part.strip() for part in text.split(",") if _catalogue_item_like(part)]
    if len(comma_items) >= 3:
        return len(comma_items)
    return max(len(comma_items), _undelimited_catalogue_count(text))


def _catalogue_item_like(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", str(text or ""))
    return 1 <= len(words) <= 5


def _undelimited_catalogue_count(text: str) -> int:
    prefix = re.split(r"\b(?:has|have|had|is|are|was|were|created?|produced?|made|makes?|forms?|becomes?)\b", str(text or ""), maxsplit=1, flags=re.I)[0]
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", prefix)
    if len(tokens) < 5:
        return 0
    if sum(1 for token in tokens if _distinctive_token(token)) < 2:
        return 0
    count = 0
    index = 0
    while index < len(tokens):
        current = tokens[index]
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if _distinctive_token(current) and following and not _distinctive_token(following):
            count += 1
            index += 2
            continue
        if _distinctive_token(current):
            count += 1
            index += 1
            continue
        if following and not _distinctive_token(following):
            count += 1
            index += 2
            continue
        index += 1
    return count


def _distinctive_token(token: str) -> bool:
    value = str(token or "")
    return value.isupper() or bool(re.search(r"[A-Z].*[A-Z]|[a-z][A-Z]", value))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "")) if part.strip()]
