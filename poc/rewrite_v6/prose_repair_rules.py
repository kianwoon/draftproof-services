from __future__ import annotations

import re
from typing import Any


def consolidated_prose_repair_rules() -> list[dict[str, Any]]:
    return [
        _rule("generic_opening", "Start with the real pressure, not the big topic.", "specific pressure/action + why it matters"),
        _rule("balanced_both_sides_opener", "Do not announce benefits and risks. Show the tension.", "benefit + specific risk"),
        _rule("abstract_noun_stack", "Turn abstract nouns into actor-action wording.", "who must judge what"),
        _rule("compressed_list", "Do not pack too many skills into one sentence.", "1-2 skills + visible action"),
        _rule("repeated_causal_glue", "Do not make every sentence a cause-effect sentence.", "observation / contrast / stakeholder effect"),
        _rule("this_is_important_because", "Let the reason stand directly when possible.", "reason as its own sentence"),
        _rule("not_only_but_also", "Break the balanced contrast.", "X still matters + Y now matters too"),
        _rule("modal_scope_drift", "Keep uncertainty attached to the same idea.", "may + but do not always + same skill group"),
        _rule("over_formal_synonym", "Do not upgrade simple words into academic filler.", "plain words such as use, help, reduce, ability"),
        _rule("same_sentence_route", "Change sentence job, not only wording.", "claim -> example -> contrast -> effect"),
        _rule("too_smooth_progression", "Add a scoped step or local contrast.", "old model still exists + new pressure beside it"),
        _rule("generic_skill_list", "Convert policy skills into student behaviour.", "check, explain, compare, apply"),
        _rule("generic_conclusion", "Close with the specific consequence, not a broad moral.", "specific next implication"),
        _rule("modern_world_claim", "Replace the big setting with a real setting.", "work, daily life, classroom, or assessment"),
        _rule("passive_institutional_voice", "Put the actor back.", "actor has a harder time judging/doing the task"),
        _rule("used_well_template", "Replace vague condition with specific behaviour.", "when actor uses tool to do a specific action"),
        _rule("broad_claim", "Narrow by setting, task, actor, or condition.", "in task, tool helps actor do specific thing"),
        _rule("repeated_can_chain", "Reduce repeated modal verbs.", "specific verb + condition"),
        _rule("dependency_claim", "Show what dependence looks like.", "visible behaviour + learning gap"),
        _rule("polished_work_claim", "Show the mismatch between output and ability.", "finished work + student cannot explain it"),
        _rule("repeated_sentence_starter", "Change the entry point.", "setting / actor / effect / contrast"),
        _rule("transition_announcement", "Do not announce the transition. Make the next claim.", "concrete changed condition"),
        _rule("three_beat_rhythm", "Break tidy list rhythm.", "A and B + detail about C"),
        _rule("circular_repetition", "Do not end by repeating the first cause.", "consequence / next implication"),
        _rule("no_longer_repetition", "Avoid repeated negation structures.", "direct positive claim + contrast"),
    ]


def prose_repair_rule_summary() -> list[str]:
    return [
        f"{rule['problem']}: {rule['simple_rule']} Use {rule['preferred_shape']}."
        for rule in consolidated_prose_repair_rules()
    ]


def risky_source_terms(source_text: str) -> set[str]:
    terms = {phrase.casefold() for phrase in risky_source_copy_phrases(source_text)}
    if re.search(r"\b(?:when\s+)?used\s+well\b", str(source_text or ""), flags=re.I):
        terms.update({"used", "well"})
    return terms


def risky_source_shape_guidance(source_text: str) -> list[str]:
    guidance: list[str] = []
    text = str(source_text or "")
    if _balanced_benefit_risk_phrases(text):
        guidance.append("balanced benefit/risk opener")
    if _used_well_phrases(text):
        guidance.append("vague good-use template")
    if _vague_danger_phrases(text):
        guidance.append("vague abstract risk label")
    if _abstract_question_phrases(text):
        guidance.append("abstract question-raising consequence")
    if _demonstrative_consequence_phrases(text):
        guidance.append("demonstrative consequence opener")
    return _dedupe(guidance)


def risky_source_copy_phrases(source_text: str) -> list[str]:
    text = str(source_text or "")
    return _dedupe([
        *_balanced_benefit_risk_phrases(text),
        *_used_well_phrases(text),
        *_vague_danger_phrases(text),
        *_abstract_question_phrases(text),
        *_demonstrative_consequence_phrases(text),
    ])


def risky_source_rewrite_guidance(source_text: str) -> list[str]:
    text = str(source_text or "")
    rows: list[str] = []
    if _balanced_benefit_risk_phrases(text):
        rows.append("Preserve the benefit/risk tension through concrete action and risk, not the balanced opener wording.")
    if _used_well_phrases(text):
        rows.append("Replace vague good-use wording with the specific actor and tool behaviour supported by the source.")
    if _vague_danger_phrases(text):
        rows.append("Show the risk as visible source behaviour instead of a vague danger label.")
    if _abstract_question_phrases(text) or _demonstrative_consequence_phrases(text):
        rows.append("Close with what becomes harder to judge, using actor-action wording rather than abstract question-raising.")
    return _dedupe(rows)


def _balanced_benefit_risk_phrases(text: str) -> list[str]:
    rows: list[str] = []
    pattern = re.compile(
        r"\b(?:both\s+)?(?:opportunities?|benefits?|advantages?)\s+and\s+"
        r"(?:risks?|challenges?|dangers?|concerns?)\b",
        flags=re.I,
    )
    for match in pattern.finditer(text):
        phrase = _clean_phrase(match.group(0))
        rows.append(phrase)
        if phrase.casefold().startswith("both "):
            rows.append(phrase[5:])
    return rows


def _used_well_phrases(text: str) -> list[str]:
    return [_clean_phrase(match.group(0)) for match in re.finditer(r"\b(?:when\s+)?used\s+well\b", text, flags=re.I)]


def _vague_danger_phrases(text: str) -> list[str]:
    pattern = re.compile(
        r"\b(?:(?:but|yet|however),?\s+)?(?:there\s+is\s+(?:also\s+)?a\s+danger|"
        r"a\s+danger\s+(?:also\s+)?(?:emerges|appears|exists|arises)|"
        r"(?:danger|risk)\s+(?:also\s+)?(?:emerges|appears|exists|arises))\b",
        flags=re.I,
    )
    rows: list[str] = []
    for match in pattern.finditer(text):
        phrase = _clean_phrase(match.group(0))
        rows.append(phrase)
        rows.append(re.sub(r"^(?:but|yet|however),?\s+", "", phrase, flags=re.I))
    return rows


def _abstract_question_phrases(text: str) -> list[str]:
    rows: list[str] = []
    for match in re.finditer(r"\braises?\s+questions(?:\s+about)?\b", text, flags=re.I):
        phrase = _clean_phrase(match.group(0))
        rows.append(phrase)
        if phrase.casefold().endswith(" about"):
            rows.append(phrase.rsplit(" ", 1)[0])
    return rows


def _demonstrative_consequence_phrases(text: str) -> list[str]:
    pattern = re.compile(
        r"\b(?:this|that|it)\s+makes?\s+[^.!?]{1,80}?\s+(?:more\s+)?(?:difficult|harder|hard)\b",
        flags=re.I,
    )
    return [_clean_phrase(match.group(0)) for match in pattern.finditer(text)]


def _clean_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" .,;:")


def planner_agent_profile(source_text: str = "") -> dict[str, Any]:
    return {
        "role": "Planner agent",
        "mission": "Convert scanner findings and source text into a paragraph route that helps Writer reduce findings without copying risky source shapes.",
        "input_boundary": [
            "Scanner findings are symptoms, not the final rewrite plan.",
            "Source sentence preserve_terms are meaning anchors, not automatic wording requirements.",
            "Risky source shapes must be converted into route instructions before Writer sees them.",
        ],
        "operating_procedure": [
            "Classify the paragraph movement before listing any terms.",
            "Identify risky source shapes using basic_rules and put them into do_not_copy_route.",
            "Build 2-4 flow_plan steps that describe paragraph movement, not sentence repairs.",
            "Remove risky wording terms from must_include while preserving their meaning in paragraph_route.",
            "Decide whether Author-proxy is required only after checking whether local source terms can ground the claim.",
        ],
        "output_contract": [
            "flow_plan must tell Writer what each paragraph beat does.",
            "validated_construction_route must give Writer ordered construction logic.",
            "coverage_terms must list meaning that survives, not risky wording to copy.",
            "do_not_copy_route must include risky exact source shapes and machine routes.",
            "author_proxy_request must be false for style-only problems.",
        ],
        "uses_basic_rules_to": [
            "identify risky prose shapes before Writer sees them",
            "separate meaning-to-preserve from wording-to-copy",
            "turn sentence findings into paragraph strategy",
            "define actor-action movement instead of generic topic movement",
        ],
        "must_do": [
            "Put risky source shapes into do_not_copy_route.",
            "Keep risky source terms out of flow_plan.must_include.",
            "Give Writer an ordered route from source condition to cause or contrast to consequence.",
            "Preserve source meaning, modality, and closing consequence.",
        ],
        "must_not_do": [
            "Do not hand Writer a list of scanner findings to fix one by one.",
            "Do not force Writer to preserve risky source openings or vague good-use templates.",
            "Do not add causes, examples, or policy claims that are not in the selected paragraph or approved Author-proxy pack.",
        ],
        "self_check_before_return": [
            "No risky source term appears in flow_plan.must_include.",
            "No flow_plan step is one sentence finding rewritten as an instruction.",
            "No source-side uncertainty has been hardened.",
            "No neighbor idea is imported unless Author-proxy is required or packed.",
        ],
        "failure_modes_to_prevent": [
            "Writer receives contradictory instructions: copy risky phrase and avoid risky phrase.",
            "Writer receives a catalogue of terms instead of a route.",
            "Writer has to infer the actor for an abstract consequence.",
            "Writer repeats the same consequence because Planner did not order the route.",
        ],
        "risky_source_shapes": risky_source_shape_guidance(source_text),
    }


def writer_agent_profile() -> dict[str, Any]:
    return {
        "role": "Writer agent",
        "mission": "Realize the Planner route as normal paragraph prose that reduces findings while preserving source meaning.",
        "input_boundary": [
            "Planner route is the operating plan.",
            "Source text provides meaning, modality, and required anchors.",
            "Rules and profile labels are not content to insert into prose.",
            "Author-proxy material is reviewable grounding only when Planner has integrated it.",
        ],
        "operating_procedure": [
            "Follow the Planner paragraph_route before choosing sentence wording.",
            "Use do_not_copy_route to avoid risky source shapes.",
            "Write actor-action sentences for abstract consequences.",
            "Show visible cause, contrast, or behaviour before the consequence.",
            "Keep source modality such as may, can, not always, no longer, and not only attached to the same idea.",
            "Self-reject any variant that repeats a consequence, appends leftover terms, or turns a rule label into prose.",
        ],
        "output_contract": [
            "Return complete paragraph variants only.",
            "Each variant must preserve source meaning and required anchors.",
            "Each variant must attempt scanner movement through route change, not synonym swaps.",
            "Any reviewable bridge must be listed in author_review_items.",
        ],
        "uses_basic_rules_to": [
            "choose sentence jobs and entry points",
            "replace generic openings with concrete pressure or action",
            "show visible behaviour before abstract consequence",
            "keep uncertainty and contrast attached to the same source idea",
        ],
        "must_do": [
            "Follow Planner route over original sentence order when the Planner marks a risky shape.",
            "Use actor-action wording for abstract consequences.",
            "Show source risks through visible behaviour where the source supports that move.",
            "Write the closing consequence once, after the behaviour or condition that causes it.",
        ],
        "must_not_do": [
            "Do not copy risky source phrases from do_not_copy_route.",
            "Do not announce a broad topic opener when a source-specific tension is available.",
            "Do not treat profile terms or rule labels as content to insert.",
            "Do not repeat the same consequence in multiple sentences.",
        ],
        "self_check_before_return": [
            "No variant starts by announcing the broad topic when a specific pressure/action is available.",
            "No variant repeats the same consequence.",
            "No variant places the closing consequence before the source condition that causes it.",
            "No variant leaves a source term dangling at the end of an unrelated sentence.",
            "No variant uses over-formal substitutes where plain source words work.",
        ],
        "failure_modes_to_prevent": [
            "A variant satisfies coverage by appending leftover terms.",
            "A variant turns rules into generic polished prose.",
            "A variant lowers mean risk but increases findings through packed or repeated sentence shapes.",
            "A variant preserves the risky source route with minor synonyms.",
        ],
        "basic_rules": consolidated_prose_repair_rules(),
    }


def selector_agent_profile() -> dict[str, Any]:
    return {
        "role": "Selector agent",
        "mission": "Select only existing Writer variants that satisfy diagnostics and the same prose contract Planner and Writer used.",
        "input_boundary": [
            "Selector must not write or edit prose.",
            "Selector can only choose among diagnostics-eligible variants shown in the prompt.",
            "Scanner metrics are primary after hard blockers have removed corrupted candidates.",
        ],
        "operating_procedure": [
            "Reject candidates with blockers before considering style.",
            "Among eligible candidates, choose lowest candidate_findings first.",
            "If findings tie, choose lowest candidate_mean_risk.",
            "If still tied, choose highest risk_drop.",
            "Use basic_rules only as tie-break evidence, not as permission to override metrics.",
        ],
        "failure_modes_to_prevent": [
            "Choosing a smoother candidate with worse findings.",
            "Rewarding generic polish over scanner movement.",
            "Selecting a candidate that repeats a risky source shape because it reads fluent.",
        ],
        "basic_rules": consolidated_prose_repair_rules(),
    }


def _rule(problem: str, simple_rule: str, preferred_shape: str) -> dict[str, str]:
    return {
        "problem": problem,
        "simple_rule": simple_rule,
        "preferred_shape": preferred_shape,
    }


def _dedupe(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows
