from __future__ import annotations

import re

from .compiler_lists import _article_phrase, _capitalize, _clean_item, _has_packed_list, _split_parts, _unpack_sentence
from .text import source_terms


def _needs_context_rebuild(operation: str) -> bool:
    lowered = operation.casefold()
    return any(marker in lowered for marker in ("context", "source-to-claim", "reviewable", "broad claim"))

def _strip_structural_suffix(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"\s+in\s+summary[.]?$", "", stripped, flags=re.I)
    stripped = re.sub(r"\s+in\s+additional[.]?$", "", stripped, flags=re.I)
    return stripped

def _rebuild_context_route(text: str, previous_text: str) -> str:
    stripped = _strip_structural_suffix(text)
    lowered = stripped.casefold()
    preposed_this = re.match(r"^((?:during|since|when|while|after|before)\s+[^,]{1,60}),\s+this\s+(.+)$", stripped, flags=re.I)
    if preposed_this:
        return f"{preposed_this.group(1).strip()}, {preposed_this.group(2).strip()}"
    because_match = re.match(r"^this\s+is\s+(?:a|an)\s+[^.!?]{1,50}?\s+because\s+(.+)$", stripped, flags=re.I)
    if because_match:
        return because_match.group(1).strip()
    made_important = re.match(r"^this\s+\w+\s+has\s+made\s+(.+?)\s+(?:even\s+)?more\s+important[.]?$", stripped, flags=re.I)
    if made_important:
        return f"{made_important.group(1).strip()} carries more of the work"
    made_important_contrast = re.match(r"^this\s+\w+\s+has\s+made\s+(.+?)\s+(?:even\s+)?more\s+important\b.*$", stripped, flags=re.I)
    if made_important_contrast:
        return f"{made_important_contrast.group(1).strip()} carries more of the work"
    important_be = re.match(r"^(.+?)\s+(is|was|are|were)\s+(an?\s+)?important\s+(.+)$", stripped, flags=re.I)
    if important_be:
        article = important_be.group(3) or ""
        return f"{important_be.group(1).strip()} {important_be.group(2)} {article}{important_be.group(4).strip()}"
    stripped = re.sub(r"\bserious\s+", "", stripped, flags=re.I)
    stripped = re.sub(r"\bnever\s+", "does not ", stripped, flags=re.I)
    for_is_priority = re.match(r"^for\s+is\s+more\s+important\s+to\s+(.+?)\s+than\s+(.+)$", stripped, flags=re.I)
    if for_is_priority:
        return f"The priority is to {for_is_priority.group(1).strip()}. The contrast is {for_is_priority.group(2).strip()}"
    for_me_priority = re.match(r"^for\s+me\s+is\s+more\s+important\s+to\s+(.+?)\s+than\s+(.+)$", stripped, flags=re.I)
    if for_me_priority:
        return f"For me, the priority is to {for_me_priority.group(1).strip()}. The contrast is {for_me_priority.group(2).strip()}"
    should_goal = re.match(r"^(.+?)\s+should\s+(not\s+)?be\s+to\s+(.+)$", stripped, flags=re.I)
    if should_goal:
        negation = "not " if should_goal.group(2) else ""
        return f"{should_goal.group(1).strip()} is {negation}to {should_goal.group(3).strip()}"
    should_not_misunderstood = re.match(r"^(.+?)\s+should\s+not\s+be\s+misunderstood\s+as\s+(.+)$", stripped, flags=re.I)
    if should_not_misunderstood:
        return f"{should_not_misunderstood.group(1).strip()} is not the same as {should_not_misunderstood.group(2).strip()}"
    either = re.match(r"^it\s+can\s+either\s+(.+)$", stripped, flags=re.I)
    if either:
        return f"One option is to {either.group(1).strip()}"
    it_can = re.match(r"^it\s+can\s+(.+)$", stripped, flags=re.I)
    if it_can:
        return f"Another option is to {it_can.group(1).strip()}"
    it_comes_from = re.match(r"^it\s+comes\s+from\s+(.+)$", stripped, flags=re.I)
    if it_comes_from:
        return f"The source is {it_comes_from.group(1).strip()}"
    it_also_faces = re.match(r"^it\s+also\s+faces\s+(.+)$", stripped, flags=re.I)
    if it_also_faces:
        pressure = it_also_faces.group(1).strip()
        related = re.match(r"^challenges?\s+related\s+to\s+(.+)$", pressure, flags=re.I)
        return f"The remaining pressure relates to {related.group(1).strip()}" if related else f"The remaining pressure is {pressure}"
    acceptable = re.match(r"^it\s+is\s+acceptable\s+to\s+(.+?)\s+as\s+(.+)$", stripped, flags=re.I)
    if acceptable:
        return f"{acceptable.group(2).strip()} makes the idea acceptable to {acceptable.group(1).strip()}"
    it_often_begins = re.match(r"^it\s+often\s+begins\s+when\s+(.+)$", stripped, flags=re.I)
    if it_often_begins:
        return f"The starting point often appears when {it_often_begins.group(1).strip()}"
    it_made = re.match(r"^it\s+made\s+clear\s+(.+)$", stripped, flags=re.I)
    if it_made:
        return f"The result made clear {it_made.group(1).strip()}"
    it_is = re.match(r"^it\s+is\s+(.+)$", stripped, flags=re.I)
    subject = _previous_subject(previous_text)
    if it_is and subject:
        return f"{subject} is {it_is.group(1).strip()}"
    it_have = re.match(r"^it\s+(has|have|had)\s+(.+)$", stripped, flags=re.I)
    if it_have and subject:
        return f"{subject} {it_have.group(1).strip()} {it_have.group(2).strip()}"
    they = re.match(r"^they\s+(.+)$", stripped, flags=re.I)
    if they and subject:
        return f"{subject} {they.group(1).strip()}"
    they_still_lack = re.match(r"^they\s+still\s+lack\s+(.+)$", stripped, flags=re.I)
    if they_still_lack:
        return f"The remaining gap is {they_still_lack.group(1).strip()}"
    they_search_match = re.match(r"^they\s+search\s+for\s+match\s+(.+)$", stripped, flags=re.I)
    if they_search_match:
        return f"The search is to match {they_search_match.group(1).strip()}"
    if lowered.startswith(("this ", "that ", "it ")):
        return _shift_predictable_start(stripped)
    return stripped

def _revoice_smoothing_route(text: str) -> str:
    stripped = _strip_structural_suffix(text)
    normalized = re.sub(r"\bparticipate\s+demonstrate\b", "participate and demonstrate", stripped, flags=re.I)
    if normalized != stripped:
        stripped = normalized
    heading_task = re.match(r"^([A-Z][A-Za-z'-]{2,})\s+to\s+([A-Za-z]+)\s+(.+)$", stripped)
    if heading_task:
        subject = heading_task.group(1).strip()
        verb = _third_person_present(heading_task.group(2).strip())
        return f"The {subject.casefold()} {verb} {heading_task.group(3).strip()}"
    malformed_direct = re.match(r"^the\s+(directly\s+affects|affects)\s+(.+)$", stripped, flags=re.I)
    if malformed_direct:
        return f"The result {malformed_direct.group(1).strip()} {malformed_direct.group(2).strip()}"
    because_it_gives = re.match(r"^(.+?)\s+because\s+it\s+gives\s+(.+?)\s+another\s+way\s+to\s+(.+)$", stripped, flags=re.I)
    if because_it_gives:
        recipient = _capitalize(because_it_gives.group(2).strip())
        return (
            f"{because_it_gives.group(1).strip()}. "
            f"{recipient} get another way to {because_it_gives.group(3).strip()}"
        )
    attributed = re.match(r"^(.+?)\s+is\s+attributed\s+to\s+(.+)$", stripped, flags=re.I)
    if attributed:
        subject = re.sub(r"^\s*the\s+main\s+", "The ", attributed.group(1).strip(), flags=re.I)
        return f"{attributed.group(2).strip()} explains {subject}"
    for_is_priority = re.match(r"^for\s+is\s+more\s+important\s+to\s+(.+?)\s+than\s+(.+)$", stripped, flags=re.I)
    if for_is_priority:
        goal = for_is_priority.group(1).strip(" .")
        goal_parts = [_clean_item(part) for part in _split_parts(goal)]
        priority_parts: list[str] = []
        if len(goal_parts) > 1 and " how to " in goal_parts[0].casefold():
            prefix = re.sub(r"\s+\S+$", "", goal_parts[0]).strip()
            priority_parts = [f"The priority is to {goal_parts[0]}."] + [
                f"The priority is also to {prefix} {part.strip(' .')}." for part in goal_parts[1:]
            ]
        priority = f"The priority is to {goal}"
        priority_parts = priority_parts or (_unpack_sentence(priority) if (_has_packed_list(priority) or len(_split_parts(priority)) > 1) else [])
        priority_text = " ".join(priority_parts or [priority])
        separator = "" if priority_text.rstrip().endswith(".") else "."
        return f"{priority_text}{separator} The contrast is {for_is_priority.group(2).strip()}"
    cannot_simply = re.match(r"^(.+?)\s+cannot\s+(?:truly\s+)?master\s+(.+?)\s+simply\s+by\s+(.+)$", stripped, flags=re.I)
    if cannot_simply:
        route = cannot_simply.group(3).strip(" .")
        subject = cannot_simply.group(1).strip()
        skill = cannot_simply.group(2).strip()
        verb = "are" if subject.casefold().endswith("s") else "is"
        return f"{subject} {verb} working on {skill}. {route.capitalize()} alone does not build mastery"
    not_only_about = re.match(r"^the\s+example\s+demonstrates\s+that\s+(.+?)\s+is\s+not\s+only\s+about\s+(.+)$", stripped, flags=re.I)
    if not_only_about:
        return f"In the example, {not_only_about.group(1).strip()} goes beyond {not_only_about.group(2).strip()}"
    carries_pressure = re.match(r"^the\s+example\s+demonstrates\s+that\s+(.+?)\s+carries\s+this\s+pressure:\s+not\s+only\s+about\s+(.+)$", stripped, flags=re.I)
    if carries_pressure:
        return f"In the example, {carries_pressure.group(1).strip()} goes beyond {carries_pressure.group(2).strip()}"
    need_compare = re.match(r"^(.+?)\s+consists\s+of\s+(.+?)\s+who\s+need\s+to\s+(.+?)\s+and\s+compar(?:e|ed)\s+(.+)$", stripped, flags=re.I)
    if need_compare:
        group = need_compare.group(2).strip()
        action = re.sub(r"^watch\s+", "", need_compare.group(3).strip(), flags=re.I)
        return (
            f"{need_compare.group(1).strip()} consists of {group}. "
            f"Learners in this group watch {action}. "
            f"Learners in this group compare {need_compare.group(4).strip()}"
        )
    and_compared = re.match(r"^(.+?)\s+and\s+compared\s+(.+)$", stripped, flags=re.I)
    if and_compared:
        return f"{and_compared.group(1).strip()} and compare {and_compared.group(2).strip()}"
    feel_confident_because = re.match(r"^(.+?)\s+feel\s+confident\s+because\s+they\s+are\s+primarily\s+working\s+with\s+(.+)$", stripped, flags=re.I)
    if feel_confident_because:
        working = feel_confident_because.group(2).strip(" .")
        return f"Working with {working} makes {feel_confident_because.group(1).strip()} feel confident"
    in_intervene = re.match(r"^(.+?)\s+in\s+intervene\s+(.+)$", stripped, flags=re.I)
    if in_intervene:
        fixed = f"{in_intervene.group(1).strip()} to intervene {in_intervene.group(2).strip()}"
        level_split = re.match(r"^(.+?\s+at\s+.+?)\s+and\s+(.+?)\s+by\s+(.+)$", fixed, flags=re.I)
        if level_split:
            return f"{level_split.group(1).strip()}. The next level is {level_split.group(2).strip()}. Observation happens by {level_split.group(3).strip()}"
        return fixed
    responsibility = re.match(r"^(.+?)\s+is\s+the\s+primary\s+responsibility\s+of\s+(.+)$", stripped, flags=re.I)
    if responsibility:
        owner = responsibility.group(2).strip(" .")
        return f"{owner} is responsible for {responsibility.group(1).strip()}"
    incorporated_to = re.match(r"^(.+?)\s+have\s+incorporated\s+(.+?)\s+into\s+(.+?)\s+to\s+(.+)$", stripped, flags=re.I)
    if incorporated_to:
        return (
            f"{incorporated_to.group(1).strip()} incorporated {incorporated_to.group(2).strip()}. "
            f"The unit is {incorporated_to.group(3).strip()}. "
            f"The simulation purpose is to {incorporated_to.group(4).strip()}"
        )
    compared_challenge = re.match(r"^(.+?)\s+compared\s+to\s+(.+?),\s+the\s+(?:greater\s+)?challenge\s+today\s+is\s+how\s+to\s+(.+)$", stripped, flags=re.I)
    if compared_challenge:
        return f"Compared to {compared_challenge.group(2).strip()}, {compared_challenge.group(1).strip()} now has to {compared_challenge.group(3).strip()}"
    leading_compared_challenge = re.match(r"^compared\s+to\s+(.+?),\s+the\s+(?:greater\s+)?challenge\s+today\s+is\s+how\s+to\s+(.+)$", stripped, flags=re.I)
    if leading_compared_challenge:
        return f"Compared to {leading_compared_challenge.group(1).strip()}, the work now has to {leading_compared_challenge.group(2).strip()}"
    transition_from = re.match(r"^the\s+transition\s+starts\s+from\s+(.+?)\s+to\s+(.+)$", stripped, flags=re.I)
    if transition_from:
        return f"The transition starts from {transition_from.group(1).strip()}. The next step is {transition_from.group(2).strip()}"
    conclusion_prefix = re.match(r"^conclusion\s+(.+)$", stripped, flags=re.I)
    if conclusion_prefix:
        return conclusion_prefix.group(1).strip()
    responsible_should = re.match(r"^is\s+responsible\s+for\s+(.+?)\s+should\s+(.+)$", stripped, flags=re.I)
    if responsible_should:
        return f"{responsible_should.group(1).strip()} is responsible to {responsible_should.group(2).strip()}"
    is_can_measure = re.match(r"^(.+?)\s+is\s+(.+?)\s+can\s+measure\s+(.+)$", stripped, flags=re.I)
    if is_can_measure:
        subject_terms = source_terms(is_can_measure.group(1), limit=1)
        subject = subject_terms[0] if subject_terms else "The subject"
        return f"{is_can_measure.group(1).strip()} is {is_can_measure.group(2).strip()}. {subject} can measure {is_can_measure.group(3).strip()}"
    followed_up = re.match(r"^(.+?)\s+works\s+best\s+with\s+(.+?)\s+that\s+is\s+followed\s+up\s+with\s+(.+)$", stripped, flags=re.I)
    if followed_up:
        return f"{followed_up.group(1).strip()} works best with {followed_up.group(2).strip()}. Follow-up comes through {followed_up.group(3).strip()}"
    proceed_to = re.match(r"^(.+?)\s+will\s+proceed\s+from\s+(.+?)\s+to\s+(.+)$", stripped, flags=re.I)
    if proceed_to:
        return f"{proceed_to.group(1).strip()} uses {proceed_to.group(2).strip()} to {proceed_to.group(3).strip()}"
    proceed_from = re.match(r"^(.+?)\s+will\s+proceed\s+from\s+(.+)$", stripped, flags=re.I)
    if proceed_from:
        return f"{proceed_from.group(1).strip()} uses {proceed_from.group(2).strip()}"
    must_ensure = re.match(r"^(.+?)\s+must\s+ensure\s+that\s+(.+)$", stripped, flags=re.I)
    if must_ensure:
        return f"{must_ensure.group(1).strip()} checks that {must_ensure.group(2).strip()}"
    accompanied = re.match(r"^(.+?)\s+is\s+best\s+accompanied\s+by\s+(.+)$", stripped, flags=re.I)
    if accompanied:
        return f"{accompanied.group(1).strip()} works best with {accompanied.group(2).strip()}"
    result_of = re.match(r"^(.+?)\s+is\s+(?:typically\s+)?a\s+result\s+of\s+(.+)$", stripped, flags=re.I)
    if result_of:
        return f"{result_of.group(2).strip()} typically results in {result_of.group(1).strip()}"
    aspect = re.match(r"^(.+?)\s+is\s+an\s+aspect\s+of\s+determining\s+(.+)$", stripped, flags=re.I)
    if aspect:
        return f"{aspect.group(1).strip()} helps determine {aspect.group(2).strip()}"
    used_by = re.match(r"^(.+?)\s+will\s+be\s+used\s+by\s+(.+?)\s+to\s+(.+)$", stripped, flags=re.I)
    if used_by:
        return f"{used_by.group(2).strip()} will use {_clean_item(used_by.group(1))} to {used_by.group(3).strip()}"
    can_be_used = re.match(r"^(.+?)\s+can\s+be\s+(.+?)\s+used\s+to\s+(.+)$", stripped, flags=re.I)
    if can_be_used:
        qualifier = can_be_used.group(2).strip()
        middle = f" {qualifier}" if qualifier else ""
        action = re.sub(r"^evaluate\s+", "", can_be_used.group(3).strip(), flags=re.I)
        return f"{can_be_used.group(1).strip()} can{middle} evaluate {action}"
    transition_question = re.match(r"^how\s+to\s+transition\s+from\s+(.+)$", stripped, flags=re.I)
    if transition_question:
        return f"The transition starts from {transition_question.group(1).strip()}"
    interpret_combine = re.match(r"^(.+?)\s+to\s+interpret\s+combine\s+(.+)$", stripped, flags=re.I)
    if interpret_combine:
        return f"{interpret_combine.group(1).strip()} to interpret the material and combine {interpret_combine.group(2).strip()}"
    return ""

def _previous_subject(text: str) -> str:
    match = re.match(
        r"^(.{1,60}?)\s+(?:is|are|was|were|has|have|can|may|might|must|should|would|will|\w+s)\b",
        text.strip(),
        flags=re.I,
    )
    if not match:
        return ""
    subject = match.group(1).strip(" .,:;")
    if not subject or subject.casefold().startswith(("this", "that", "it", "they", "these", "those")):
        return ""
    return subject

def _remove_leading_discourse(text: str) -> str:
    normalized = re.sub(r"^today'?s\s+", "The ", text.replace("’", "'"), flags=re.I)
    in_addition = re.match(r"^in\s+addition(?:al)?(?:\s+to\s+(.+))?[.]?$", normalized, flags=re.I)
    if in_addition:
        return (in_addition.group(1) or "").strip()
    markers = (
        r"today'?s|today|now|however|therefore|moreover|furthermore|additionally|secondly|meanwhile|overall|hence|next|lastly|finally|"
        r"because\s+of\s+this|in\s+other\s+words|for\s+this\s+reason|as\s+a\s+result|"
        r"used\s+well|but"
    )
    return re.sub(rf"^({markers}),?\s+", "", normalized, flags=re.I).strip()

def _rotate_be_claim(text: str) -> str:
    match = re.match(r"^(.+?)\s+is\s+(.+?)\.$", text, flags=re.I)
    if not match:
        return text
    subject = _article_phrase(match.group(1).strip())
    rest = match.group(2).strip()
    return f"{subject} carries this pressure: {rest}."

def _shift_predictable_start(text: str) -> str:
    text = _strip_structural_suffix(text)
    it_begins = re.match(r"^it\s+begins\s+in\s+(.+)$", text, flags=re.I)
    if it_begins:
        return f"The starting point is {it_begins.group(1).strip()}"
    it_also_faces = re.match(r"^it\s+also\s+faces\s+(.+)$", text, flags=re.I)
    if it_also_faces:
        pressure = it_also_faces.group(1).strip()
        related = re.match(r"^challenges?\s+related\s+to\s+(.+)$", pressure, flags=re.I)
        return f"The remaining pressure relates to {related.group(1).strip()}" if related else f"The remaining pressure is {pressure}"
    it_comes_from = re.match(r"^it\s+comes\s+from\s+(.+)$", text, flags=re.I)
    if it_comes_from:
        return f"The source is {it_comes_from.group(1).strip()}"
    there_may = re.match(r"^there\s+may\s+be\s+(.+)$", text, flags=re.I)
    if there_may:
        return f"{_article_phrase(there_may.group(1).strip())} may remain"
    it_result_verb = re.match(r"^it\s+((?:directly\s+)?affects|creates|develops|encourages|supports|limits)\s+(.+)$", text, flags=re.I)
    if it_result_verb:
        return f"The result {it_result_verb.group(1).strip()} {it_result_verb.group(2).strip()}"
    shifted = re.sub(r"^(this|that)\s+(can|could|may|might|must|should|would|will|is|has|makes|made)\s+", r"The result \2 ", text, flags=re.I)
    shifted = re.sub(r"^(this|that)\s+([a-z][\w'-]+)\s+", _replace_this_that_subject, shifted, flags=re.I)
    shifted = re.sub(r"^(these|those)\s+([a-z][\w'-]+)\s+", r"The \2 ", shifted, flags=re.I)
    shifted = re.sub(r"^it\s+(can|could|may|might|must|should|would|will|is|has|makes|made)\s+", r"The option \1 ", shifted, flags=re.I)
    return shifted

def _replace_this_that_subject(match: re.Match[str]) -> str:
    word = match.group(2)
    if word.casefold() in {"shows", "demonstrates", "suggests", "means", "creates", "falls"}:
        return f"The point {word} "
    return f"The {word} "

def _third_person_present(stem: str) -> str:
    base = {"outlin": "outline", "describ": "describe", "creat": "create"}.get(stem.casefold(), stem)
    return base + ("es" if base.endswith(("s", "sh", "ch", "x", "z")) else "s")
