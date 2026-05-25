from __future__ import annotations

from .compiler_lists import _capitalize, _dedupe, _has_packed_list, _period, _split_existing_sentences, _unpack_sentence
from .compiler_routes import (
    _needs_context_rebuild,
    _rebuild_context_route,
    _remove_leading_discourse,
    _revoice_smoothing_route,
    _rotate_be_claim,
    _shift_predictable_start,
)
from .compiler_splits import (
    _has_contrast,
    _reframe_author_year_claim,
    _split_because_wrapper,
    _split_because_clause,
    _split_colon_claim,
    _split_comma_pronoun_result,
    _split_comma_subject_clause,
    _split_creating_clause,
    _split_from_who_clause,
    _split_attribution,
    _split_and_has_begun,
    _split_citation_relation,
    _split_contrast,
    _split_dash_clause,
    _split_during_gerund_context,
    _split_if_then,
    _split_including_clause,
    _split_involves_represents,
    _split_is_modal_predicate,
    _split_heading_merge,
    _split_and_then_clause,
    _split_if_comma,
    _split_leading_result,
    _split_not_but,
    _split_not_only,
    _split_not_only_subject,
    _split_or_modal,
    _split_prepositional_fragment,
    _split_preposed_context_clause,
    _split_provides_through,
    _split_rather_than_clause,
    _split_rather_than_no_comma,
    _split_relative_used_to,
    _split_report_claim,
    _split_semicolon_claim,
    _split_showing_clause,
    _split_then_clause,
    _split_thereby_clause,
    _split_when_clause,
    _split_while_clause,
    _split_where_clause,
    _split_which_clause,
)
from .plan import Plan
from .text import Paragraph


def compile_plain_text(paragraph: Paragraph, plan: Plan) -> str:
    operations = {action.sentence_id: action.operation for action in plan.actions}
    sentences: list[str] = []
    previous = ""
    for sentence in paragraph.sentences:
        rewritten = _rewrite_sentence(sentence.text, operations.get(sentence.id, ""), previous_text=previous)
        sentences.extend(rewritten)
        previous = rewritten[-1] if rewritten else sentence.text
    return " ".join(_dedupe(sentences))

def _rewrite_sentence(text: str, operation: str, *, previous_text: str = "") -> list[str]:
    source = " ".join(str(text or "").split())
    if not operation.strip():
        return [_period(_capitalize(source))] if source else []
    clean = _remove_leading_discourse(source)
    if not clean:
        return []
    if _needs_context_rebuild(operation):
        context_rebuilt = _rebuild_context_route(clean, previous_text)
        if context_rebuilt:
            clean = context_rebuilt
    revoiced = _revoice_smoothing_route(clean)
    if revoiced:
        clean = revoiced
        boundary_split = _split_existing_sentences(clean)
        if boundary_split:
            return boundary_split
    if "opener" in operation:
        clean = _rotate_be_claim(clean)
        clean = _shift_predictable_start(clean)
    dash_split = _split_dash_clause(clean)
    if dash_split:
        return dash_split
    author_year_claim = _reframe_author_year_claim(clean)
    if author_year_claim:
        return author_year_claim
    report_claim = _split_report_claim(clean)
    if report_claim:
        return report_claim
    citation_split = _split_citation_relation(clean)
    if citation_split:
        return citation_split
    preposed_context = _split_preposed_context_clause(clean)
    if preposed_context:
        return preposed_context
    preposition_fragment = _split_prepositional_fragment(clean)
    if preposition_fragment:
        return preposition_fragment
    during_gerund = _split_during_gerund_context(clean)
    if during_gerund:
        return during_gerund
    heading_merge = _split_heading_merge(clean)
    if heading_merge:
        return heading_merge
    because_wrapper = _split_because_wrapper(clean)
    if because_wrapper:
        return because_wrapper
    where_clause = _split_where_clause(clean)
    if where_clause:
        return where_clause
    rather_than_split = _split_rather_than_clause(clean)
    if rather_than_split:
        return rather_than_split
    comma_pronoun = _split_comma_pronoun_result(clean)
    if comma_pronoun:
        return comma_pronoun
    showing_clause = _split_showing_clause(clean)
    if showing_clause:
        return showing_clause
    then_clause = _split_then_clause(clean)
    if then_clause:
        return then_clause
    not_only_subject = _split_not_only_subject(clean)
    if not_only_subject:
        return not_only_subject
    thereby_clause = _split_thereby_clause(clean)
    if thereby_clause:
        return thereby_clause
    creating_clause = _split_creating_clause(clean)
    if creating_clause:
        return creating_clause
    comma_subject = _split_comma_subject_clause(clean)
    if comma_subject:
        return comma_subject
    which_clause = _split_which_clause(clean)
    if which_clause:
        return which_clause
    because_clause = _split_because_clause(clean)
    if because_clause:
        return because_clause
    from_who = _split_from_who_clause(clean)
    if from_who:
        return from_who
    not_only_split = _split_not_only(clean)
    if not_only_split:
        return not_only_split
    attribution_split = _split_attribution(clean)
    if attribution_split:
        return attribution_split
    leading_split = _split_leading_result(clean)
    if leading_split:
        return leading_split
    if_then_split = _split_if_then(clean)
    if if_then_split:
        return if_then_split
    if_comma_split = _split_if_comma(clean)
    if if_comma_split:
        return if_comma_split
    semicolon_split = _split_semicolon_claim(clean)
    if semicolon_split:
        return semicolon_split
    colon_split = _split_colon_claim(clean)
    if colon_split:
        return colon_split
    if _has_contrast(clean):
        return _split_contrast(clean)
    not_but_split = _split_not_but(clean)
    if not_but_split:
        return not_but_split
    rather_than_no_comma = _split_rather_than_no_comma(clean)
    if rather_than_no_comma:
        return rather_than_no_comma
    and_has_begun = _split_and_has_begun(clean)
    if and_has_begun:
        return and_has_begun
    and_then = _split_and_then_clause(clean)
    if and_then:
        return and_then
    involves_split = _split_involves_represents(clean)
    if involves_split:
        return involves_split
    provides_split = _split_provides_through(clean)
    if provides_split:
        return provides_split
    relative_used = _split_relative_used_to(clean)
    if relative_used:
        return relative_used
    is_modal_split = _split_is_modal_predicate(clean)
    if is_modal_split:
        return is_modal_split
    or_modal_split = _split_or_modal(clean)
    if or_modal_split:
        return or_modal_split
    when_split = _split_when_clause(clean)
    if when_split:
        return when_split
    while_split = _split_while_clause(clean)
    if while_split:
        return while_split
    including_split = _split_including_clause(clean)
    if including_split:
        return including_split
    if _has_packed_list(clean):
        unpacked = _unpack_sentence(clean)
        if unpacked:
            return unpacked
    if "“" in clean or "”" in clean:
        return [_period(_capitalize(clean))]
    return [_period(_capitalize(clean))]
