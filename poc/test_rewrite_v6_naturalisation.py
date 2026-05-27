from __future__ import annotations

import json

from poc.rewrite_v6 import pipeline as v6_pipeline
from poc.rewrite_v6.naturalisation import (
    NaturalisationOperation,
    _naturalisation_candidates,
    apply_naturalisation_operations,
    run_naturalisation_repair,
)
from poc.rewrite_v6.pipeline import run_v6_rewrite_all
from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.scan import scan_text


class StaticJsonResponse:
    def __init__(self, content: str):
        self.content = content
        self.raw_content = content


class SequencedQualityClient:
    model = "grammer-test"

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls = 0
        self.app_labels: list[str | None] = []

    def chat(self, _prompt, **kwargs):
        self.calls += 1
        self.app_labels.append(kwargs.get("app_label"))
        return StaticJsonResponse(json.dumps(self.responses.pop(0)))


def test_naturalisation_adds_deterministic_candidate_for_three_repeated_starts():
    text = (
        "Teachers guide students to question sources. "
        "Teachers compare viewpoints. "
        "Teachers help students develop judgment."
    )
    client = SequencedQualityClient([{"operations": []}])

    result = run_naturalisation_repair(text, client=client)

    assert result.repaired_text == "Teachers guide students to question sources, compare viewpoints, and help students develop judgment."
    assert [operation.reason for operation in result.operations] == ["repeated_subject_start"]


def test_naturalisation_adds_deterministic_candidate_for_two_repeated_starts():
    text = "The model still exists. The model no longer reflects current learning."
    client = SequencedQualityClient([{"operations": []}])

    result = run_naturalisation_repair(text, client=client)

    assert result.repaired_text == "The model still exists but no longer reflects current learning."


def test_naturalisation_adds_deterministic_passive_voice_candidate():
    text = "Knowledge was received by students from trusted sources."
    client = SequencedQualityClient([{"operations": []}])

    result = run_naturalisation_repair(text, client=client)

    assert result.repaired_text == "Students received knowledge from trusted sources."


def test_naturalisation_repeated_goal_uses_but_to_not_and_is_to():
    text = "The goal is not to reject technology. The goal is to help students learn."
    client = SequencedQualityClient([{"operations": []}])

    result = run_naturalisation_repair(text, client=client)

    assert result.repaired_text == "The goal is not to reject technology but to help students learn."


def test_naturalisation_applies_selective_repeated_subject_merge():
    text = (
        "Teachers guide students to question sources. "
        "Teachers compare viewpoints. "
        "Teachers help students develop judgment."
    )
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="Teachers guide students to question sources, compare viewpoints, and help students develop judgment.",
            reason="repeated_subject_start",
        )
    ])

    assert repaired == "Teachers guide students to question sources, compare viewpoints, and help students develop judgment."
    assert [operation.reason for operation in applied] == ["repeated_subject_start"]
    assert skipped == []


def test_naturalisation_rejects_single_sentence_polish_and_paragraph_boundary_change():
    text = "The model still exists. The model no longer reflects current learning.\n\nStable paragraph stays here."
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find="The model still exists.",
            replace="The older model still exists.",
            reason="paragraph_flow",
        ),
        NaturalisationOperation(
            find=text,
            replace=text.replace("\n\n", " "),
            reason="mechanical_decomposition",
        ),
    ])

    assert repaired == text
    assert applied == []
    assert {row["skip_reason"] for row in skipped} == {"single_sentence_polish", "paragraph_boundary_change"}


def test_naturalisation_rejects_short_sentence_merge_without_dependency_signal():
    text = "Teachers provide instruction. YouTube offers tutorials."
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="Teachers provide instruction and YouTube offers tutorials.",
            reason="short_sentence_chain",
        )
    ])

    assert repaired == text
    assert applied == []
    assert skipped[0]["skip_reason"] == "no_overrepair_signal"


def test_naturalisation_rejects_broad_same_first_word_merge():
    text = "Education systems change quickly. Education in the past used textbooks."
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="Education systems change quickly, and education in the past used textbooks.",
            reason="repeated_subject_start",
        )
    ])

    assert repaired == text
    assert applied == []
    assert skipped[0]["skip_reason"] == "no_overrepair_signal"


def test_naturalisation_rejects_new_semicolon_polish():
    text = "A teacher no longer merely delivers information. A good teacher helps students make sense of information."
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="A teacher no longer merely delivers information; a good teacher helps students make sense of information.",
            reason="repeated_subject_start",
        )
    ])

    assert repaired == text
    assert applied == []
    assert skipped[0]["skip_reason"] == "overpolish_punctuation"


def test_naturalisation_rejects_loose_three_sentence_dependent_merge():
    text = "They require standard answers. The pressure can encourage memorisation. It does not foster understanding."
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="They require standard answers, the pressure can encourage memorisation, and it does not foster understanding.",
            reason="dependent_short_sentence_chain",
        )
    ])

    assert repaired == text
    assert applied == []
    assert skipped[0]["skip_reason"] == "no_overrepair_signal"


def test_naturalisation_candidates_include_parallel_short_clusters_across_document():
    text = (
        "Opening sentence stays here.\n\n"
        "Teachers provide instruction. YouTube offers tutorials. TikTok shares bite-size lessons. "
        "Online courses deliver structured curricula.\n\n"
        "Individuals who can analyse are rewarded. People who can adapt receive rewards. "
        "Students who can communicate gain rewards as well."
    )

    candidates = _naturalisation_candidates(text)
    candidate_text = "\n".join(candidate.text for candidate in candidates)

    assert "Teachers provide instruction. YouTube offers tutorials. TikTok shares bite-size lessons." in candidate_text
    assert "Individuals who can analyse are rewarded. People who can adapt receive rewards." in candidate_text


def test_naturalisation_candidates_include_no_longer_rhetorical_ladder():
    text = (
        "Knowledge is no longer scarce. Access is no longer the biggest problem. "
        "The real challenge is knowing what is accurate and useful."
    )

    candidates = _naturalisation_candidates(text)

    assert any(candidate.reason == "rhetorical_ladder" and candidate.text == text for candidate in candidates)


def test_naturalisation_allows_light_rhetorical_ladder_repair():
    text = (
        "Knowledge is no longer scarce. Access is no longer the biggest problem. "
        "The real challenge is knowing what is accurate and useful."
    )
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="Knowledge is no longer scarce, and access is no longer the biggest problem. The real challenge is knowing what is accurate and useful.",
            reason="rhetorical_ladder",
        )
    ])

    assert repaired == "Knowledge is no longer scarce, and access is no longer the biggest problem. The real challenge is knowing what is accurate and useful."
    assert [operation.reason for operation in applied] == ["rhetorical_ladder"]
    assert skipped == []


def test_naturalisation_adds_deterministic_light_rhetorical_ladder_candidate():
    text = (
        "Knowledge is no longer scarce. Access is no longer the biggest problem. "
        "The real challenge is knowing what is accurate and useful."
    )
    client = SequencedQualityClient([{"operations": []}])

    result = run_naturalisation_repair(text, client=client)

    assert result.repaired_text == (
        "Knowledge is no longer scarce, and access is no longer the biggest problem. "
        "The real challenge is knowing what is accurate and useful."
    )
    assert result.operations[0].reason == "rhetorical_ladder"


def test_naturalisation_rejects_revoiced_rhetorical_ladder_repair():
    text = (
        "Knowledge is no longer scarce. Access is no longer the biggest problem. "
        "The real challenge is knowing what is accurate and useful."
    )
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="Knowledge is everywhere now, so access is not the same problem it used to be. The harder part is working out what is accurate and useful.",
            reason="rhetorical_ladder",
        )
    ])

    assert repaired == text
    assert applied == []
    assert skipped[0]["skip_reason"].startswith("content_term_dropped:")


def test_naturalisation_allows_parallel_short_list_sequence_with_preserved_terms():
    text = (
        "Teachers provide instruction. YouTube offers tutorials. TikTok shares bite-size lessons. "
        "Online courses deliver structured curricula."
    )
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="Teachers provide instruction while YouTube offers tutorials, TikTok shares bite-size lessons, and online courses deliver structured curricula.",
            reason="parallel_short_list_sequence",
        )
    ])

    assert repaired == "Teachers provide instruction while YouTube offers tutorials, TikTok shares bite-size lessons, and online courses deliver structured curricula."
    assert [operation.reason for operation in applied] == ["parallel_short_list_sequence"]
    assert skipped == []


def test_naturalisation_adds_deterministic_parallel_clause_candidate():
    text = (
        "Teachers provide instruction. YouTube offers tutorials. TikTok shares bite-size lessons. "
        "Online courses deliver structured curricula."
    )
    client = SequencedQualityClient([{"operations": []}])

    result = run_naturalisation_repair(text, client=client)

    assert result.repaired_text == (
        "Teachers provide instruction, YouTube offers tutorials, TikTok shares bite-size lessons, "
        "and online courses deliver structured curricula."
    )
    assert result.operations[-1].reason == "parallel_short_list_sequence"


def test_naturalisation_rejects_merge_that_drops_content_terms():
    text = (
        "The goal is to help students become thoughtful learners. "
        "Supporting responsible learners in a world full of information completes the goal."
    )
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="The goal is to help students become thoughtful and responsible learners.",
            reason="repeated_subject_start",
        )
    ])

    assert repaired == text
    assert applied == []
    assert skipped[0]["skip_reason"].startswith("content_term_dropped:")


def test_naturalisation_allows_inline_punctuation_linebreak_artifact():
    text = "Major companies such as Apple\n, Microsoft\n, Google\n, and Tesla\n have influenced society."
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="Major companies such as Apple, Microsoft, Google, and Tesla have influenced society.",
            reason="inline_punctuation_flow",
        )
    ])

    assert repaired == "Major companies such as Apple, Microsoft, Google, and Tesla have influenced society."
    assert [operation.reason for operation in applied] == ["inline_punctuation_flow"]
    assert skipped == []


def test_naturalisation_allows_awkward_passive_repair_without_general_polish():
    text = "Knowledge was received by students from trusted sources."
    repaired, applied, skipped = apply_naturalisation_operations(text, [
        NaturalisationOperation(
            find=text,
            replace="Students received knowledge from trusted sources.",
            reason="passive_voice",
        )
    ])

    assert repaired == "Students received knowledge from trusted sources."
    assert [operation.reason for operation in applied] == ["passive_voice"]
    assert skipped == []


def test_v6_runs_naturalisation_after_grammer_and_before_layout_restore(monkeypatch):
    source = "This process uses a form, a queue, and a review."
    rewritten = (
        "Teachers guide students to question sources. "
        "Teachers compare viewpoints. "
        "Teachers help students develop judgment."
    )
    naturalised = "Teachers guide students to question sources, compare viewpoints, and help students develop judgment."

    def fake_run(current, **_kwargs):
        scan = scan_text(current)
        paragraph, plan = build_plan(scan)
        return v6_pipeline.Result(scan=scan, plan=plan, variants=[], selected=None, rewritten_text=rewritten)

    client = SequencedQualityClient([
        {"operations": []},
        {"operations": [{
            "find": rewritten,
            "replace": naturalised,
            "reason": "repeated_subject_start",
        }]},
    ])
    monkeypatch.setattr(v6_pipeline, "run_v6_rewrite", fake_run)
    monkeypatch.setattr(v6_pipeline, "_acceptable_progress", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(v6_pipeline, "_cross_paragraph_regression", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(v6_pipeline, "_risk_safe_naturalisation_repair", lambda _current, repair: repair)

    result = run_v6_rewrite_all(source, quality_client=client, max_passes=1, residual_followup_passes=0)

    assert client.calls == 2
    assert client.app_labels == ["Grammer", "naturalisation"]
    assert result.rewritten_text == naturalised
    assert result.quality_repair is not None
    assert result.quality_repair.status == "no_changes"
    assert result.naturalisation_repair is not None
    assert result.naturalisation_repair.status == "applied"
