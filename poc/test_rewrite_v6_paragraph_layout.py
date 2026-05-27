from __future__ import annotations

from types import SimpleNamespace

from poc.rewrite_v6 import pipeline as v6_pipeline
from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.paragraph_layout import restore_original_paragraph_layout
from poc.rewrite_v6.quality_repair import QualityRepairOperation, QualityRepairResult
from poc.rewrite_v6.scan import scan_text
from poc.rewrite_v6.write import Variant


def test_v6_layout_restores_split_rewrite_to_original_paragraph_slot():
    original = "Opening paragraph stays here.\n\nTarget paragraph needs repair.\n\nClosing paragraph stays here."
    split_rewrite = (
        "Opening paragraph stays here.\n\n"
        "Target paragraph first repaired block.\n\n"
        "Target paragraph second repaired block.\n\n"
        "Closing paragraph stays here."
    )
    after_grammer = split_rewrite.replace("second repaired block", "second repaired block with grammar fixed")
    result = SimpleNamespace(
        scan=scan_text(original),
        plan=SimpleNamespace(paragraph_id="p002"),
        selected=Variant(
            id="v1",
            text="Target paragraph first repaired block.\n\nTarget paragraph second repaired block.",
            source="llm",
        ),
        rewritten_text=split_rewrite,
    )

    restored = restore_original_paragraph_layout(original, after_grammer, [result])

    assert restored == (
        "Opening paragraph stays here.\n\n"
        "Target paragraph first repaired block. Target paragraph second repaired block with grammar fixed.\n\n"
        "Closing paragraph stays here."
    )
    assert scan_text(restored).scores["paragraph_count"] == scan_text(original).scores["paragraph_count"]


def test_v6_layout_keeps_split_child_residuals_inside_source_paragraph():
    original = "First target paragraph needs repair.\n\nSecond paragraph stays stable."
    first_pass_text = (
        "First target opening repair.\n\n"
        "First target follow-up repair.\n\n"
        "Second paragraph stays stable."
    )
    second_pass_text = (
        "First target opening repair.\n\n"
        "First target follow-up repair after residual pass.\n\n"
        "Second paragraph stays stable."
    )
    first_result = SimpleNamespace(
        scan=scan_text(original),
        plan=SimpleNamespace(paragraph_id="p001"),
        selected=Variant(
            id="v1",
            text="First target opening repair.\n\nFirst target follow-up repair.",
            source="llm",
        ),
        rewritten_text=first_pass_text,
    )
    second_result = SimpleNamespace(
        scan=scan_text(first_pass_text),
        plan=SimpleNamespace(paragraph_id="p002"),
        selected=Variant(
            id="v2",
            text="First target follow-up repair after residual pass.",
            source="llm",
        ),
        rewritten_text=second_pass_text,
    )

    restored = restore_original_paragraph_layout(original, second_pass_text, [first_result, second_result])

    assert restored == (
        "First target opening repair. First target follow-up repair after residual pass.\n\n"
        "Second paragraph stays stable."
    )
    assert scan_text(restored).scores["paragraph_count"] == scan_text(original).scores["paragraph_count"]


def test_v6_document_result_restores_layout_after_grammer_layer(monkeypatch):
    source = (
        "This method uses forms, queues, labels, reviews, approvals, and checks because students should improve.\n\n"
        "Stable paragraph stays here."
    )
    split_rewrite = (
        "This method uses forms, queues, and labels.\n\n"
        "The follow-up review uses approvals and checks because students should improve.\n\n"
        "Stable paragraph stays here."
    )

    def fake_rewrite(current: str, **_: object) -> v6_pipeline.Result:
        scan = scan_text(current)
        paragraph, plan = build_plan(scan)
        return v6_pipeline.Result(
            scan=scan,
            plan=plan,
            variants=[],
            selected=Variant(
                id="v1",
                text=(
                    "This method uses forms, queues, and labels.\n\n"
                    "The follow-up review uses approvals and checks because students should improve."
                ),
                source="llm",
            ),
            rewritten_text=split_rewrite,
        )

    def fake_grammer(current: str, **_: object) -> QualityRepairResult:
        repaired = current.replace("follow-up", "follow up")
        return QualityRepairResult(
            original_text=current,
            repaired_text=repaired,
            operations=[QualityRepairOperation(find="follow-up", replace="follow up", reason="grammar")],
            status="applied",
        )

    monkeypatch.setattr(v6_pipeline, "run_v6_rewrite", fake_rewrite)
    monkeypatch.setattr(v6_pipeline, "run_quality_repair_once", fake_grammer)
    monkeypatch.setattr(v6_pipeline, "_risk_safe_quality_repair", lambda _current, repair: repair)
    monkeypatch.setattr(v6_pipeline, "_acceptable_progress", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(v6_pipeline, "_cross_paragraph_regression", lambda *_args, **_kwargs: False)

    result = v6_pipeline.run_v6_rewrite_all(source, max_passes=1, residual_followup_passes=0)

    assert result.rewritten_text == (
        "This method uses forms, queues, and labels. "
        "The follow up review uses approvals and checks because students should improve.\n\n"
        "Stable paragraph stays here."
    )
    assert result.final_scan.scores["paragraph_count"] == result.initial_scan.scores["paragraph_count"]
