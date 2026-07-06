from __future__ import annotations

import json
from pathlib import Path

from poc.rewrite_v6 import pipeline as v6_pipeline
from poc.rewrite_v6.llm_config import provider_from_env
from poc.rewrite_v6.pipeline import _writer_provider, run_v6_rewrite, run_v6_rewrite_all
from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.planner_llm import run_planner_llm
from poc.rewrite_v6.quality_repair import QualityRepairOperation, apply_quality_repair_operations, run_quality_repair_once
from poc.rewrite_v6 import production as v6_production
from poc.rewrite_v6.scan import findings_for_paragraph, scan_text
from poc.rewrite_v6.write import Variant, build_prompt, choose_variant
from poc.report.render_rewrite import render_rewrite_report


class FakeResponse:
    content = '{"variants":[]}'
    raw_content = content


class FakeClient:
    calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse()


class BadJsonResponse:
    content = '{"variants":[{"id":"v1","text":"broken"}'
    raw_content = content


class BadJsonClient:
    def __init__(self):
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return BadJsonResponse()


class StaticJsonResponse:
    def __init__(self, content: str):
        self.content = content
        self.raw_content = content


class StaticJsonClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return StaticJsonResponse(self.content)


def sample_text() -> str:
    return (
        "The intake process is built around a form, a queue, and a review. "
        "Clients submit details, wait for confirmation, and receive a response."
    )


def test_v6_named_reference_extraction_is_not_domain_keyword_filtered():
    scan = scan_text(
        "Students in Group Alpha used Review Method (2024).\n\n"
        "This result shows an important concern because the process should improve."
    )
    _paragraph, plan = build_plan(scan)
    assert "Group Alpha" in plan.author_proxy_context["named_references"]
    assert "Review Method" in plan.author_proxy_context["named_references"]
    assert "Students" not in plan.author_proxy_context["named_references"]


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def test_v6_selection_rejects_malformed_not_only_concern():
    source = (
        "This is a serious concern because the modern world does not only reward people who remember facts. "
        "It rewards people who can analyse, adapt, communicate, and create."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": (
                    "The modern world rewards people who remember facts, not only a serious concern. "
                    "People who analyse and adapt also matter."
                ),
            }
        ]
    })
    selector_payload = json.dumps({"selected_id": "v1", "rationale": "reviewable author bridge"})
    result = run_v6_rewrite(
        source,
        writer_client=StaticJsonClient(writer_payload),
        selector_client=StaticJsonClient(selector_payload),
    )
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_selection_rejects_not_only_without_also_side():
    source = (
        "The process does not only focus on what clients know, but also on how clients decide. "
        "Reviewers guide clients through the check."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": (
                    "The process does not only focus on what clients know. "
                    "Reviewers guide clients through the check."
                ),
            }
        ]
    })
    selector_payload = json.dumps({"selected_id": "v1", "rationale": "reviewable author bridge"})
    result = run_v6_rewrite(
        source,
        writer_client=StaticJsonClient(writer_payload),
        selector_client=StaticJsonClient(selector_payload),
    )
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_selection_rejects_unresolved_comma_list_contract():
    source = "The process rewards people who analyse, adapt, communicate, and create."
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": "The process rewards people who analyse, adapt, communicate, and create.",
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_rewrite_preserves_source_when_writer_returns_no_candidate():
    client = FakeClient()
    result = run_v6_rewrite(sample_text(), writer_client=client)
    assert result.selected is not None
    assert result.selected.source == "source_preserved"
    assert "form" in result.selected.text
    assert "form" in result.rewritten_text
    # The writer layer now retries empty/invalid responses (variant re-request loop),
    # so a no-candidate outcome takes more than one call.
    assert client.calls >= 1


def test_v6_bad_writer_json_preserves_source_instead_of_compiling():
    client = BadJsonClient()
    source = "This result shows an important concern because the process should improve."
    result = run_v6_rewrite(source, writer_client=client)
    assert result.selected is not None
    assert result.selected.source == "source_preserved"
    assert result.rewritten_text == source
    # Bad JSON is retried before falling back to the preserved source.
    assert client.calls >= 1


def test_v6_parse_variants_preserves_string_author_review_rows():
    from poc.rewrite_v6.write import parse_variants

    variants = parse_variants({
        "variants": [
            {
                "id": "v1",
                "text": "The route needs a bridge.",
                "author_proxy_provenance": ["bridge inferred from nearby context"],
                "author_review_items": ["confirm this bridge"],
                "coverage_map": [{"coverage_beat_id": "b1", "sentence": "The route needs a bridge."}],
            }
        ]
    })
    assert variants[0].author_proxy_provenance
    assert variants[0].author_proxy_provenance[0]["generated_text"] == "bridge inferred from nearby context"
    assert variants[0].author_review_items
    assert variants[0].author_review_items[0]["provenance"] == "needs_author_confirmation"


def test_v6_parse_variants_drops_duplicate_texts():
    from poc.rewrite_v6.write import parse_variants

    variants = parse_variants({
        "variants": [
            {"id": "v1", "mode": "coverage_beat_generation", "text": "The route changes."},
            {"id": "v2", "mode": "golden_question_generation", "text": "The route changes."},
            {"id": "v3", "mode": "context_anchor_generation", "text": "The route changes differently."},
        ]
    })
    assert [variant.id for variant in variants] == ["v1", "v3"]
    assert variants[0].mode == "coverage_beat_generation"


def test_v6_files_stay_within_project_line_cap_and_do_not_import_v5():
    # Project hard rule: a single file must not exceed 1500 lines. The original
    # 1000-line aspiration was outgrown by pipeline.py/plan.py/direct_rewrite.py.
    root = Path(__file__).resolve().parent / "rewrite_v6"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 1500, path
        assert "rewrite_v5" not in text, path


def test_v6_json_result_is_serializable():
    result = run_v6_rewrite("A process uses a step, a check, and a result.", writer_client=FakeClient())
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["variants"] == []
    assert payload["source_preserved"]["source"] == "source_preserved"


def test_v6_document_rewrite_reports_paragraph_progress():
    events = []

    run_v6_rewrite_all(
        "A process uses a form, a queue, and a review.",
        writer_client=FakeClient(),
        progress_callback=lambda percent, message: events.append((percent, message)),
    )

    assert any(percent > 62 for percent, _message in events)
    assert any("Planning V6 paragraph" in message for _percent, message in events)
    assert any("Writing V6 paragraph" in message for _percent, message in events)


def test_v6_document_rewrite_stops_before_llm_when_runtime_budget_is_tight():
    writer = FakeClient()
    events = []

    result = run_v6_rewrite_all(
        "A process uses a form, a queue, and a review.",
        writer_client=writer,
        progress_callback=lambda percent, message: events.append((percent, message)),
        runtime_budget_seconds=1,
        min_llm_request_seconds=30,
    )

    assert writer.calls == 0
    assert result.passes == []
    assert result.rewritten_text == "A process uses a form, a queue, and a review."
    assert any("runtime budget reached" in message for _percent, message in events)


def test_v6_quality_repair_applies_minimal_exact_operations():
    text = "Approach increases student motivation. I am an educator who need to listen."
    repaired, applied, skipped = apply_quality_repair_operations(text, [
        QualityRepairOperation("Approach increases", "This approach increases", "fragment"),
        QualityRepairOperation("who need", "who needs", "agreement"),
    ])

    assert repaired == "This approach increases student motivation. I am an educator who needs to listen."
    assert [operation.reason for operation in applied] == ["fragment", "agreement"]
    assert skipped == []


def test_v6_quality_repair_skips_expansive_or_protected_changes():
    text = "The unit SHBHCUT006 requires six assessments."
    repaired, applied, skipped = apply_quality_repair_operations(text, [
        QualityRepairOperation("SHBHCUT006", "SHBHCUT007", "typo"),
        QualityRepairOperation("The unit", "The practical unit that I use with learners in the classroom", "style"),
    ])

    assert repaired == text
    assert applied == []
    assert {row["skip_reason"] for row in skipped} == {"protected_token_changed", "replacement_too_expansive"}


def test_v6_quality_repair_runs_once_after_selected_rewrite(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_GRAMMER_REPAIR_ENABLED", "1")
    monkeypatch.setenv("DRAFTPROOF_V6_NATURALISATION_ENABLED", "0")
    source = "This process uses a form, a queue, and a review."
    rewritten = "I am an educator who need to listen."
    seen = {}

    class QualityClient:
        model = "grammer-test"

        def __init__(self):
            self.calls = 0

        def chat(self, _prompt, **kwargs):
            self.calls += 1
            seen["app_label"] = kwargs.get("app_label")
            return StaticJsonResponse(json.dumps({
                    "operations": [{
                    "find": "who need",
                    "replace": "who needs",
                    "reason": "agreement",
                }]
            }))

    def fake_run(current, **_kwargs):
        scan = scan_text(current)
        paragraph, plan = build_plan(scan)
        # A pass is only accepted when a selected (llm) variant carries the change.
        selected = Variant(id="v1", text=rewritten, source="llm")
        return v6_pipeline.Result(scan=scan, plan=plan, variants=[selected], selected=selected, rewritten_text=rewritten)

    quality = QualityClient()
    monkeypatch.setattr(v6_pipeline, "run_v6_rewrite", fake_run)
    monkeypatch.setattr(v6_pipeline, "_acceptable_progress", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(v6_pipeline, "_cross_paragraph_regression", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(v6_pipeline, "_target_paragraph_acceptance_ok", lambda *_args, **_kwargs: True)

    result = run_v6_rewrite_all(source, quality_client=quality, max_passes=1, residual_followup_passes=0)

    assert quality.calls == 1
    assert seen["app_label"] == "Grammer"
    assert result.final_text_before_quality_repair == rewritten
    assert result.rewritten_text == "I am an educator who needs to listen."
    assert result.quality_repair is not None
    assert result.quality_repair.status == "applied"


def test_v6_quality_repair_reverts_scan_regression(monkeypatch):
    # The revert gate is exercised directly with a controlled scan (the re-tuned scanner
    # no longer regresses on any fixed real-text repair, so the old end-to-end fixture
    # cannot trigger this path deterministically).
    from poc.rewrite_v6.quality_repair import QualityRepairResult

    original = "Original text."
    repaired = "Repaired text."

    def fake_scan(text):
        scan = scan_text("The process uses a form.")
        scores = dict(scan.scores)
        scores["finding_count"] = 3.0
        scores["mean_sentence_shape_risk"] = 5.0 if text == original else 6.0
        return scan.__class__(
            source_text=scan.source_text,
            paragraphs=scan.paragraphs,
            findings=scan.findings,
            scores=scores,
        )

    monkeypatch.setattr(v6_pipeline, "scan_text", fake_scan)
    monkeypatch.setenv("DRAFTPROOF_V6_GRAMMER_RISK_TOLERANCE", "0")

    result = v6_pipeline._risk_safe_quality_repair(
        original,
        QualityRepairResult(original_text=original, repaired_text=repaired, status="applied"),
    )

    assert result is not None
    assert result.repaired_text == original
    assert result.status == "reverted_scan_regression"
    assert result.skipped_operations[-1]["skip_reason"] == "scan_regression"


def test_v6_quality_repair_layer_defaults_off(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_GRAMMER_REPAIR_ENABLED", raising=False)
    client = FakeClient()

    result = run_quality_repair_once(
        "I am an educator who need to listen.",
        original_text="I am an educator who needed to listen.",
        quality_client=client,
        api_key=None,
        base_url=None,
        cancellation_check=None,
        progress_callback=None,
    )

    assert result is None
    assert client.calls == 0


def test_v6_quality_repair_allows_minor_shape_risk_increase(monkeypatch):
    from poc.rewrite_v6.quality_repair import QualityRepairResult

    original = "Original text."
    repaired = "Repaired text."

    def fake_scan(text):
        scan = scan_text("The process uses a form.")
        scores = dict(scan.scores)
        scores["finding_count"] = 3.0
        scores["mean_sentence_shape_risk"] = 5.0 if text == original else 6.0
        return scan.__class__(
            source_text=scan.source_text,
            paragraphs=scan.paragraphs,
            findings=scan.findings,
            scores=scores,
        )

    monkeypatch.setattr(v6_pipeline, "scan_text", fake_scan)
    monkeypatch.setenv("DRAFTPROOF_V6_GRAMMER_RISK_TOLERANCE", "1")

    result = v6_pipeline._risk_safe_quality_repair(
        original,
        QualityRepairResult(original_text=original, repaired_text=repaired, status="applied"),
    )

    assert result is not None
    assert result.repaired_text == repaired
    assert result.status == "applied"


def test_v6_document_rewrite_honors_cancellation_before_llm():
    class Canceled(BaseException):
        pass

    writer = FakeClient()

    try:
        run_v6_rewrite_all(
            "A process uses a form, a queue, and a review.",
            writer_client=writer,
            cancellation_check=lambda: (_ for _ in ()).throw(Canceled()),
        )
    except Canceled:
        pass
    else:
        raise AssertionError("Expected cancellation to stop V6 before LLM calls")

    assert writer.calls == 0


def test_v6_production_adapter_returns_worker_contract(tmp_path, monkeypatch):
    # Pin the legacy planner/selector path: the default direct-rewrite path builds a real
    # LLM client (requires an API key), which this offline contract test cannot provide.
    monkeypatch.setenv("DRAFTPROOF_V6_DIRECT_REWRITE", "0")
    monkeypatch.setattr(v6_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))
    monkeypatch.setattr(v6_production, "_scan_report", fake_full_scan_report)
    events = []
    result = v6_production.run_rewrite_pipeline_v6(
        detect_json={"input_text": "The review moved from intake to approval."},
        output_dir=str(tmp_path),
        model="qwen/qwen3-30b-a3b-instruct-2507",
        progress_callback=lambda percent, message: events.append((percent, message)),
    )
    summary = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert result["status"] in {"ai_mitigated", "partial_candidate_not_strict_safe", "original_preserved"}
    assert Path(result["md_path"]).exists()
    assert Path(result["pdf_path"]).exists()
    assert summary["rewrite_pipeline_version"] == "rewrite_v6_scanner_planner_writer"
    assert summary["rewrite_effective_config"]["model"] == "qwen/qwen3-30b-a3b-instruct-2507"
    assert summary["final_text"]
    assert summary["detect_scores"]["original_ai_authorship"] == 2
    assert summary["detect_scores"]["rewritten_ai_authorship"] == 2
    assert summary["detect_scores"]["original_human_contribution"] == 96
    assert summary["detect_scores"]["rewritten_human_contribution"] == 96
    assert summary["detect_scores"]["original_ai_transformation"] == 4
    assert summary["detect_scores"]["rewritten_ai_transformation"] == 4
    assert summary["detect_scores"]["human_shift_score"] == 0
    assert summary["original_risk"] == 2
    assert summary["final_risk"] == 2
    assert any(percent > 62 for percent, _message in events)
    rewritten_scan = summary["detect_scan_rewritten"]
    rewritten_badge = rewritten_scan["ai_risk_badge"]
    rewritten_intelligence = rewritten_scan["scan_intelligence"]["transformation"]
    assert rewritten_badge["transformation_classification"]["code"] == "low_ai_signal"
    assert rewritten_badge["ai_components"]["topk_calibrated_risk"] == 4
    assert rewritten_intelligence["contribution"]["human_contribution_ratio"] == 96
    assert rewritten_intelligence["core_signals"][0]["key"] == "topk_calibrated_risk"
    assert rewritten_scan["integrity_layers"]["layers"]["human_contribution_signal"]["score"] == 96


def test_v6_rewrite_pdf_metrics_use_detect_scores_when_scan_badges_are_compact():
    summary = {
        "outcome": "ai_mitigated",
        "status": "ai_mitigated",
        "detect_scores": {
            "original_ai_authorship": 70,
            "rewritten_ai_authorship": 30,
            "original_human_contribution": 56,
            "rewritten_human_contribution": 83,
            "original_ai_transformation": 44,
            "rewritten_ai_transformation": 17,
            "original_grounding_quality_risk": 90,
            "rewritten_grounding_quality_risk": 25,
        },
        "detect_scan_original": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 0, "writing_quality_score": 0},
        },
        "detect_scan_rewritten": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 0, "writing_quality_score": 0},
        },
    }
    report = render_rewrite_report(summary, [], [], original_text="Original.", final_text="Rewritten.")
    # AI Likelihood shows the canonical (un-halved, 00352fec) detect_scores values.
    assert "| **AI Likelihood** | `70%` | `30%` | `-40%` |" in report
    assert "| **Human Contribution** | `56%` | `83%` | `+27%` |" in report
    assert "| **AI Transformation** | `44%` | `17%` | `-27%` |" in report
    assert "| **Grounding Quality Risk** | `90.00%` | `25.00%` | `-65.00%` |" in report


def test_v6_rewrite_pdf_stamp_uses_calibrated_authorship_band():
    summary = {
        "outcome": "partial_candidate_not_strict_safe",
        "status": "partial_candidate_not_strict_safe",
        "detect_scores": {
            "original_ai_authorship": 70,
            "rewritten_ai_authorship": 43,
            # The hero panel formats before→after pairs, so both sides must be present.
            "original_human_contribution": 55,
            "rewritten_human_contribution": 72,
            "original_ai_transformation": 45,
            "rewritten_ai_transformation": 28,
        },
        "detect_scan_original": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 70, "writing_quality_score": 60},
        },
        "detect_scan_rewritten": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {
                "ai_likelihood_score": 43,
                "writing_quality_score": 72,
                "authorship_rating": {"label": "AI-Generated / AI-Paraphrased Signals", "short_label": "AI Signals"},
            },
            "scan_intelligence": {
                "transformation": {
                    "contribution": {
                        "human_contribution_ratio": 72,
                        "ai_transformation_ratio": 28,
                    }
                }
            },
        },
    }

    report = render_rewrite_report(summary, [], [], original_text="Original.", final_text="Rewritten.")

    # 43% is banded on the canonical (un-halved) scale, so the stamp reads Possible,
    # not the old halved-display Unlikely band.
    assert "<h3>Possible AI-Assisted</h3>" in report


def fake_full_scan_report(text: str) -> dict:
    return {
        "input_text": text,
        "findings": {"critical": [], "high": [], "medium": [], "low": []},
        "ai_score": 2,
        "writing_score": 3,
        "ai_risk_badge": {
            "ai_likelihood_score": 2,
            "writing_quality_score": 3,
            "tier": "green",
            "authorship_rating": {"label": "Good", "short_label": "Good", "code": "low_ai_signal"},
            "authorship_rating_label": "Good",
            "ai_components": {
                "topk_pattern_raw": 7,
                "topk_calibrated_risk": 4,
                "topk_calibration_eligible": True,
            },
            "transformation_classification": {
                "code": "low_ai_signal",
                "label": "Human contribution pattern",
                "confidence": "high",
                "features": {
                    "ai_likelihood": 2,
                    "human_anchor_score": 96,
                    "calibrated_ai_risk": 2,
                    "topk_calibrated_risk": 4,
                },
            },
        },
        "scan_intelligence": {
            "transformation": {
                "contribution": {
                    "human_contribution_ratio": 96,
                    "ai_transformation_ratio": 4,
                    "calibrated_ai_risk": 2,
                    "human_anchor_discount": 70,
                    "calibration_confidence": 92,
                    "reporting_suppression": 15,
                    "summary": "Human anchoring dominates the rewritten scan.",
                },
                "core_signals": [
                    {
                        "key": "topk_calibrated_risk",
                        "label": "Calibrated Top-k Risk",
                        "score": 4,
                    }
                ],
            }
        },
        "integrity_layers": {
            "layers": {
                "human_contribution_signal": {"score": 96},
                "ai_transformation_risk": {"score": 4},
            }
        },
    }


def test_v6_scanner_does_not_count_parenthetical_citation_commas_as_lists():
    scan = scan_text("Participants reviewed the service before choosing a provider (Riley et al., 2024).")
    assert not any("packed_list" in finding.tags for finding in scan.findings)


def test_v6_scanner_treats_parenthetical_citation_as_smoothing_support():
    scan = scan_text("The monthly review programme can evaluate staff attentiveness during the service experience (Riley, 2024).")
    assert not any("paraphrase_smoothing" in finding.tags for finding in scan.findings)


def test_v6_scanner_does_not_treat_local_must_as_unsupported_gap():
    scan = scan_text("The method gives participants a concrete way to grasp the pattern before they must name it.")
    assert not any("unsupported_claim_gap" in finding.tags for finding in scan.findings)


def test_v6_scanner_does_not_count_single_context_comma_as_list_density():
    scan = scan_text("During the first stage, the process moved from intake to review.")
    assert not any("packed_list" in finding.tags for finding in scan.findings)


def test_v6_scanner_does_not_count_single_contrast_comma_as_list_density():
    scan = scan_text("The result changed the process, not the whole service.")
    assert not any("packed_list" in finding.tags for finding in scan.findings)


def test_v6_scanner_does_not_overflag_short_result_sentence():
    scan = scan_text("They may experience frustration.")
    assert not scan.findings


def test_v6_scanner_does_not_mark_named_method_as_smooth_paraphrase():
    scan = scan_text("The Alpha Method gives participants a concrete way to grasp projection angles before naming them.")
    assert not any("paraphrase_smoothing" in finding.tags for finding in scan.findings)


def test_v6_scanner_flags_repeated_sentence_frames():
    scan = scan_text(
        "The review process changed after intake. "
        "Clients also use the portal. "
        "Clients also use phone support. "
        "Clients also use the front desk. "
        "The final step checks the result."
    )
    repeated = [finding for finding in scan.findings if "repeated_sentence_frame" in finding.tags]
    assert len(repeated) == 3
    assert {finding.evidence["repeated_frame"] for finding in repeated} == {"clients also use"}


def test_v6_selection_accepts_finding_drop_when_risk_not_worse():
    source = (
        "Inclusive learning design must move beyond passive accommodation and instead focus on embedding metacognitive strategies. "
        "Complex spatial tasks such as forms, checks, queues, workflows, reviewers, and decisions strain working memory. "
        "The process demonstrates the difference between giving a person an answer and teaching them how to work."
    )
    candidate = (
        "Inclusive learning design moves beyond passive accommodation by embedding metacognitive strategies. "
        "Complex spatial tasks can strain working memory. "
        "Forms and checks carry one part of the process. "
        "Queues and reviewer decisions carry the next part. "
        "The process compares giving a person an answer with teaching them how to work."
    )
    paragraph = scan_text(source).paragraphs[0]
    selected = choose_variant([
        Variant(id="source_preserved", text=source, source="source_preserved"),
        Variant(id="v1", text=candidate, source="llm"),
    ], paragraph)

    assert selected is not None
    assert selected.id == "v1"


def test_v6_document_rewrite_preserves_source_when_writer_has_no_candidates():
    result = run_v6_rewrite_all(sample_text() + "\n\n" + sample_text(), writer_client=FakeClient(), max_passes=2)
    assert result.passes == []
    assert "form" in result.rewritten_text
    json.dumps(result.to_dict())


# Model-specific provider defaults (glm-4.7 -> Cerebras) were removed in the rewrite
# de-hardcode work: provider routing is now purely env-driven via provider_from_env.


def test_v6_planner_provider_has_no_model_hardcoded_default(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_CEREBRAS_DIRECT", "0")
    for name in ("ROUTING_JSON", "ORDER", "ONLY", "IGNORE", "SORT"):
        monkeypatch.delenv(f"DRAFTPROOF_V6_PLANNER_PROVIDER_{name}", raising=False)
    monkeypatch.delenv("DRAFTPROOF_V6_PLANNER_ALLOW_FALLBACKS", raising=False)
    monkeypatch.delenv("DRAFTPROOF_V6_PROVIDER_DEFAULT_ORDER", raising=False)

    assert provider_from_env("PLANNER", "z-ai/glm-4.7") is None


def test_v6_writer_provider_has_no_model_hardcoded_default(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_CEREBRAS_DIRECT", "0")
    for name in ("PROVIDER_ROUTING_JSON", "PROVIDER_ORDER", "PROVIDER_ONLY", "PROVIDER_IGNORE", "PROVIDER_SORT", "PROVIDER_ALLOW_FALLBACKS"):
        monkeypatch.delenv(f"DRAFTPROOF_V6_WRITER_{name}", raising=False)
    monkeypatch.delenv("DRAFTPROOF_V6_WRITER_ALLOW_FALLBACKS", raising=False)
    monkeypatch.delenv("DRAFTPROOF_V6_PROVIDER_DEFAULT_ORDER", raising=False)

    assert _writer_provider("z-ai/glm-4.7") is None


def test_v6_planner_provider_env_overrides_default(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_CEREBRAS_DIRECT", "0")
    monkeypatch.delenv("DRAFTPROOF_V6_PLANNER_PROVIDER_ROUTING_JSON", raising=False)
    monkeypatch.setenv("DRAFTPROOF_V6_PLANNER_PROVIDER_ORDER", "Cerebras,Fireworks")
    monkeypatch.setenv("DRAFTPROOF_V6_PLANNER_PROVIDER_SORT", "throughput")
    monkeypatch.setenv("DRAFTPROOF_V6_PLANNER_ALLOW_FALLBACKS", "0")

    assert provider_from_env("PLANNER", "z-ai/glm-4.7") == {
        "order": ["Cerebras", "Fireworks"],
        "allow_fallbacks": False,
        "sort": "throughput",
    }
