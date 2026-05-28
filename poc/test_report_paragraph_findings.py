import os
import tempfile

from report.pdf import render_pdf
from report.paragraph_explainer import generate_paragraph_explanations
from report.render import render_markdown
from report.report import DraftReport, Finding, PredictabilitySummary, Tier, report_to_dict


def _sentence(sentence_id, paragraph_id, sentence, risk, label, start, end):
    return {
        "sentence_id": sentence_id,
        "paragraph_id": paragraph_id,
        "sentence": sentence,
        "risk": risk,
        "risk_label": label,
        "start_char": start,
        "end_char": end,
        "top10_ratio": risk,
        "top50_ratio": risk / 2,
        "avg_probability": 0.18,
        "avg_surprisal": 1.2,
        "top_predicted_tokens": [],
        "predictable_token_spans": [],
    }


def test_scan_markdown_and_pdf_group_findings_by_paragraph():
    sentences = [
        _sentence(
            "s001",
            "p001",
            "This paragraph opens with a very general claim.",
            0.7,
            "high",
            0,
            47,
        ),
        _sentence(
            "s002",
            "p001",
            "It also follows a predictable route.",
            0.6,
            "medium",
            48,
            86,
        ),
        _sentence(
            "s003",
            "p002",
            "A second paragraph has no finding.",
            0.1,
            "low",
            88,
            121,
        ),
    ]
    report = DraftReport(
        overall_tier=Tier.HIGH,
        finding_count=2,
        findings_by_tier={
            "critical": [],
            "high": [
                Finding(
                    Tier.HIGH,
                    "predictability",
                    "predictability",
                    "high_predictability",
                    "Sentence scored 70% predictability (common ratio: 66.7%, category: statistical predictability)",
                    "This paragraph opens with a very general claim.",
                    "Use a more specific paragraph route.",
                    sentence_id="s001",
                    finding_id="f001",
                ),
                Finding(
                    Tier.HIGH,
                    "genericity",
                    "predictability",
                    "generic_phrase",
                    "Generic phrase detected: 'important to note'",
                    "It also follows a predictable route.",
                    "Replace generic phrasing with concrete detail.",
                    sentence_id="s002",
                    finding_id="f002",
                ),
            ],
            "medium": [],
            "low": [],
            "clean": [],
        },
        predictability=PredictabilitySummary(0.6, {"high": 2}, sentences, [], []),
        original_text=(
            "This paragraph opens with a very general claim. "
            "It also follows a predictable route.\n\n"
            "A second paragraph has no finding."
        ),
    )

    markdown = render_markdown(report)
    payload = report_to_dict(report)

    assert "### High (1 paragraph, 2 findings)" in markdown
    assert "| # | Src | Sig | Findings | Paragraph | Suggestions |" in markdown
    assert "p001 (s001-s002)" in markdown
    assert payload["scan_intelligence"]["document"]["paragraphs"][0]["text"].startswith(
        "This paragraph opens"
    )
    assert payload["scan_intelligence"]["document"]["paragraphs"][0]["primary_signal"]

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "report.pdf")
        render_pdf(markdown, pdf_path)
        assert os.path.getsize(pdf_path) > 0


def test_paragraph_explainer_generates_once_from_grouped_findings():
    class Gateway:
        model = "planner-test"

        def __init__(self):
            self.calls = 0
            self.prompt = ""

        def chat(self, *args, **kwargs):
            self.calls += 1
            self.prompt = str(args[0])
            assert kwargs["app_label"] == "Dignose"
            return type("Response", (), {
                "content": (
                    '{"paragraphs":[{"paragraph_id":"p001",'
                    '"source_finding_ids":["f001","f002"],'
                    '"reader_summary":"A reader may understand the general topic, but may not see how the teaching reflection connects to the unit example.",'
                    '"main_issue":"Connect the broad learning-design claim to one specific classroom or assessment problem.",'
                    '"why_flagged":["The paragraph moves from a general claim to a unit example without enough linking explanation.","The source references are present, but the student voice needs to explain how they support the point."],'
                    '"recommendation":"Add one concrete teaching or assessment detail already supported by the draft.",'
                    '"rewrite_hint":"For example, add a sentence explaining what learners struggle with before naming the unit.",'
                    '"confidence":"medium"}]}'
                )
            })()

    sentences = [
        _sentence("s001", "p001", "This paragraph opens with a very general claim.", 0.7, "high", 0, 47),
        _sentence("s002", "p001", "It also follows a predictable route.", 0.6, "medium", 48, 86),
    ]
    report = DraftReport(
        overall_tier=Tier.HIGH,
        finding_count=2,
        findings_by_tier={
            "critical": [],
            "high": [
                Finding(Tier.HIGH, "predictability", "predictability", "high_predictability", "Predictable", "Evidence", "Revise.", sentence_id="s001", finding_id="f001"),
                Finding(Tier.HIGH, "genericity", "predictability", "generic_phrase", "Generic", "Evidence", "Revise.", sentence_id="s002", finding_id="f002"),
            ],
            "medium": [],
            "low": [],
            "clean": [],
        },
        predictability=PredictabilitySummary(0.6, {"high": 2}, sentences, [], []),
        original_text="This paragraph opens with a very general claim. It also follows a predictable route.",
    )
    payload = report_to_dict(report)
    gateway = Gateway()

    explanations = generate_paragraph_explanations(payload, gateway=gateway, model="planner-test")

    assert gateway.calls == 1
    assert "Combine repeated sentence findings into one paragraph-level diagnosis" in gateway.prompt
    assert "Do not list every signal mechanically" in gateway.prompt
    assert explanations["schema_version"] == "paragraph_explanations.v2"
    assert explanations["paragraphs"][0]["paragraph_id"] == "p001"
    assert explanations["paragraphs"][0]["reader_summary"].startswith("A reader may understand")
    assert explanations["paragraphs"][0]["main_issue"].startswith("Connect the broad")
    assert explanations["paragraphs"][0]["recommendation"].startswith("Add one concrete")
    assert explanations["paragraphs"][0]["rewrite_hint"].startswith("For example")

    report.paragraph_explanations = explanations
    markdown = render_markdown(report)
    assert "A reader may understand" in markdown
    assert "Main fix: Connect the broad" in markdown
    assert "Add one concrete teaching" in markdown
    assert "For example, add a sentence" in markdown


def test_paragraph_explainer_uses_worker_supplied_llm_settings(monkeypatch):
    from llm import gateway as gateway_module

    captured = {}

    class Gateway:
        model = "planner-test"

        def __init__(self, config):
            captured["api_key"] = config.api_key
            captured["base_url"] = config.base_url

        def chat(self, *_args, **kwargs):
            return type("Response", (), {
                "content": (
                    '{"paragraphs":[{"paragraph_id":"p001",'
                    '"reader_summary":"A reader may need a clearer link between the paragraph claim and the teaching example.",'
                    '"main_issue":"Make the teaching example do more work.",'
                    '"why_flagged":["The paragraph names a concern but does not yet show how it appears in practice."],'
                    '"recommendation":"Rewrite the paragraph with one concrete detail.",'
                    '"rewrite_hint":"Add one sentence that points to the exact classroom issue.",'
                    '"confidence":"medium"}]}'
                )
            })()

    monkeypatch.setattr(gateway_module, "LLMGateway", Gateway)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "wrong-env-key")

    sentences = [
        _sentence("s001", "p001", "This paragraph opens with a very general claim.", 0.7, "high", 0, 47),
    ]
    report = DraftReport(
        overall_tier=Tier.HIGH,
        finding_count=1,
        findings_by_tier={
            "critical": [],
            "high": [
                Finding(Tier.HIGH, "predictability", "predictability", "high_predictability", "Predictable", "Evidence", "Revise.", sentence_id="s001", finding_id="f001"),
            ],
            "medium": [],
            "low": [],
            "clean": [],
        },
        predictability=PredictabilitySummary(0.7, {"high": 1}, sentences, [], []),
        original_text="This paragraph opens with a very general claim.",
    )

    explanations = generate_paragraph_explanations(
        report_to_dict(report),
        api_key="settings-key",
        base_url="https://settings-base.example/v1",
        model="planner-test",
    )

    assert captured == {
        "api_key": "settings-key",
        "base_url": "https://settings-base.example/v1",
    }
    assert explanations["paragraphs"][0]["recommendation"].startswith("Rewrite the paragraph")
