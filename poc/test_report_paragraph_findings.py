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

        def chat(self, *_args, **kwargs):
            self.calls += 1
            assert kwargs["app_label"] == "Dignose"
            return type("Response", (), {
                "content": (
                    '{"paragraphs":[{"paragraph_id":"p001",'
                    '"source_finding_ids":["f001","f002"],'
                    '"summary":"This paragraph is broad and reads more like a general explanation than a specific teaching reflection.",'
                    '"why_flagged":["Several findings point to the same paragraph.","The fix should happen at paragraph level."],'
                    '"recommendation":"Add one concrete teaching or assessment detail already supported by the draft.",'
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
    assert explanations["schema_version"] == "paragraph_explanations.v1"
    assert explanations["paragraphs"][0]["paragraph_id"] == "p001"
    assert explanations["paragraphs"][0]["recommendation"].startswith("Add one concrete")

    report.paragraph_explanations = explanations
    markdown = render_markdown(report)
    assert "This paragraph is broad" in markdown
    assert "Add one concrete teaching" in markdown
