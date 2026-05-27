import json

from rewrite_v6.plan import build_plan
from rewrite_v6.scan import scan_text
from rewrite_v6.write import build_prompt


def _payload(source: str) -> dict:
    paragraph, plan = build_plan(scan_text(source))
    return json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])


def test_v6_writer_prompt_uses_source_units_once_with_reference_contract():
    source = (
        "The intake process is built around a form, a queue, and a review. "
        "Clients submit details, wait for confirmation, and receive a response."
    )

    payload = _payload(source)

    assert "source_units" in payload
    assert "writer_execution_contract" in payload
    assert "coverage_beats_must_all_appear" not in payload
    assert "paragraph_sentence_plan" not in payload
    assert "coverage_loss_contract" not in payload
    assert "construction_recipes" not in payload
    assert "author_route_questions" not in payload
    assert "planner_decision" not in payload
    assert [unit["sentence_id"] for unit in payload["source_units"]] == [
        "p001_s001",
        "p001_s002",
    ]
    assert {row["source_sentence_id"] for row in payload["writer_execution_contract"]["rows"]} == {
        "p001_s001",
        "p001_s002",
    }


def test_v6_writer_prompt_does_not_duplicate_source_text_inside_contract_rows():
    source = "The service uses calls, forms, queues, reviewers, messages, dashboards, and follow-up checks."

    payload = _payload(source)
    contract_text = json.dumps(payload["writer_execution_contract"], ensure_ascii=False)

    assert source in json.dumps(payload["source_units"], ensure_ascii=False)
    assert source not in contract_text
    assert "calls" in contract_text.casefold()
    assert "forms" in contract_text.casefold()
