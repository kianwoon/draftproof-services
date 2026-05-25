from __future__ import annotations

import json

from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.pipeline import run_v6_rewrite
from poc.rewrite_v6.planner_llm import build_planner_prompt
from poc.rewrite_v6.scan import scan_text
from poc.rewrite_v6.write import build_prompt


class StaticJsonResponse:
    def __init__(self, content: str):
        self.content = content
        self.raw_content = content


class StaticJsonClient:
    def __init__(self, content: str):
        self.content = content

    def chat(self, *args, **kwargs):
        return StaticJsonResponse(self.content)


class CaptureClient:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[tuple[tuple, dict]] = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return StaticJsonResponse(self.content)


class SequenceClient:
    def __init__(self, contents: list[str]):
        self.contents = contents
        self.calls: list[tuple[tuple, dict]] = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return StaticJsonResponse(self.contents[index])


def _payload(text: str) -> dict:
    paragraph, plan = build_plan(scan_text(text))
    return json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])


def test_v6_planner_emits_positive_construction_recipes():
    payload = _payload(
        "The process uses forms, queues, reviewers, feedback, decisions, and follow-up checks."
    )
    recipes = payload["construction_recipes"]
    assert recipes
    assert any(recipe["positive_pattern"] for recipe in recipes)
    assert any("separate source-relation beats" in recipe["build_route"] for recipe in recipes)
    assert "Use construction_recipes as the positive build plan" in json.dumps(payload)


def test_v6_coverage_beats_link_to_construction_recipes():
    payload = _payload(
        "This is an important concern because the process should improve across teams."
    )
    beats = payload["coverage_beats_must_all_appear"]
    assert beats
    assert all(beat.get("construction_recipe_id") for beat in beats)
    assert all(beat.get("construction_recipe") for beat in beats)
    assert any(
        "observable source relation" in " ".join(beat["construction_recipe"]["build_steps"])
        for beat in beats
    )


def test_v6_writer_schema_requires_recipe_id_in_coverage_map():
    payload = _payload("This process shows a concern because support should improve.")
    schema_text = json.dumps(payload["output_schema"])
    assert "construction_recipe_id" in schema_text


def test_v6_llm_planner_prompt_receives_scanner_findings():
    scan = scan_text("This result shows a concern because the process should improve.")
    paragraph, plan = build_plan(scan)
    prompt = build_planner_prompt(paragraph, plan, scan.findings)
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["scanner_findings"]
    assert payload["deterministic_route_skeleton"]["construction_recipes"]
    assert payload["required_decision"]["finding_contracts"]
    assert payload["required_decision"]["finding_recipe_overrides"]
    assert payload["required_decision"]["paragraph_blueprint"]
    assert "Return one finding_contract for every scanner_findings row" in prompt
    assert "Do not use placeholder-only safe shapes" in prompt
    assert "actual submitted source terms" in prompt
    assert "planning labels" in prompt
    assert "scoped partial-relation shape" in prompt
    assert "concrete enough that a writer can follow it without guessing" in prompt
    assert "Do not write replacement paragraph prose" in prompt


def test_v6_pipeline_calls_planner_before_writer_when_supplied():
    planner_payload = json.dumps({
        "planner_decision": {
            "paragraph_route": "Use source relation before claim.",
            "finding_contracts": [
                {
                    "finding_id": "p001_s001:predictable_start",
                    "source_sentence_id": "p001_s001",
                    "finding_tags": ["predictable_start"],
                    "unsafe_original_shape": "This result shows",
                    "safe_rebuild_shape": "The process shows a narrow concern.",
                    "writer_must_do": ["start from source object"],
                    "writer_must_not_do": ["reuse This result shows"],
                    "coverage_terms": ["process"],
                }
            ],
            "paragraph_blueprint": [
                {
                    "step_id": "b001",
                    "function": "start from the source object",
                    "source_basis": ["p001_s001"],
                    "must_include": ["process"],
                    "must_avoid_shape": ["This result shows"],
                    "safe_sentence_shape": "<source object> shows <narrow relation>",
                }
            ],
            "finding_recipe_overrides": [
                {
                    "source_sentence_id": "p001_s001",
                    "safe_route": "anchor first",
                    "build_steps": ["name source object first"],
                    "positive_pattern": "<object> shows <relation>",
                }
            ],
            "author_proxy_plan": "mark inferred bridges",
            "do_not_copy_route": ["same opener"],
        }
    })
    writer_payload = json.dumps({"variants": []})
    planner = CaptureClient(planner_payload)
    writer = CaptureClient(writer_payload)
    result = run_v6_rewrite(
        "This result shows a concern because the process should improve.",
        planner_client=planner,
        writer_client=writer,
    )
    assert len(planner.calls) == 1
    assert len(writer.calls) == 1
    writer_prompt = writer.calls[0][0][0]
    assert "Use source relation before claim" in writer_prompt
    assert "p001_s001:predictable_start" in writer_prompt
    assert "Treat planner_decision.finding_contracts as the primary build contract" in writer_prompt
    assert "If safe_rebuild_shape contains placeholder brackets" in writer_prompt
    assert "Do not copy planning labels" in writer_prompt
    assert "scoped partial-relation sentence" in writer_prompt
    assert "safe_sentence_shape" in writer_prompt
    assert result.plan.ai_safe_route["llm_planner_decision"]["status"] == "ok"


def test_v6_planner_retries_when_contract_copies_risky_source_route():
    bad = json.dumps({
        "planner_decision": {
            "paragraph_route": "bad",
            "finding_contracts": [
                {
                    "finding_id": "p001_s001:broad_claim",
                    "source_sentence_id": "p001_s001",
                    "finding_tags": ["broad_claim"],
                    "unsafe_original_shape": "This model no longer fully reflects how people learn.",
                    "safe_rebuild_shape": "This model no longer fully reflects how people learn.",
                    "writer_must_do": ["copy"],
                    "writer_must_not_do": [],
                    "coverage_terms": ["model"],
                }
            ],
            "paragraph_blueprint": [],
        }
    })
    good = json.dumps({
        "planner_decision": {
            "paragraph_route": "good",
            "finding_contracts": [
                {
                    "finding_id": "p001_s001:broad_claim",
                    "source_sentence_id": "p001_s001",
                    "finding_tags": ["broad_claim"],
                    "unsafe_original_shape": "This model no longer fully reflects how people learn.",
                    "safe_rebuild_shape": "The model explains only part of how people learn.",
                    "writer_must_do": ["scope relation"],
                    "writer_must_not_do": ["copy broad predicate"],
                    "coverage_terms": ["model", "people learn"],
                }
            ],
            "paragraph_blueprint": [],
        }
    })
    writer = StaticJsonClient(json.dumps({"variants": []}))
    planner = SequenceClient([bad, good])
    result = run_v6_rewrite("This model no longer fully reflects how people learn.", planner_client=planner, writer_client=writer)
    assert len(planner.calls) == 2
    assert result.plan.ai_safe_route["llm_planner_decision"]["paragraph_route"] == "good"


def test_v6_planner_retries_when_contract_uses_planning_labels():
    bad = json.dumps({
        "planner_decision": {
            "paragraph_route": "bad",
            "finding_contracts": [
                {
                    "finding_id": "p001_s001:packed_list",
                    "source_sentence_id": "p001_s001",
                    "finding_tags": ["packed_list"],
                    "unsafe_original_shape": "A, B, and C",
                    "safe_rebuild_shape": "A carries one relation and B carries the next relation.",
                    "writer_must_do": ["split relation"],
                    "writer_must_not_do": [],
                    "coverage_terms": ["A", "B"],
                }
            ],
            "paragraph_blueprint": [],
        }
    })
    good = json.dumps({
        "planner_decision": {
            "paragraph_route": "good",
            "finding_contracts": [
                {
                    "finding_id": "p001_s001:packed_list",
                    "source_sentence_id": "p001_s001",
                    "finding_tags": ["packed_list"],
                    "unsafe_original_shape": "A, B, and C",
                    "safe_rebuild_shape": "A handled the first part. B handled the next part.",
                    "writer_must_do": ["split list"],
                    "writer_must_not_do": ["use three-item list"],
                    "coverage_terms": ["A", "B"],
                }
            ],
            "paragraph_blueprint": [],
        }
    })
    planner = SequenceClient([bad, good])
    writer = StaticJsonClient(json.dumps({"variants": []}))
    result = run_v6_rewrite("A, B, and C shaped the process.", planner_client=planner, writer_client=writer)
    assert len(planner.calls) == 2
    assert result.plan.ai_safe_route["llm_planner_decision"]["paragraph_route"] == "good"
