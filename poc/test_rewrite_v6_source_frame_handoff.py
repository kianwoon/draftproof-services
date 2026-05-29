from __future__ import annotations

import json

from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.planner_llm import build_planner_prompt, run_planner_llm
from poc.rewrite_v6.scan import scan_text
from poc.rewrite_v6.source_quality import created_effect_frame_meanings
from poc.rewrite_v6.write import build_prompt


class StaticJsonClient:
    def __init__(self, payload: str):
        self.payload = payload

    def chat(self, *_args, **_kwargs):
        return type("Response", (), {"content": self.payload, "raw_content": self.payload})()


def _source() -> str:
    return (
        "Technology has also created both opportunities and risks. "
        "AI tools can help students brainstorm ideas, explain difficult topics, improve writing, and practise skills. "
        "Used well, technology can support personalised learning and reduce barriers for students who need extra help. "
        "But there is also a danger. "
        "Students may become too dependent on AI-generated answers without understanding the work behind them. "
        "They may submit polished work that does not reflect their real ability. "
        "This makes assessment more difficult and raises questions about fairness, originality, and learning integrity."
    )


def test_v6_planner_prompt_exposes_created_effect_frame_as_route_constraint():
    scan = scan_text("Technology has also created both opportunities and risks. AI tools can help students brainstorm ideas.")
    paragraph, plan = build_plan(scan)

    payload = json.loads(build_planner_prompt(paragraph, plan, scan.findings).split("\n", 1)[1])
    policy = payload["risky_source_shape_policy"]

    assert policy["source_frame_constraints"]
    assert policy["source_frame_constraints"][0]["subject"] == "Technology"
    assert "Do not turn the subject/create relation into a concrete entity-creation route." in policy["source_frame_rule"]


def test_v6_source_frame_detection_ignores_coordinated_clause_tail():
    assert created_effect_frame_meanings("The review process has created pressure and uncertainty.")
    assert not created_effect_frame_meanings(
        "Used well, technology can support personalised learning and reduce barriers for students who need extra help."
    )
    assert not created_effect_frame_meanings(
        "This makes assessment more difficult and raises questions about fairness, originality, and learning integrity."
    )


def test_v6_planner_merge_removes_created_frame_mechanics_from_opening_beat():
    scan = scan_text(_source())
    paragraph, plan = build_plan(scan)
    planner = StaticJsonClient(json.dumps({"planner_decision": {}}))

    planned = run_planner_llm(paragraph, plan, scan.findings, client=planner)
    opening_beat = planned.ai_safe_route["llm_planner_decision"]["coverage_beats"][0]
    beat_text = json.dumps(opening_beat, ensure_ascii=False).casefold()

    assert "technology" not in {str(item).casefold() for item in opening_beat["must_cover"]}
    assert "created" not in {str(item).casefold() for item in opening_beat["must_cover"]}
    assert any("alongside" in str(item) for item in opening_beat["must_cover"])
    assert "technology creates" not in beat_text


def test_v6_writer_handoff_removes_created_frame_mechanics_from_execution_plan():
    scan = scan_text(_source())
    paragraph, plan = build_plan(scan)
    planner = StaticJsonClient(json.dumps({"planner_decision": {}}))

    planned = run_planner_llm(paragraph, plan, scan.findings, client=planner)
    payload = json.loads(build_prompt(paragraph, planned).split("\n", 1)[1])
    opening_step = payload["writer_execution_plan"][0]
    opening_targets = {str(item).casefold() for item in opening_step["meaning_targets"]}
    final_step = payload["writer_execution_plan"][-1]
    must_keep = {str(item).casefold() for item in payload["must_keep_terms"]}

    assert "technology" not in opening_targets
    assert "created" not in opening_targets
    assert "technology" not in must_keep
    assert "created" not in must_keep
    assert any("alongside" in item for item in opening_targets)
    assert "do not return to this step" in opening_step["step_order_rule"]
    assert "paragraph must end here" in final_step["step_order_rule"]
    assert "Do not rewrite it as the subject creating a concrete tool" in json.dumps(
        payload["source_frame_constraints"],
        ensure_ascii=False,
    )
