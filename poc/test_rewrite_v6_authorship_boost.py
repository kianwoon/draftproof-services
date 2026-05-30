"""The direct writer's prompt must carry protect+target instructions when an authorship_targets
brief is supplied, and omit them when it is empty (no behaviour change off-flag)."""
import json
from poc.rewrite_v6.direct_rewrite import _prompt


def test_prompt_includes_protected_and_grounding_when_supplied():
    targets = {"protected_spans": ["I taught in one room for years."],
               "grounding_targets": ["Tie your key claims to a source or citation"]}
    out = _prompt("Some flagged paragraph text.", None, ["generic_assertion"], targets)
    payload = json.loads(out.split("\n", 1)[1])
    assert "protected_spans" in payload and payload["protected_spans"] == targets["protected_spans"]
    assert "grounding_targets" in payload and payload["grounding_targets"] == targets["grounding_targets"]
    assert any("verbatim" in r.lower() for r in payload["instructions"])


def test_prompt_omits_fields_when_targets_empty():
    out = _prompt("Some flagged paragraph text.", None, ["generic_assertion"], {})
    payload = json.loads(out.split("\n", 1)[1])
    assert "protected_spans" not in payload and "grounding_targets" not in payload
