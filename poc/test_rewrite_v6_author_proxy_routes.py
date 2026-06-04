import json

from poc.rewrite_v6 import direct_rewrite
from poc.rewrite_v6.author_proxy_routes import select_author_proxy_routes


def _payload(text, diagnosis=None, tags=None):
    prompt = direct_rewrite._prompt(
        text,
        diagnosis or {"main_issue": "generic broad claim", "recommendation": "ground the claim"},
        tags or ["author_anchor_gap", "context_anchor_gap"],
        lane="diversified",
    )
    return json.loads(prompt.split("\n", 1)[1])


def test_generic_anchor_gap_returns_non_numeric_author_proxy_routes():
    routes = select_author_proxy_routes(
        "The process creates concern because teams need better review before approval.",
        {"main_issue": "generic broad claim", "recommendation": "ground the claim"},
        ["author_anchor_gap", "context_anchor_gap", "broad_claim"],
    )
    modes = [route["mode"] for route in routes]

    assert len([mode for mode in modes if mode != "scale_detail"]) >= 3
    assert any(mode in modes for mode in ["observed_process", "decision_moment", "actor_interaction", "condition_trigger"])
    assert "scale_detail" not in modes


def test_scale_detail_is_optional_and_ordered_last_when_source_has_scale():
    routes = select_author_proxy_routes(
        "Some teams repeat the check several times before approval.",
        {"main_issue": "generic broad claim"},
        ["author_anchor_gap"],
    )

    assert routes[-1]["mode"] == "scale_detail"


def test_diversified_prompt_removes_privileged_numeric_examples():
    payload = _payload("This concern matters because the process affects teams.")
    prompt_text = json.dumps(payload, ensure_ascii=False)

    assert "author_proxy_routes" in payload
    assert "about a third" not in prompt_text
    assert "representative figure" not in prompt_text
    assert "bare figure" not in prompt_text
    assert "author-owned vantage" in prompt_text
    assert "concrete particular" in prompt_text


def test_reflective_text_preserves_first_person_compatible_routes():
    routes = select_author_proxy_routes(
        "When I review the report, I compare the claim with the source notes.",
        {"main_issue": "generic broad claim"},
        ["author_anchor_gap"],
    )

    assert routes[0]["mode"] == "observed_process"


def test_formal_text_gets_non_forced_first_person_routes():
    routes = select_author_proxy_routes(
        "The institution uses a policy framework to review source evidence before approval.",
        {"main_issue": "generic broad claim"},
        ["context_anchor_gap"],
    )
    modes = [route["mode"] for route in routes]

    assert "actor_interaction" in modes or "source_use" in modes
    assert all("force" not in route["use_when"].casefold() for route in routes)
