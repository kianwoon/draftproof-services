"""Unit tests for the LLM-authored showcase case layer.

No network: a FakeClient is injected. These cover parsing, the guard rails (disabled / no real
change / LLM error all return [] so the rewrite is never blocked), and paragraph alignment.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from rewrite_v6.showcase_cases import (  # noqa: E402
    author_showcase_cases,
    changed_paragraph_pairs,
)

_GOOD_JSON = (
    '{"cases":[{"submitted_quote":"significant pedagogical challenges",'
    '"marker_sees":"Could sit in any education essay.",'
    '"move_label":"Abstraction -> observed instance",'
    '"rewritten_quote":"a quarter of learners stumble",'
    '"why_it_lands":"A verifiable particular only a teacher would know.",'
    '"your_rule":"Name what you actually saw, and how many."}]}'
)

ORIG = (
    "Introduction\n\n"
    "This unit presents significant pedagogical challenges for many learners in various settings."
)
FINAL = (
    "Introduction\n\n"
    "In my classes, I have seen about a quarter of learners stumble at Box Hill Institute."
)


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeClient:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def chat(self, prompt, **kwargs):
        self.calls += 1
        return _Resp(self._content)


class _RaisingClient:
    def __init__(self):
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("provider down")


def test_changed_pairs_skips_heading_unchanged_and_mismatch():
    pairs = changed_paragraph_pairs(ORIG, FINAL)
    assert len(pairs) == 1                       # the heading "Introduction" is skipped
    assert "significant pedagogical challenges" in pairs[0]["original"]
    # paragraph counts differ -> no mis-pairing
    assert changed_paragraph_pairs("a\n\nb", "a") == []
    # nothing changed -> no pairs
    assert changed_paragraph_pairs(FINAL, FINAL) == []


def test_author_parses_cases_from_client():
    client = _FakeClient(_GOOD_JSON)
    cases = author_showcase_cases(ORIG, FINAL, client=client)
    assert client.calls == 1
    assert len(cases) == 1
    c = cases[0]
    assert c["submitted_quote"] == "significant pedagogical challenges"
    assert c["rewritten_quote"] == "a quarter of learners stumble"
    assert c["your_rule"] and c["move_label"]


def test_no_real_change_returns_empty_without_calling_client():
    client = _RaisingClient()
    assert author_showcase_cases(FINAL, FINAL, client=client) == []
    assert client.calls == 0                     # short-circuits before any LLM call


def test_llm_error_returns_empty():
    assert author_showcase_cases(ORIG, FINAL, client=_RaisingClient()) == []


def test_disabled_flag_returns_empty():
    prev = os.environ.get("DRAFTPROOF_SHOWCASE_CASES_ENABLED")
    os.environ["DRAFTPROOF_SHOWCASE_CASES_ENABLED"] = "0"
    try:
        client = _FakeClient(_GOOD_JSON)
        assert author_showcase_cases(ORIG, FINAL, client=client) == []
        assert client.calls == 0
    finally:
        if prev is None:
            os.environ.pop("DRAFTPROOF_SHOWCASE_CASES_ENABLED", None)
        else:
            os.environ["DRAFTPROOF_SHOWCASE_CASES_ENABLED"] = prev


def test_incomplete_case_is_dropped():
    bad = '{"cases":[{"submitted_quote":"x","marker_sees":"y"}]}'  # no rewritten_quote / rule
    assert author_showcase_cases(ORIG, FINAL, client=_FakeClient(bad)) == []


def test_unescaped_inner_quotes_recovered_via_loose_parser():
    # The real gpt-oss prod failure: valid-looking JSON with UNescaped inner double-quotes that
    # breaks json.loads. The loose fallback must still recover the case.
    messy = (
        '{"cases":[{'
        '"submitted_quote":"Students are surrounded by information.",'
        '"marker_sees":"It is generic; readers cannot picture the "flood" of sources.",'
        '"move_label":"Abstraction -> observed instance",'
        '"rewritten_quote":"In my classroom, I see students arriving with a flood of articles.",'
        '"why_it_lands":"It names what the teacher actually sees.",'
        '"your_rule":"Turn vague claims about "information" into a concrete snapshot."}]}'
    )
    import json as _json
    try:
        _json.loads(messy)
        assert False, "fixture should be invalid JSON (unescaped inner quotes)"
    except ValueError:
        pass
    cases = author_showcase_cases(ORIG, FINAL, client=_FakeClient(messy))
    assert len(cases) == 1
    assert cases[0]["submitted_quote"] == "Students are surrounded by information."
    assert cases[0]["your_rule"].startswith("Turn vague claims about")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("ok")
