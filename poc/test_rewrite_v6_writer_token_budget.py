"""Writer LLM token budget. gpt-oss spends reasoning tokens FIRST and they count toward max_tokens;
a content-heavy paragraph starved at 4000 (reasoning_tokens=3997, finish_reason=length) -> empty
content -> EmptyLLMContentError -> source_preserved (original kept, no rewrite shown). Same
starvation class as the QC reviewer. The budget must cover reasoning + output and be tunable."""
from poc.rewrite_v6 import direct_rewrite


def test_writer_max_tokens_default_and_env(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_WRITER_MAX_TOKENS", raising=False)
    assert direct_rewrite._writer_max_tokens() == 16000
    monkeypatch.setenv("DRAFTPROOF_V6_WRITER_MAX_TOKENS", "12000")
    assert direct_rewrite._writer_max_tokens() == 12000
    # invalid / non-positive values fall back to the default
    for bad in ("0", "-5", "abc", "", "  "):
        monkeypatch.setenv("DRAFTPROOF_V6_WRITER_MAX_TOKENS", bad)
        assert direct_rewrite._writer_max_tokens() == 16000
