from __future__ import annotations

from types import SimpleNamespace

from poc.rewrite_v6 import full_doc_rewrite as fdr
from poc.rewrite_v6.llm_config import full_doc_rewrite_gateway
from poc.rewrite_v6.pipeline import DocumentResult
from poc.rewrite_v6.scan import scan_text
from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6 import llm_config


class _Gateway:
    def __init__(self, text: str):
        self.text = text

    def chat(self, *_args, **_kwargs):
        return SimpleNamespace(content=self.text, raw_content=self.text)


def test_full_doc_rewrite_accepts_score_improving_candidate(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE", "1")
    monkeypatch.setattr(fdr, "_score", lambda text: 5.0 if "better" in text else 9.0)

    out, result = fdr.apply_full_doc_rewrite(
        "I taught 12 students.\n\nThey tested a draft.",
        gateway=_Gateway("I taught 12 students better.\n\nThey tested a draft."),
    )

    assert out == "I taught 12 students better.\n\nThey tested a draft."
    assert result.changed is True
    assert result.status == "accepted"


def test_full_doc_rewrite_rejects_number_change_even_when_score_improves(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE", "1")
    monkeypatch.setattr(fdr, "_score", lambda text: 5.0 if "15 students" in text else 9.0)

    out, result = fdr.apply_full_doc_rewrite(
        "I taught 12 students.\n\nThey tested a draft.",
        gateway=_Gateway("I taught 15 students.\n\nThey tested a draft."),
    )

    assert out == "I taught 12 students.\n\nThey tested a draft."
    assert result.changed is False
    assert result.status == "rejected"
    assert "numbers_changed" in result.reasons


def test_full_doc_rewrite_rejects_duplicate_number_count_change(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE", "1")
    monkeypatch.setattr(fdr, "_score", lambda text: 8.0)

    out, result = fdr.apply_full_doc_rewrite(
        "I opened the 2023 packet. The 2023 packet repeated the same line.",
        gateway=_Gateway("I opened the 2023 packet and noticed the same line repeated."),
    )

    assert out == "I opened the 2023 packet. The 2023 packet repeated the same line."
    assert result.changed is False
    assert result.status == "rejected"
    assert result.metrics["numbers_preserved"] is True
    assert result.metrics["number_multiset_preserved"] is False
    assert "number_count_changed" in result.reasons


def test_full_doc_rewrite_allows_small_score_regression(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE", "1")
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE_SCORE_TOLERANCE", "5")
    monkeypatch.setattr(fdr, "_score", lambda text: 13.0 if "candidate" in text else 9.0)

    out, result = fdr.apply_full_doc_rewrite(
        "I taught 12 students.\n\nThey tested a draft.",
        gateway=_Gateway("I taught 12 students candidate.\n\nThey tested a draft."),
    )

    assert out == "I taught 12 students candidate.\n\nThey tested a draft."
    assert result.changed is True
    assert result.score_before == 9.0
    assert result.score_after == 13.0


def test_full_doc_rewrite_rejects_large_score_regression(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE", "1")
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE_SCORE_TOLERANCE", "5")
    monkeypatch.setattr(fdr, "_score", lambda text: 16.0 if "candidate" in text else 9.0)

    out, result = fdr.apply_full_doc_rewrite(
        "I taught 12 students.\n\nThey tested a draft.",
        gateway=_Gateway("I taught 12 students candidate.\n\nThey tested a draft."),
    )

    assert out == "I taught 12 students.\n\nThey tested a draft."
    assert result.changed is False
    assert "score_regressed" in result.reasons


def test_apply_full_doc_rewrite_wires_trace_and_rescan(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE", "1")
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE_MODEL", "provider/test")
    monkeypatch.setattr(fdr, "_score", lambda text: 4.0 if "better" in text else 8.0)

    class _ConfiguredGateway:
        def chat(self, *_args, **_kwargs):
            return SimpleNamespace(
                content="I taught 12 students better.\n\nThey tested a draft.",
                raw_content="I taught 12 students better.\n\nThey tested a draft.",
            )

    monkeypatch.setattr(llm_config, "full_doc_rewrite_gateway", lambda **_kwargs: _ConfiguredGateway())
    doc = DocumentResult(
        initial_scan=scan_text("I taught 12 students.\n\nThey tested a draft."),
        final_scan=scan_text("I taught 12 students.\n\nThey tested a draft."),
        passes=[],
        rewritten_text="I taught 12 students.\n\nThey tested a draft.",
        pass_trace=[],
    )

    out = dr._apply_full_doc_rewrite(doc, api_key=None, base_url=None, cancellation_check=None)

    assert "better" in out.rewritten_text
    assert out.final_scan.source_text == out.rewritten_text
    assert out.pass_trace[-1]["selected_source"] == "full_doc_rewrite"
    assert out.pass_trace[-1]["status"] == "accepted"


def test_full_doc_rewrite_gateway_uses_role_specific_openrouter_config(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_CEREBRAS_DIRECT", "1")
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE_MODEL", "provider/full-doc-model")
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("DRAFTPROOF_V6_FULL_DOC_REWRITE_API_KEY", "role-key")

    gateway = full_doc_rewrite_gateway(api_key="writer-key", base_url="https://api.cerebras.ai/v1")

    assert gateway is not None
    assert gateway.model == "provider/full-doc-model"
    assert gateway.base_url == "https://openrouter.ai/api/v1"
    assert gateway.api_key == "role-key"
