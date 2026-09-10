"""P0 voice-match spike tests (DRAFTPROOF_V6_VOICE_MATCH)."""

import os

import pytest

from .voice_profile import build_voice_profile, render_voice_contract, voice_match_enabled
from . import direct_rewrite

NORMAL = (
    "The migration project required careful coordination between three teams over several months. "
    "We underestimated how much legacy documentation was missing, which delayed the first milestone. "
    "Our team eventually rebuilt the dependency map by interviewing the original maintainers directly. "
    "This experience taught me that institutional knowledge often lives in people, not repositories. "
    "Future projects should budget explicit time for that discovery work before committing to dates."
)

SHORT = "This is short. Only three sentences here. Not enough voice data yet."


def test_profile_abstains_on_short_text():
    profile = build_voice_profile(SHORT)
    assert profile is not None
    assert profile["abstain"] is True
    assert profile["n_sentences"] < 4
    assert render_voice_contract(profile) == []


def test_profile_non_abstain_on_normal_paragraph():
    profile = build_voice_profile(NORMAL)
    assert profile is not None
    assert profile["abstain"] is False
    assert profile["n_sentences"] >= 4
    for key in ("formality_mean", "formality_spread", "sent_len_buckets",
                "first_person_density", "hedging_density", "lexical_diversity"):
        assert key in profile
    assert sum(profile["sent_len_buckets"].values()) == profile["n_sentences"]
    assert profile["first_person_density"] > 0  # source is first-person


def test_contract_mentions_plain_vs_academic_correctly():
    plain = {"abstain": False, "formality_mean": 0.3, "sent_len_buckets": {"short": 2, "med": 1, "long": 1},
             "first_person_density": 0.02, "hedging_density": 0.0, "lexical_diversity": 0.7}
    academic = {"abstain": False, "formality_mean": 0.8, "sent_len_buckets": {"short": 0, "med": 2, "long": 2},
                "first_person_density": 0.0, "hedging_density": 0.05, "lexical_diversity": 0.6}
    plain_contract = render_voice_contract(plain)
    academic_contract = render_voice_contract(academic)
    assert 3 <= len(plain_contract) <= 5 and 3 <= len(academic_contract) <= 5
    assert "plain" in plain_contract[0].lower() and "academic prose" in plain_contract[0].lower()
    assert "academic" in academic_contract[0].lower() and "plain" not in academic_contract[0].lower()
    assert "first-person voice" in " ".join(plain_contract).lower()
    assert "impersonal" in " ".join(academic_contract).lower()


def test_kill_switch_off_disables_contract(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_VOICE_MATCH", "0")
    assert voice_match_enabled() is False
    prompt = direct_rewrite._prompt(NORMAL, None, [])
    assert "[voice-match]" not in prompt
    assert "Write in PLAIN, direct, everyday voice" in prompt
    monkeypatch.delenv("DRAFTPROOF_V6_VOICE_MATCH")
    assert voice_match_enabled() is True
    prompt_on = direct_rewrite._prompt(NORMAL, None, [])
    assert "[voice-match]" in prompt_on
    assert "Write in PLAIN, direct, everyday voice" not in prompt_on


def test_fail_open_on_empty_string():
    assert build_voice_profile("") is None
    assert build_voice_profile(None) is None
    assert render_voice_contract(None) == []
    assert direct_rewrite._voice_contract_for("") == []
    # Internal error path also fails open (monkeypatch build to raise).
    orig = direct_rewrite.build_voice_profile
    direct_rewrite.build_voice_profile = lambda t: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        assert direct_rewrite._voice_contract_for(NORMAL) == []
    finally:
        direct_rewrite.build_voice_profile = orig
