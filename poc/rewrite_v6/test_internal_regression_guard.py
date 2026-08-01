"""Focused unit tests for the internal no-regression guard (incident 5bacaeb3)."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewrite_v6.production import (  # noqa: E402
    _internal_regression_card,
    _internal_regression_guard,
    _regression_guard_advisory,
)

ENV_KEYS = (
    "DRAFTPROOF_V6_REGRESSION_GUARD",
    "DRAFTPROOF_V6_REGRESSION_GUARD_ADVISORY",
    "DRAFTPROOF_V6_REGRESSION_GUARD_MARGIN",
    "DRAFTPROOF_V6_REGRESSION_GUARD_FLOOR",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_incident_case_triggers():
    guard = _internal_regression_guard({"original_ai": 13.38, "rewritten_ai": 66.05})
    assert guard["triggered"] is True
    assert guard["reason"] == "rewritten_ai_regressed_vs_original"
    assert guard["delta"] == pytest.approx(52.67)


def test_no_trigger_below_margin_and_below_floor():
    # improvement
    assert _internal_regression_guard({"original_ai": 70.0, "rewritten_ai": 30.0})["triggered"] is False
    # big delta but under the absolute floor
    assert _internal_regression_guard({"original_ai": 5.0, "rewritten_ai": 45.0})["triggered"] is False
    # over the floor but under the margin
    assert _internal_regression_guard({"original_ai": 55.0, "rewritten_ai": 70.0})["triggered"] is False


def test_advisory_default_on_and_revert_mode(monkeypatch):
    assert _regression_guard_advisory() is True
    monkeypatch.setenv("DRAFTPROOF_V6_REGRESSION_GUARD_ADVISORY", "0")
    assert _regression_guard_advisory() is False
    card = _internal_regression_card(
        {"original_ai": 13.38, "rewritten_ai": 66.05, "reverted": True}
    )
    assert card["card_id"] == "internal-regression-guard"
    assert "13.38" in card["instruction"] and "66.05" in card["instruction"]
    assert "original text has been kept" in card["instruction"]


def test_kill_switch_off(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_REGRESSION_GUARD", "0")
    guard = _internal_regression_guard({"original_ai": 13.38, "rewritten_ai": 66.05})
    assert guard == {"triggered": False, "status": "disabled"}


def test_missing_or_none_scores_skip_silently():
    for payload in ({}, None, {"original_ai": None, "rewritten_ai": 66.05},
                    {"original_ai": 13.38}, {"original_ai": "n/a", "rewritten_ai": 66.05}):
        guard = _internal_regression_guard(payload)
        assert guard["triggered"] is False
        assert guard["status"] == "scores_unavailable"


def test_env_thresholds_are_configurable(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_REGRESSION_GUARD_MARGIN", "10")
    monkeypatch.setenv("DRAFTPROOF_V6_REGRESSION_GUARD_FLOOR", "40")
    guard = _internal_regression_guard({"original_ai": 30.0, "rewritten_ai": 41.0})
    assert guard["triggered"] is True
    assert guard["margin"] == 10.0 and guard["floor"] == 40.0
