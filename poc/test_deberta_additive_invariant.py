"""Additive-invariant gate for the DeBERTa-signal module.

THE safety test for this module: turning the DeBERTa field ON must NOT change any
decision-driving field (overall_tier, badge tier, ai_likelihood_score, review_priority,
rewrite_decision). The only allowed difference is the new `ai_signal_deberta` key appearing
on the badge. If this test ever fails, the module is leaking into the authoritative path and
must not ship.

Uses the canonical scan path (calibration.measure_end_to_end.scan_text) and mocks
score_windows so no 400MB model is loaded — this tests WIRING/additivity, not model quality
(model quality is the Phase-0 SCoCESLE gate's job).
"""
import json
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent  # poc/
for _p in (str(_HERE), str(_HERE.parent)):  # poc/ + repo root (detect.*, poc.* both resolvable)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detect.run import DetectionRunner  # noqa: E402
from detect import deberta_signal  # noqa: E402
from calibration.measure_end_to_end import scan_text  # noqa: E402

# >150 words so the field is available (not abstained) on the ON path.
TEXT = " ".join(
    ["Effective academic writing demands a clear thesis statement supported by concrete evidence."]
    * 30
)


def _stub_prob(_windows):
    return [0.9] * len(_windows)


def _scan(runner):
    return scan_text(runner, TEXT)


def test_additive_invariant():
    runner = DetectionRunner()
    deberta_signal.score_windows = _stub_prob  # mock: no model load, deterministic

    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "0"
    off = _scan(runner)
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    on = _scan(runner)

    off_badge = off.get("ai_risk_badge") or {}
    on_badge = on.get("ai_risk_badge") or {}

    # 1. Headline tier unchanged.
    assert off["overall_tier"] == on["overall_tier"], (
        f"overall_tier changed: {off['overall_tier']} -> {on['overall_tier']}"
    )

    # 2. Every decision-driving badge field unchanged.
    for key in ("tier", "ai_likelihood_score", "writing_quality_score", "review_priority"):
        assert off_badge.get(key) == on_badge.get(key), (
            f"badge[{key}] changed: {off_badge.get(key)} -> {on_badge.get(key)}"
        )

    # 3. Rewrite decision unchanged (if present).
    if "rewrite_decision" in off or "rewrite_decision" in on:
        assert json.dumps(off.get("rewrite_decision"), sort_keys=True, default=str) == \
               json.dumps(on.get("rewrite_decision"), sort_keys=True, default=str), \
               "rewrite_decision changed"

    # 4. The ONLY new badge key is ai_signal_deberta (nothing else added/removed).
    assert set(on_badge) - set(off_badge) == {"ai_signal_deberta"}, (
        f"unexpected new keys: {set(on_badge) - set(off_badge)}"
    )
    assert set(off_badge) - set(on_badge) == set(), (
        f"keys dropped: {set(off_badge) - set(on_badge)}"
    )

    # 5. The new field is a valid schema and present only when ON.
    assert "ai_signal_deberta" in on_badge
    assert "ai_signal_deberta" not in off_badge
    sig = on_badge["ai_signal_deberta"]
    assert sig["available"] is True
    # v2 schema: band is insufficient|amber|orange|red (no green); model_version present.
    assert sig["band"] in {"insufficient", "amber", "orange", "red"}
    assert sig["model_version"] == "deberta_signal_v2"
    assert isinstance(sig.get("flagged_passages"), list)


if __name__ == "__main__":
    test_additive_invariant()
    print("test_additive_invariant PASSED")
    print("ADDITIVE INVARIANT HOLDS — DeBERTa field does not leak into any decision field.")
