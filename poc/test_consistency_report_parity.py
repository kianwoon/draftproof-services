"""Report-level kill-switch parity for the Phase-1 ConsistencyDetector wiring.

Task 3: proves (1) DetectionRunner.run_all's report is byte-identical whether
DRAFTPROOF_CONSISTENCY is unset or explicitly "0" (the detector is a pure no-op in
both OFF states), and (2) that guarantee is STRUCTURAL -- ConsistencyDetector is
never constructed at all when OFF, not merely "constructed but silent" -- proven via
an instantiation-counting spy substituted into detect.run's own namespace (the name
DetectionRunner._build_detectors actually calls), plus a positive control (env="1")
showing the same spy DOES fire, so the negative assertions aren't vacuously true.

Runs the REAL DetectionRunner.run_all end-to-end (not a reimplementation) -- see the
lighter-approach lesson in test_claim_graph_report_parity.py's docstring. A raw
dataclasses.asdict(report) diff across two independent (but env-identical) run_all
calls on identical input is NOT byte-identical even without touching this switch --
predictability_cache hit/miss/store counters, raw.scan_seconds timing, and
RewriteDecision.targets' id()-based fallback finding IDs (poc/detect/run.py
_compute_rewrite_decision) are all pre-existing, unrelated sources of run-to-run
non-determinism (verified empirically before writing this test). _normalize_report
strips exactly those volatile fields so the comparison isolates the kill-switch's
effect from that pre-existing noise.

allow-hardcode: _TEXT below is fixed prose fed to DetectionRunner.run_all as TEST
INPUT TEXT to exercise the real detector pipeline end-to-end -- not a
matching/scoring list consumed by production code.
"""
from __future__ import annotations

import dataclasses
import json

from detect.consistency import CONSISTENCY_KILL_SWITCH_ENV
from detect.profiles import resolve_profile
from detect.run import DetectionRunner

_TEXT = (
    "The committee reviewed the proposal carefully before reaching a decision. "
    "Because the budget projections were incomplete, several members requested "
    "additional documentation. However, the overall direction of the plan "
    "remained sound and defensible. Therefore, the group agreed to proceed with "
    "a revised timeline that addressed the outstanding concerns."
)


def _normalize_report(report) -> dict:
    """Strip pre-existing (unrelated) run-to-run non-determinism -- see module
    docstring -- so the comparison isolates the DRAFTPROOF_CONSISTENCY switch."""
    d = dataclasses.asdict(report)
    d.pop("postprocess_results", None)
    for scanner_result in d.get("scanner_results", []):
        scanner_result.pop("raw", None)
    rewrite_decision = d.get("rewrite_decision")
    if rewrite_decision is not None:
        rewrite_decision["targets"] = len(rewrite_decision.get("targets") or [])
    return d


def _run_all(monkeypatch, env_value):
    if env_value is None:
        monkeypatch.delenv(CONSISTENCY_KILL_SWITCH_ENV, raising=False)
    else:
        monkeypatch.setenv(CONSISTENCY_KILL_SWITCH_ENV, env_value)
    report = DetectionRunner().run_all(_TEXT, auto_extract_bibliography=False)
    return _normalize_report(report)


def test_report_byte_identical_unset_vs_explicit_off(monkeypatch):
    unset = _run_all(monkeypatch, None)
    off = _run_all(monkeypatch, "0")

    assert json.dumps(unset, sort_keys=True, default=str) == json.dumps(
        off, sort_keys=True, default=str
    )
    scanners = {r["scanner"] for r in unset["scanner_results"]}
    assert "consistency" not in scanners


class _InstantiationSpy:
    """Stand-in for ConsistencyDetector that records every construction."""

    call_count = 0

    def __init__(self, *args, **kwargs):
        type(self).call_count += 1

    def detect(self, content, **kwargs):  # pragma: no cover - never expected to run
        raise AssertionError("spy ConsistencyDetector.detect() should never be called")

    @property
    def name(self):  # pragma: no cover
        return "consistency"


def test_consistency_detector_never_instantiated_when_off(monkeypatch):
    _InstantiationSpy.call_count = 0
    monkeypatch.setattr("detect.run.ConsistencyDetector", _InstantiationSpy)

    monkeypatch.delenv(CONSISTENCY_KILL_SWITCH_ENV, raising=False)
    DetectionRunner().run_all(_TEXT, auto_extract_bibliography=False)
    assert _InstantiationSpy.call_count == 0, "unset must not construct ConsistencyDetector"

    monkeypatch.setenv(CONSISTENCY_KILL_SWITCH_ENV, "0")
    DetectionRunner().run_all(_TEXT, auto_extract_bibliography=False)
    assert _InstantiationSpy.call_count == 0, "explicit '0' must not construct ConsistencyDetector"


def test_consistency_detector_instantiated_when_on_positive_control(monkeypatch):
    # Positive control: proves the spy substitution actually intercepts
    # _build_detectors' construction call, so the negative assertions above are not
    # vacuously true because the spy was never wired to anything real. Calls
    # _build_detectors directly (not run_all) so the spy's intentionally-raising
    # detect() is never invoked.
    _InstantiationSpy.call_count = 0
    monkeypatch.setattr("detect.run.ConsistencyDetector", _InstantiationSpy)
    monkeypatch.setenv(CONSISTENCY_KILL_SWITCH_ENV, "1")

    profile = resolve_profile(_TEXT, "default")
    detectors = DetectionRunner()._build_detectors(profile=profile, auto_extract_bibliography=False)

    assert _InstantiationSpy.call_count == 1
    assert any(isinstance(d, _InstantiationSpy) for d in detectors)
