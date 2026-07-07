"""Capture per-doc V7 signal vectors (fused path, cached Modal) so weight
tuning replays category_scoring.score_paragraph offline — no re-scans.

Numbers-only output (signals + label + text hash); corpus text never leaves
the gitignored captures/ dir. Usage:
    python -m calibration.v12_validation.capture_signals            # fused
    python -m calibration.v12_validation.capture_signals --quick    # no Modal
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "captures" / "signals_fused.jsonl"


def capture_one(runner, text: str, label: str) -> tuple[dict, dict]:
    """Run the real scan once; return (capture_row, full_report).

    Mirrors detect_v7.pipeline_bridge.run_v7_breakdown's exact composition
    (_extract_calibrated_score -> get_deep_scan_proportion -> detector_scores
    -> compute_calibrated_detector_score -> _build_raw_signals ->
    adapt_paragraph_signals) so the captured (v7_signals,
    calibrated_detector_score) pair replays offline through
    category_scoring.score_paragraph to the SAME primary_category the
    end-to-end pipeline produced for this text.
    """
    from detect.document_structure import normalize_submitted_text
    from detect_v7 import detector_fusion, pipeline_bridge, signal_adapter
    from report.report import ReportBuilder, report_to_dict

    # Inlined from calibration.measure_end_to_end.scan_text (rather than
    # calling it) so we can keep a handle on the raw DetectionResult:
    # detection_result.criterion_scores never round-trips through
    # report_to_dict (report.py has no such key) and the fused badge doesn't
    # carry it either, so it has to be pulled from the same source
    # ReportBuilder reads it from, before the report dict is built. Exactly
    # one scan runs (runner.run_all) — no duplicate detection.
    norm_text = normalize_submitted_text(str(text or ""))
    detection_result = runner.run_all(norm_text)
    b = ReportBuilder()
    b.add_detection_report(detection_result)
    if getattr(detection_result, "postprocess_results", None):
        b.add_postprocess_results(detection_result.postprocess_results)
    b.set_meta(scan_time=0.0, original_text=norm_text)
    rep = report_to_dict(b.build())

    badge = dict(rep.get("ai_risk_badge") or {})
    # ai_risk_badge carries document-level scores only, not raw text (see
    # pipeline_bridge._extract_document_text docstring) — attach it under one
    # of the recognized candidate keys so get_deep_scan_proportion can reach
    # Modal on the --fused (non-quick) path.
    det = {**badge, "document_text": text}
    # Mirror builder.py L370-371's exact guard for surfacing
    # criterion_scores into the dict handed downstream (there:
    # run_v7_breakdown's input; here: pipeline_bridge._build_raw_signals'
    # input) — same source, same condition, so capture parity holds.
    if hasattr(detection_result, "criterion_scores") and detection_result.criterion_scores:
        det["criterion_scores"] = detection_result.criterion_scores

    composite = pipeline_bridge._extract_calibrated_score(det)
    detector_scores = {"composite": composite}
    deep = pipeline_bridge.get_deep_scan_proportion(det)  # None on quick-scan
    if deep is not None:
        # run_v7_breakdown adds deberta_large unconditionally whenever a deep
        # scan result exists (no below_floor/uncalibrated gate at this call
        # site) — mirror that exactly, don't invent a gate here.
        detector_scores["deberta_large"] = deep["proportion"]
    calibrated, _detail = detector_fusion.compute_calibrated_detector_score(detector_scores)

    raw = pipeline_bridge._build_raw_signals(det)
    v7_signals = signal_adapter.adapt_paragraph_signals(raw)
    row = {
        "label": label,
        "doc_key": hashlib.sha256(text.encode()).hexdigest()[:16],
        "v7_signals": v7_signals,
        "calibrated_detector_score": calibrated,
        "has_comparison_text": False,
    }
    return row, rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="quick-scan path, no Modal")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    os.environ["DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN"] = "1"
    from calibration.v12_validation import measure as m
    if args.quick:
        os.environ.pop("DRAFTPROOF_V7_DEEP_SCAN", None)
    else:
        os.environ["DRAFTPROOF_V7_DEEP_SCAN"] = "1"
        from calibration.retune.deepscan_cache import DEFAULT_CACHE
        m.install_cached_deep_scan(DEFAULT_CACHE)
    try:
        # measure_end_to_end installs the sys.path shim that makes the
        # detect stack's absolute `poc.*` imports resolvable when running
        # as `python -m calibration...` from poc/ — it must load BEFORE
        # detect.run (capture_one gets it for free; this path does not).
        import calibration.measure_end_to_end  # noqa: F401
        from calibration.retune.intake import DEFAULT_SCOCESLE
        from detect.run import DetectionRunner
        manifest = json.loads((m.CORPUS_DIR / "manifest.json").read_text())
        by_class: dict[str, list] = defaultdict(list)
        for r in manifest["rows"]:
            by_class[r["label"]].append(r)
        runner = DetectionRunner()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with args.out.open("w", encoding="utf-8") as f:
            for label in m._CLASSES:
                rows = by_class.get(label, [])
                if args.limit:
                    rows = rows[: args.limit]
                for r in rows:
                    text = m._resolve_text(r, DEFAULT_SCOCESLE)
                    if not text:
                        continue
                    row, _ = capture_one(runner, text, label)
                    f.write(json.dumps(row) + "\n")
                    n += 1
                    print(".", end="", flush=True)
        print(f"\ncaptured {n} docs -> {args.out}")
    finally:
        if not args.quick:
            m.uninstall_cached_deep_scan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
