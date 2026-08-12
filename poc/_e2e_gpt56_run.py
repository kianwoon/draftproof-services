"""One-off E2E scan->rewrite run for GPT-5.6 casual case under the new fine-tune v1 detector."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detect.run import DetectionRunner
from detect.document_structure import normalize_submitted_text
from report.report import ReportBuilder, report_to_dict
from rewrite_v6.production import run_rewrite_pipeline_v6

CASE = Path(__file__).resolve().parent / "calibration/authorship_cases/ai_gpt_5_6_user_casual_00.json"
OUT = Path(__file__).resolve().parent / "test_output/_e2e_gpt56_run"


def scan_text(runner, text):
    text = normalize_submitted_text(str(text or ""))
    det = runner.run_all(text)
    b = ReportBuilder()
    b.add_detection_report(det)
    if getattr(det, "postprocess_results", None):
        b.add_postprocess_results(det.postprocess_results)
    b.set_meta(scan_time=0.0, original_text=text)
    return report_to_dict(b.build())


def extract_badge(rep: dict) -> dict:
    badge = rep.get("badge") or {}
    ta = rep.get("tier_authority") or badge.get("tier_authority") or {}
    return {
        "ai_likelihood_score": badge.get("ai_likelihood_score"),
        "tier": badge.get("tier") or rep.get("tier"),
        "fused": ta.get("fused") if isinstance(ta, dict) else None,
        "composite": ta.get("composite") if isinstance(ta, dict) else None,
        "proportion": ta.get("proportion") if isinstance(ta, dict) else None,
        "status": ta.get("status") if isinstance(ta, dict) else None,
    }


def main():
    d = json.loads(CASE.read_text())
    text = d["text"]
    print(f"case_id={d.get('case_id')} words={d.get('words')}")

    runner = DetectionRunner()
    t0 = time.time()
    before_report = scan_text(runner, text)
    t1 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "before_report.json").write_text(json.dumps(before_report, indent=2))
    before_badge = extract_badge(before_report)
    print(f"BEFORE scan_time={t1-t0:.1f}s badge={before_badge}")

    t2 = time.time()
    envelope = run_rewrite_pipeline_v6(detect_json=before_report, output_dir=str(OUT / "rewrite"))
    t3 = time.time()
    result = json.loads(Path(envelope["json_path"]).read_text())
    (OUT / "rewrite_result.json").write_text(json.dumps(result, indent=2))

    print(f"REWRITE runtime={t3-t2:.1f}s")
    print("keys:", sorted(result.keys()))
    print(f"final_risk={result.get('final_risk')} original_risk={result.get('original_risk')}")
    print(f"outcome/result_label={result.get('outcome')}/{result.get('result_label')}")
    print(f"ai_mitigated={result.get('ai_mitigated')} score_worse={result.get('score_worse')}")

    after_badge = result.get("after_badge") or result.get("badge_after") or {}
    print(f"after_badge={after_badge}")

    rewritten_text = result.get("rewritten_text") or result.get("final_text") or ""
    print("REWRITTEN SNIPPET:", rewritten_text[:400])


if __name__ == "__main__":
    main()
