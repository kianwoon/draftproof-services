"""Confirm the fragmentation cluster moves content11 out of green WITHOUT re-tiering
normal docs. Scan-only (no rewrite). Run via Bash:

    PYTHONPATH=. DRAFTPROOF_V6_DETERMINISTIC=1 python3 poc/_check_frag_scan.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("DRAFTPROOF_V6_DETERMINISTIC", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parent.parent
from poc.detect_pipeline import run_detect

# content11 = fragmented evasion (should re-tier); content9 = next-shortest control
# (mean 12.1, must NOT re-tier); content1/3 = normal prose controls.
for name in ["test_content11.txt", "test_content9.txt", "test_content1.txt", "test_content3.txt"]:
    t = (REPO / name).read_text()
    det = run_detect(t, output_dir=str(REPO / f"test_output/_frag_check/{name}"))
    dj = json.loads(Path(det["json_path"]).read_text())
    badge = dj.get("ai_risk_badge") or {}
    comps = badge.get("ai_components") or {}
    print(
        f"{name:20} tier={str(det['tier']):11} "
        f"ai_likelihood={badge.get('ai_likelihood_score')}  "
        f"gen_assert={comps.get('generic_assertion_risk')} topk={comps.get('topk_pattern')}"
    )
