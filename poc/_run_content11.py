"""Ad-hoc: run the production V6 pipeline on test_content11.txt and dump
scan signals, the raw progress sequence, the rewrite status/risk, the
per-paragraph trace, and a before/after view. Run via Bash (needs network+model):

    DRAFTPROOF_V6_DETERMINISTIC=1 python3 poc/_run_content11.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("DRAFTPROOF_V6_DETERMINISTIC", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parent.parent
text = (REPO / "test_content11.txt").read_text()

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from poc.detect_pipeline import run_detect
from poc.rewrite_v6.production import run_rewrite_pipeline_v6

det = run_detect(text, output_dir=str(REPO / "test_output/_content11/scan"))
detect_json = json.loads(Path(det["json_path"]).read_text())

print("=" * 72)
print("SCAN")
print("  tier:", det["tier"], "| findings:", det["findings"])
badge = detect_json.get("ai_risk_badge") or {}
print("  ai_score:", detect_json.get("ai_score"), "| ai_likelihood:", badge.get("ai_likelihood_score"))
comps = badge.get("ai_components") or {}
top = sorted(
    ((k, v) for k, v in comps.items() if isinstance(v, (int, float))),
    key=lambda kv: kv[1],
    reverse=True,
)[:8]
print("  top ai_components:", [(k, round(v, 1)) for k, v in top])

prog = []
env = run_rewrite_pipeline_v6(
    detect_json=detect_json,
    output_dir=str(REPO / "test_output/_content11/rewrite"),
    progress_callback=lambda p, m: prog.append((p, m)),
)
result = json.loads(Path(env["json_path"]).read_text())
summary = result.get("summary") or {}

print("=" * 72)
print("PROGRESS SEQUENCE (raw v6 emissions, pre worker-clamp):")
for p, m in prog:
    print(f"   {p:>4}%  {m}")
seq = [p for p, _ in prog]
backsteps = [(seq[i - 1], seq[i]) for i in range(1, len(seq)) if seq[i] < seq[i - 1]]
print("   >>> NON-MONOTONIC backward steps:", backsteps or "none")

print("=" * 72)
print("REWRITE")
print("  status     :", result.get("status"))
print("  original ai:", detect_json.get("ai_score"))
print("  final_risk :", result.get("final_risk"))
print("  result keys:", list(result.keys()))
print("  summary keys:", list(summary.keys())[:30])

pt = summary.get("pass_trace") or result.get("pass_trace") or []
print("  pass_trace (paragraph -> mode/reason):")
for e in pt:
    if isinstance(e, dict):
        print(f"     {e.get('paragraph_id', e.get('index'))}: {e.get('mode')} reason={e.get('reason')} flags={[f.get('tag') if isinstance(f,dict) else f for f in (e.get('review_flags') or [])]}")
    else:
        print("    ", e)

orig_paras = [p for p in text.split("\n\n") if p.strip()]
final_text = result.get("final_text", "") or ""
new_paras = [p for p in final_text.split("\n\n") if p.strip()]
print("=" * 72)
print(f"PARAGRAPHS: original={len(orig_paras)}  rewritten={len(new_paras)}")
for i in range(max(len(orig_paras), len(new_paras))):
    o = orig_paras[i] if i < len(orig_paras) else "<MISSING>"
    n = new_paras[i] if i < len(new_paras) else "<MISSING>"
    same = o.strip() == n.strip()
    print(f"\n--- p{i+1} {'(UNCHANGED)' if same else '(changed)'} ---")
    print("  ORIG:", " ".join(o.split())[:220])
    if not same:
        print("  NEW :", " ".join(n.split())[:220])
