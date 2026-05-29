"""End-to-end test of the lean direct-rewrite path: original vs rewritten AI score, the component
breakdown (did the content-lacking signals drop?), review flags, and the rewritten text itself."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

root = "/Users/kianwoonwong/Downloads/draftproof_services"
load_dotenv(root + "/.env")
sys.path.insert(0, root)
os.environ["DRAFTPROOF_V6_DIRECT_REWRITE"] = "1"

from poc.rewrite_v6.production import run_rewrite_pipeline_v6

REPORT = (
    root + "/test_output/production_rewrite_ad62c7f1_20260529/"
    "reports__7f9eada9-e81a-4e4c-be2b-0308c7bc8b61__report.json"
)
OUT = root + "/test_output/_test_lean_e2e"
KEYS = ["predictability", "generic_assertion_risk", "qualifying_text_ai_density",
        "repeated_sentence_structure_risk", "burstiness_risk"]


def comps(report):
    b = (report or {}).get("ai_risk_badge", {}) or {}
    return b.get("ai_components", {}) or {}


def main() -> int:
    detect_json = json.loads(Path(REPORT).read_text())
    env = run_rewrite_pipeline_v6(detect_json=detect_json, output_dir=OUT)
    result = json.loads(Path(env["json_path"]).read_text())

    o, f = result.get("original_risk"), result.get("final_risk")
    pt = result.get("candidate_generation_status", {}).get("pass_trace", [])
    srcs = Counter(p.get("selected_source") for p in pt)
    flags = [fl for p in pt for fl in (p.get("author_review_items") or [])]
    oc, fc = comps(result.get("detect_scan_original")), comps(result.get("detect_scan_rewritten"))

    orig_words = len((detect_json.get("submitted_text") or "").split()) or sum(
        len(p.get("paragraph_excerpt", "").split()) for p in detect_json.get("rewrite_edit_briefs", [])
        if p.get("paragraph_id") in {b.get("paragraph_id") for b in detect_json.get("rewrite_edit_briefs", [])}
    )
    # simpler: reconstruct original words from unique paragraph excerpts
    seen = {}
    for b in detect_json.get("rewrite_edit_briefs", []):
        pid, exc = b.get("paragraph_id"), b.get("paragraph_excerpt")
        if pid and exc and pid not in seen:
            seen[pid] = exc
    orig_words = len(" ".join(seen.values()).split())
    new_words = len((result.get("final_text") or "").split())

    print("\n================= LEAN DIRECT-REWRITE E2E =================")
    print(f"AI likelihood:  {o}  ->  {f}   (Δ {round((f - o), 1)})")
    print(f"word count:     {orig_words}  ->  {new_words}   ({new_words/max(1,orig_words):.2f}x)")
    print(f"paragraphs rewritten: {srcs.get('direct_llm', 0)}   fallback(no solution): {srcs.get('source_preserved', 0)}")
    print(f"\ncontent-signal components (original -> rewritten):")
    for k in KEYS:
        if isinstance(oc.get(k), (int, float)) and isinstance(fc.get(k), (int, float)):
            print(f"  {k:<34} {oc[k]:>6} -> {fc[k]:<6} ({fc[k]-oc[k]:+.1f})")
    print(f"\nreview flags for the user ({len(flags)}):")
    for fl in flags[:8]:
        print(f"  - {fl.get('added')}: {fl.get('why')}")

    print("\n----- REWRITTEN TEXT -----")
    print(result.get("final_text", "")[:2400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
