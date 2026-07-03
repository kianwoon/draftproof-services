"""Fused-score ESL gate run — the LAST precondition for the V7 authority re-base.

Question this answers with a number: if the authoritative Risk Tier were driven by the
V7 fused score (fusion_weights.deep_scan_2detector: 0.40 x composite ai_likelihood +
0.60 x deep-scan proportion x 100) instead of the composite alone, would the ESL
false-positive gate still pass?

Mirrors poc/calibration/fpr_subgroup_gate.py exactly: same corpus loader, same
FPR_THRESHOLDS / PRIMARY_THRESHOLD, same tolerances (MAX_FPR_RISE_PTS / MAX_AUC_DROP /
MAX_PARITY_WIDEN_PTS) — applied fused-vs-composite measured in the SAME run, so the
comparison is apples-to-apples on identical texts.

Inputs:
- 272 SCoCESLE human essays (DEFAULT_CORPUS): composite via local DetectionRunner
  (free, ~4s/essay), deep-scan proportion via the live Modal endpoint (~$0.31).
- AI set = authorship_cases (65, original family) + authorship_cases_v2 (59, three
  families from the diversity check). Proportions already paid for in
  v7_deberta_diversity_v2.json are REUSED, not re-scored.

Progress is cached incrementally to v7_fused_gate_progress.jsonl — reruns skip
completed rows, so a crash or interrupt never loses paid Modal work.

Usage (from repo root):  python poc/calibration/v7_fused_gate_run.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from calibration.fpr_subgroup_gate import (  # noqa: E402
    FPR_THRESHOLDS, PRIMARY_THRESHOLD,
    MAX_FPR_RISE_PTS, MAX_AUC_DROP, MAX_PARITY_WIDEN_PTS,
    DEFAULT_CORPUS, AI_CASES, _proficiency_groups, _auc,
)
from calibration.measure_end_to_end import scan_text  # noqa: E402
from calibration.v7_deberta_diversity_check import (  # noqa: E402
    _load_env, _score_essay, CASES_V2,
)
from detect.run import DetectionRunner  # noqa: E402

HERE = Path(__file__).resolve().parent
PROGRESS = HERE / "v7_fused_gate_progress.jsonl"
OUT_PATH = HERE / "v7_fused_gate_result.json"
DIVERSITY = HERE / "v7_deberta_diversity_v2.json"


def _weights_cfg() -> tuple[float, float, float]:
    w = json.loads((_POC / "detect_v7" / "weights.json").read_text())
    fusion = w["fusion_weights"]["deep_scan_2detector"]
    sent_thr = w["deep_scan_calibration"]["sent_threshold"]
    return float(fusion["fakespot"]), float(fusion["deberta_large"]), float(sent_thr)


def _load_progress() -> dict[str, dict]:
    done: dict[str, dict] = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    return done


def _append_progress(row: dict) -> None:
    with PROGRESS.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _prepaid_proportions() -> dict[str, float]:
    """case_id -> proportion already scored (and paid for) by the diversity run."""
    if not DIVERSITY.exists():
        return {}
    d = json.loads(DIVERSITY.read_text())
    out: dict[str, float] = {}
    for fam in d.get("families", {}).values():
        out.update(fam.get("per_essay", {}))
    out.update(d.get("consistency_check_original_family", {}).get("per_essay", {}))
    return out


def _fpr(vals: list[float], thr: float) -> float | None:
    return round(100.0 * sum(1 for v in vals if v >= thr) / len(vals), 1) if vals else None


def main() -> None:
    env = _load_env()
    url, token = env.get("DRAFTPROOF_MODAL_ENDPOINT_URL"), env.get("DRAFTPROOF_MODAL_ENDPOINT_TOKEN")
    if not url or not token:
        print("Modal endpoint env missing", file=sys.stderr)
        raise SystemExit(2)
    fw_fakespot, fw_deberta, sent_thr = _weights_cfg()
    print(f"fusion weights: fakespot={fw_fakespot} deberta={fw_deberta}; sent_threshold={sent_thr}", flush=True)

    groups = _proficiency_groups(DEFAULT_CORPUS)
    ai_files = sorted(glob.glob(str(AI_CASES / "ai_*.json"))) + sorted(glob.glob(str(CASES_V2 / "ai_*.json")))
    prepaid = _prepaid_proportions()
    done = _load_progress()
    runner = DetectionRunner()
    t0 = time.time()

    work: list[tuple[str, str, str]] = []  # (id, kind, text)
    for key in ("higher", "lower"):
        for p in groups[key]:
            # encoding/errors mirror fpr_subgroup_gate.py EXACTLY — some SCoCESLE files
            # carry Windows-1252 bytes (0x93 curly quotes) and identical preprocessing
            # is required for the composite-vs-fused comparison to be apples-to-apples.
            work.append((f"human_{key}:{Path(p).name}", f"human_{key}",
                         Path(p).read_text(encoding="utf-8", errors="ignore")))
    for p in ai_files:
        d = json.loads(Path(p).read_text())
        if (d.get("authorship") or "").lower() == "ai" and d.get("text"):
            work.append((f"ai:{d['case_id']}", "ai", d["text"]))

    print(f"total texts: {len(work)} (done already: {len(done)})", flush=True)
    for i, (rid, kind, text) in enumerate(work):
        if rid in done:
            continue
        composite = None
        try:
            rep = scan_text(runner, text)
            badge = rep.get("ai_risk_badge") or {}
            composite = float(badge.get("ai_likelihood_score") or 0.0)
        except Exception as e:  # noqa: BLE001 — record and continue, don't lose the run
            print(f"  [{rid}] composite failed: {e}", flush=True)
            continue
        case_id = rid.split(":", 1)[1].replace(".json", "") if kind == "ai" else None
        if case_id and case_id in prepaid:
            proportion = float(prepaid[case_id])
        else:
            try:
                scores, _n = _score_essay(url, token, text)
                proportion = (sum(1 for x in scores if x >= sent_thr) / len(scores)) if scores else None
            except Exception as e:  # noqa: BLE001
                print(f"  [{rid}] modal failed: {e}", flush=True)
                continue
        if proportion is None:
            continue
        row = {"id": rid, "kind": kind, "composite": round(composite, 2), "proportion": round(proportion, 4)}
        _append_progress(row)
        done[rid] = row
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  progress {len(done)}/{len(work)} ({elapsed:.0f}s)", flush=True)

    # ── Metrics ────────────────────────────────────────────────────────────
    def fused(row: dict) -> float:
        return fw_fakespot * row["composite"] + fw_deberta * (row["proportion"] * 100.0)

    by_kind: dict[str, list[dict]] = {}
    for row in done.values():
        by_kind.setdefault(row["kind"], []).append(row)
    humans = by_kind.get("human_higher", []) + by_kind.get("human_lower", [])
    ais = by_kind.get("ai", [])

    result: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "n": {k: len(v) for k, v in by_kind.items()},
        "fusion_weights": {"fakespot": fw_fakespot, "deberta_large": fw_deberta},
        "sent_threshold": sent_thr,
        "scores": {}, "verdict": {},
    }
    for label, score_fn in (("composite", lambda r: r["composite"]), ("fused", fused)):
        hi = [score_fn(r) for r in by_kind.get("human_higher", [])]
        lo = [score_fn(r) for r in by_kind.get("human_lower", [])]
        ai_vals = [score_fn(r) for r in ais]
        entry = {
            "fpr": {str(t): {"higher": _fpr(hi, t), "lower": _fpr(lo, t),
                             "overall": _fpr(hi + lo, t)} for t in FPR_THRESHOLDS},
            "parity_gap_at_primary": round((_fpr(lo, PRIMARY_THRESHOLD) or 0) - (_fpr(hi, PRIMARY_THRESHOLD) or 0), 1),
            "auc_ai_vs_human": _auc(ai_vals, hi + lo),
            "ai_tpr_at_primary": _fpr(ai_vals, PRIMARY_THRESHOLD),  # same formula, applied to AI = TPR
        }
        result["scores"][label] = entry

    comp, fus = result["scores"]["composite"], result["scores"]["fused"]
    fpr_rise = (fus["fpr"][str(PRIMARY_THRESHOLD)]["overall"] or 0) - (comp["fpr"][str(PRIMARY_THRESHOLD)]["overall"] or 0)
    auc_drop = (comp["auc_ai_vs_human"] or 0) - (fus["auc_ai_vs_human"] or 0)
    parity_widen = abs(fus["parity_gap_at_primary"]) - abs(comp["parity_gap_at_primary"])
    result["verdict"] = {
        "fpr_rise_pts": round(fpr_rise, 1), "fpr_rise_ok": fpr_rise <= MAX_FPR_RISE_PTS,
        "auc_drop": round(auc_drop, 4), "auc_drop_ok": auc_drop <= MAX_AUC_DROP,
        "parity_widen_pts": round(parity_widen, 1), "parity_widen_ok": parity_widen <= MAX_PARITY_WIDEN_PTS,
    }
    result["verdict"]["gate_pass"] = all(
        result["verdict"][k] for k in ("fpr_rise_ok", "auc_drop_ok", "parity_widen_ok"))
    OUT_PATH.write_text(json.dumps(result, indent=1))
    print(f"\nsaved {OUT_PATH}", flush=True)
    print(json.dumps(result["scores"], indent=1), flush=True)
    print("VERDICT:", json.dumps(result["verdict"]), flush=True)


if __name__ == "__main__":
    main()
