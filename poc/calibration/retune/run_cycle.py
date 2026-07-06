"""Phase 1 -> Phase 2 orchestrator. Builds the manifest, runs the FPR-gate oracle, and
appends a RETUNE_LOG.md decision row. Produces CANDIDATE artifacts only — promotion to
production is a separate, human-approved step (see docs/runbooks/v7-retune.md)."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from .intake import write_manifest_only, load_env, generate_ai_essays, TOPICS, DEFAULT_OUT, DEFAULT_SCOCESLE, DEFAULT_MANIFEST
from .generators import load_generators
from .gate import run_fpr_gate, GateResult

HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "RETUNE_LOG.md"
_HEADER = "| version | n_rows | families | gate | auc_line |\n|---|---|---|---|---|\n"

def append_log(log_path: Path, entry: dict) -> None:
    if not log_path.exists():
        log_path.write_text("# V7 Re-Tune Decision Log\n\n" + _HEADER)
    with log_path.open("a") as f:
        f.write(f"| {entry['version']} | {entry['n_rows']} | {entry['families']} "
                f"| {entry['gate']} | {entry['auc_line']} |\n")

def _families(manifest_rows) -> str:
    return ",".join(sorted({r["family"] for r in manifest_rows if r["label"] == "ai"}))

def run_cycle(ai_dir: Path, scocesle_dir: Path | None, manifest_path: Path, log_path: Path,
              now_iso: str, generate: bool, gate_fn=run_fpr_gate, generate_fn=None) -> GateResult:
    if generate:
        load_env()
        (generate_fn or generate_ai_essays)(ai_dir, load_generators(), TOPICS)
    n_rows = write_manifest_only(ai_dir, scocesle_dir, manifest_path, now_iso)
    rows = json.loads(manifest_path.read_text())["rows"]
    result = gate_fn(corpus=scocesle_dir)
    verdict = "PASS" if result.passed else ("NO-CORPUS" if not result.corpus_available else "FAIL")
    auc_line = next((ln.strip() for ln in result.stdout.splitlines() if "AUC" in ln), "")
    append_log(log_path, {"version": now_iso, "n_rows": n_rows, "families": _families(rows),
                          "gate": verdict, "auc_line": auc_line})
    print(f"\n=== V7 RE-TUNE: {verdict} ===")
    return result

def main() -> int:
    ap = argparse.ArgumentParser(description="V7 re-tune cycle: Phase 1 intake -> Phase 2 gate")
    ap.add_argument("--generate", action="store_true", help="generate fresh AI essays first")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scocesle", type=Path, default=DEFAULT_SCOCESLE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args()
    now_iso = datetime.now(timezone.utc).isoformat()
    scocesle = args.scocesle if args.scocesle.exists() else None
    res = run_cycle(args.out, scocesle, args.manifest, args.log, now_iso, args.generate)
    return 0 if (res.passed or not res.corpus_available) else 1

if __name__ == "__main__":
    raise SystemExit(main())
