"""M4 evaluation harness — run the M2/M3 claim graph over the mini-corpus.

Measurement, not a test (plan §7 deterministic-measurement convention). For each
manifest doc: canonical segments -> M2 extraction (real gateway, in-process cache
by text hash) -> validated graph -> M3 signals -> a row with the interrogatability
band+components, the OFFLINE verified-weighted A/B variant (interrog_ab), the
substitutability value, the origin map, ``generic_assertion_risk`` (cheap
deterministic, for the §4b correlation), and extraction stats. Output is
resume-safe JSONL keyed by ``doc_id`` — a re-run skips completed docs.

BUDGET CAP (plan §M4): <=60 docs; batching/retries tuned so each doc stays within
~5 gateway calls; a global abort trips politely past ``GLOBAL_CALL_ABORT`` calls.
Never runs on the free path; kill-switch ``DRAFTPROOF_CLAIM_GRAPH`` forced on here.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_POC_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_POC_ROOT, ".."))
# The `detect` package imports `poc.predictability...`, so `poc` must resolve as
# a top-level package — put the repo/worktree root (parent of poc) on the path.
for _p in (_REPO_ROOT, _POC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAX_DOCS = 60
PER_DOC_CALL_SOFT_CAP = 5
GLOBAL_CALL_ABORT = 350
_DEFAULT_OUT = os.path.join(_HERE, "eval_rows.jsonl")
_MANIFEST = os.path.join(_HERE, "corpus_manifest.json")


def _tune_budget_env() -> None:
    """Force the kill-switch on and bias extraction toward <=5 calls/doc.

    These essays are short (1-5 paragraphs); a large paragraph batch collapses the
    doc into ~1 extraction call, +1 reconcile, +<=1 retry.
    """
    os.environ["DRAFTPROOF_CLAIM_GRAPH"] = "1"
    os.environ.setdefault("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "8")
    os.environ.setdefault("DRAFTPROOF_CLAIM_GRAPH_MAX_SENTENCES_PER_BATCH", "24")
    os.environ.setdefault("DRAFTPROOF_CLAIM_GRAPH_MAX_RETRIES", "1")


def _load_env_file() -> None:
    """Load repo-root .env (Cerebras/OpenRouter keys) if not already in env."""
    env_path = os.path.abspath(os.path.join(_POC_ROOT, "..", ".env"))
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _signal(container: dict[str, Any], name: str) -> dict[str, Any]:
    for s in container.get("signals") or []:
        if s.get("signal") == name:
            return s
    return {}


def _completed_ids(out_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["doc_id"])
            except Exception:
                continue
    return done


def _entailment_summary(container: dict[str, Any], claims: list) -> dict[str, Any]:
    """N6: capture the N2/N3/N4 verdict data for the §B calibration report.

    Reads the FLAG-ON container's ``evidence`` (N2 resolution + N3 attached
    entailment verdicts) and the N4-promoted ``verification_status`` on the real
    CLAIM nodes. Empty/quiet when entailment was OFF (no ``evidence``)."""
    real = [c for c in claims if c.get("node_type") == "CLAIM" and not c.get("references")]
    vcounts: dict[str, int] = {}
    for c in real:
        st = c.get("verification_status") or "unverified"
        vcounts[st] = vcounts.get(st, 0) + 1

    res_statuses: list[str] = []
    ent_scores: list[float] = []
    con_scores: list[float] = []
    verdict_statuses: list[str] = []
    for node in container.get("evidence") or []:
        detail = node.get("detail") if isinstance(node, dict) else None
        resolution = detail.get("resolution") if isinstance(detail, dict) else None
        if not isinstance(resolution, dict):
            continue
        res_statuses.append(str(resolution.get("status")))
        verdicts = resolution.get("entailment")
        if isinstance(verdicts, dict):
            for v in verdicts.values():
                if not isinstance(v, dict):
                    continue
                verdict_statuses.append(str(v.get("status")))
                ent_scores.append(float(v.get("entailment_score") or 0.0))
                con_scores.append(float(v.get("contradiction_score") or 0.0))
    return {
        "verification_counts": vcounts,
        "n_verified": vcounts.get("verified", 0),
        "n_contradicted": vcounts.get("contradicted", 0),
        "verified_fired": vcounts.get("verified", 0) > 0,
        "contradicted_fired": vcounts.get("contradicted", 0) > 0,
        "resolution_statuses": res_statuses,
        "verdict_statuses": verdict_statuses,
        "max_entailment_score": round(max(ent_scores), 4) if ent_scores else None,
        "max_contradiction_score": round(max(con_scores), 4) if con_scores else None,
        "entailment_scores": [round(x, 4) for x in ent_scores],
        "contradiction_scores": [round(x, 4) for x in con_scores],
    }


def build_row(doc: dict[str, Any], text: str, gateway: Any) -> dict[str, Any]:
    from poc.detect.document_structure import structured_sentence_segments
    from poc.detect.layer3_scoring import estimate_generic_assertion_risk
    from poc.claim_graph.build import build_claim_graph_extracted
    from poc.claim_graph.eval import interrog_ab

    segments = structured_sentence_segments(text)
    container = build_claim_graph_extracted(text, segments, gateway)
    claims = container.get("claims") or []
    edges = container.get("edges") or []

    interro = _signal(container, "interrogatability")
    subst = _signal(container, "substitutability")
    origin = _signal(container, "origin_map")
    stats = container.get("extraction_stats") or {}
    llm_calls = int(stats.get("extract_calls", 0)) + int(stats.get("reconcile_calls", 0))

    real_claims = [c for c in claims if c.get("node_type") == "CLAIM" and not c.get("references")]
    return {
        "doc_id": doc["doc_id"], "group": doc["group"], "source": doc.get("source"),
        "case_type": doc.get("case_type"),  # N6 real-citation slice: entailed|misattributed
        "words": doc.get("words"),
        # N6: N2/N3/N4 verdict capture (entailment ON) for the §B report.
        "entailment": _entailment_summary(container, claims),
        "interrogatability": {
            "composite": (interro.get("value") or {}).get("composite"),
            "band": interro.get("band"),
            "components": interro.get("components"),
            "unverified_specific_rate": (interro.get("value") or {}).get("unverified_specific_rate"),
            "questions_emitted": (interro.get("value") or {}).get("questions_emitted"),
        },
        # OFFLINE A/B — same graph, verification-weighted specificity (report Q c).
        "interrog_verified": interrog_ab.recompute(claims, edges, variant="verified"),
        "interrog_current_recompute": interrog_ab.recompute(claims, edges, variant="current"),
        "substitutability": {"value": subst.get("value"), "band": subst.get("band")},
        "generic_assertion_risk": round(estimate_generic_assertion_risk(text), 4),
        "origin_map": origin.get("value"),
        "graph": {
            "claims_real": len(real_claims),
            "questions": sum(1 for c in claims if c.get("node_type") == "QUESTION"),
            "edges": len(edges),
            "truncated": container.get("truncated"),
        },
        "extraction_stats": {k: stats.get(k) for k in (
            "batches", "extract_calls", "retries", "reconcile_calls",
            "parse_failures", "proposed", "accepted", "rejected")},
        "llm_calls": llm_calls,
        "lifecycle": container.get("lifecycle"),
        "error": None,
    }


def run(manifest_path: str = _MANIFEST, out_path: str = _DEFAULT_OUT,
        limit: Optional[int] = None, groups: Optional[set] = None,
        max_docs: int = MAX_DOCS) -> dict[str, Any]:
    """Run the M2/M3(/N2-N4) claim graph over a manifest → resume-safe JSONL rows.

    ``groups`` optionally restricts to a subset (e.g. ``{"G"}`` for the gaming-only
    N6 re-run). ``max_docs`` caps the manifest size guard. Entailment ON/OFF is
    governed by ``DRAFTPROOF_ENTAILMENT`` in the environment (N6 runs it ON)."""
    _load_env_file()
    _tune_budget_env()
    from poc.claim_graph.extract import resolve_gateway
    from poc.claim_graph.eval.build_corpus_manifest import load_persuade_cache, resolve_text

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    docs = manifest["docs"]
    if groups is not None:
        docs = [d for d in docs if d.get("group") in groups]
    if len(docs) > max_docs:
        raise RuntimeError("manifest has %d docs > max_docs %d" % (len(docs), max_docs))

    gateway = resolve_gateway()
    if gateway is None:
        raise RuntimeError("no gateway (set CEREBRAS_API_KEY / OPENROUTER_API_KEY)")
    persuade_cache = load_persuade_cache()

    done = _completed_ids(out_path)
    total_calls = 0
    processed = 0
    started = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        for doc in docs:
            if limit is not None and processed >= limit:
                break
            if doc["doc_id"] in done:
                continue
            if total_calls >= GLOBAL_CALL_ABORT:
                print("[ABORT] global call budget %d reached; stopping politely" % GLOBAL_CALL_ABORT)
                break
            try:
                text = resolve_text(doc, persuade_cache)
                row = build_row(doc, text, gateway)
            except Exception as exc:
                row = {"doc_id": doc["doc_id"], "group": doc["group"],
                       "error": str(exc)[:300], "llm_calls": 0}
            calls = int(row.get("llm_calls") or 0)
            total_calls += calls
            if calls > PER_DOC_CALL_SOFT_CAP:
                row["over_soft_cap"] = True
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            processed += 1
            print("[%s] %s calls=%d cum=%d" % (
                row.get("group"), row["doc_id"], calls, total_calls))

    elapsed = round(time.time() - started, 1)
    summary = {"processed": processed, "total_llm_calls": total_calls,
               "wall_seconds": elapsed, "out": out_path}
    print("SUMMARY %s" % json.dumps(summary))
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Claim-graph eval harness (M4/N6).")
    ap.add_argument("--manifest", default=_MANIFEST)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--groups", default=None,
                    help="comma-separated group filter, e.g. 'G' or 'R'")
    ap.add_argument("--max-docs", type=int, default=MAX_DOCS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    grp = set(g.strip() for g in args.groups.split(",")) if args.groups else None
    run(manifest_path=args.manifest, out_path=args.out, limit=args.limit,
        groups=grp, max_docs=args.max_docs)
