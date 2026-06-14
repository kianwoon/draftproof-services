#!/usr/bin/env python3
"""Attribute the external-detector estimate to its component signals.

Answers: "which group / signal is driving (over-rating) the external % on THIS doc?"

It does NOT recompute anything — it decomposes the numbers the scan already
recorded in the report at: ai_risk_badge.external_detector_estimate
(see poc/detect/external_grouped_scoring.py for the source math).

Usage:
    python _attribute_external.py path/to/report.json
    python _attribute_external.py path/to/external_block.json   # just the estimate block
    cat external_block.json | python _attribute_external.py -

The decomposition mirrors external_grouped_scoring.py exactly:
    final = 0.35*P + 0.30*W + 0.20*D + 0.15*Ggap        (group weights)
    risk group score   = sum(value*w)/sum(w)  over AVAILABLE signals
    grounding_strength = sum(value*w)/sum(w)  over AVAILABLE grounding signals
    Ggap               = 100 - grounding_strength

Per-signal "points contributed to the final score":
    risk signal s in group g:  W_g * (value_s * w_s) / sum(w over avail in g)
    grounding deficit s:       W_gap * ((100 - value_s) * w_s) / sum(w over avail)
The grounding rows show points the MISSING grounding adds to the gap.
"""
import json
import sys


def _load(arg: str) -> dict:
    raw = sys.stdin.read() if arg == "-" else open(arg, encoding="utf-8").read()
    data = json.loads(raw)
    # Accept: full report, the ai_risk_badge, or the estimate block itself.
    for path in (
        ("ai_risk_badge", "external_detector_estimate"),
        ("external_detector_estimate",),
    ):
        node = data
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and "signals" in node:
            return node
    if "signals" in data:
        return data
    raise SystemExit("Could not find external_detector_estimate block in input.")


def attribute(est: dict) -> None:
    weights = est.get("weights", {})
    W = {
        "probability_shape_risk": weights.get("probability_shape_risk", 0.35),
        "detector_agreement_risk": weights.get("detector_agreement_risk", 0.20),
        "writing_pattern_risk": weights.get("writing_pattern_risk", 0.30),
        "grounding_gap_risk": weights.get("grounding_gap_risk", 0.15),
    }
    signals = est.get("signals", [])

    # Group available signals exactly as _group_score does.
    risk_groups = ("probability_shape_risk", "detector_agreement_risk", "writing_pattern_risk")
    rows = []  # (points_to_final, group, key, value, weight, note)

    for g in risk_groups:
        members = [s for s in signals if s.get("group") == g]
        avail = [s for s in members if s.get("available") and s.get("value") is not None]
        wsum = sum(s["weight"] for s in avail) or 1.0
        for s in avail:
            contrib = W[g] * (s["value"] * s["weight"]) / wsum
            rows.append((contrib, g, s["key"], s["value"], s["weight"], ""))
        for s in members:
            if s not in avail:
                rows.append((0.0, g, s["key"], None, s.get("weight"), s.get("note", "unavailable")))

    # Grounding: strength reduces the gap; attribute the GAP to missing grounding.
    gmembers = [s for s in signals if s.get("group") == "grounding_strength"]
    gavail = [s for s in gmembers if s.get("available") and s.get("value") is not None]
    gwsum = sum(s["weight"] for s in gavail) or 1.0
    Wgap = W["grounding_gap_risk"]
    for s in gavail:
        deficit = (100.0 - s["value"]) * s["weight"] / gwsum
        contrib = Wgap * deficit
        rows.append((contrib, "grounding_gap_risk", s["key"] + " (missing)", s["value"], s["weight"], "gap driver"))

    rows.sort(key=lambda r: r[0], reverse=True)

    groups = est.get("groups", {})
    print(f"\nFINAL external estimate: {est.get('score')}%  (band={est.get('band')}, confidence={est.get('confidence')})")
    print("\nGROUP CONTRIBUTIONS (points of the final score):")
    for g in risk_groups:
        gs = groups.get(g)
        if gs is not None:
            print(f"  {g:26s} score={gs:6.2f}  x{W[g]:.2f} = {W[g]*gs:6.2f} pts")
    gap = groups.get("grounding_gap_risk")
    if gap is not None:
        print(f"  {'grounding_gap_risk':26s} score={gap:6.2f}  x{Wgap:.2f} = {Wgap*gap:6.2f} pts"
              f"   (grounding_strength={groups.get('grounding_strength')})")

    print("\nSIGNAL CONTRIBUTIONS (ranked by points added to the final score):")
    print(f"  {'pts':>6}  {'group':24s}  {'signal':34s}  {'value':>6}  {'wt':>4}  note")
    for contrib, g, key, val, wt, note in rows:
        vstr = f"{val:6.1f}" if isinstance(val, (int, float)) else "   n/a"
        wstr = f"{wt:4.1f}" if isinstance(wt, (int, float)) else "  - "
        print(f"  {contrib:6.2f}  {g:24s}  {key:34s}  {vstr}  {wstr}  {note}")

    caps = est.get("caps") or []
    if caps:
        print("\nCAPS APPLIED:")
        for c in caps:
            print(f"  {c.get('code')}: {c.get('before')} -> {c.get('after')}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    attribute(_load(sys.argv[1]))
