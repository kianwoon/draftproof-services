"""DETERMINISTIC gate test — proves the green+saturated case on the REAL render path.

Text-tweaking can't reliably hit the narrow "green verdict WITH a saturated
sentence" window (the fakespot per-sentence classifier saturates unpredictably).
So we monkeypatch the ONE scorer every consumer shares
(detect.deberta_signal.compose_from_sentences) with a faithful stand-in: it reuses
the REAL band_for_sentence() and only substitutes the raw score — 0.999 for any
sentence containing a sentinel phrase, 0.02 otherwise. Everything downstream
(builder badge/findings synthesis, report.py heatmap gate, render.py cards,
render_panels underline) is the UNMODIFIED production code.

Cases (all on page_parity=True, the real production branch):
  GREEN+SAT : 1 saturated / 11 sentences ≈ 9%  -> badge green -> saturated
              sentence MUST render muted gold ("review"), 0 red.  <-- the fix.
  CRITICAL  : 9 saturated / 11 sentences ≈ 82% -> badge red   -> red PRESERVED.
"""
import os, sys, glob, json

os.environ["DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN"] = "1"
os.environ["DRAFTPROOF_DEBERTA_AUTHORITATIVE"] = "1"
os.environ["DRAFTPROOF_V7_DEEP_SCAN"] = "0"
os.environ.setdefault("DRAFTPROOF_V7_TIER_AUTHORITY", "1")

ROOT = "/Users/kianwoonwong/Downloads/draftproof_services/.claude/worktrees/ai-detection-accuracy-assessment-1238c8"
POC = ROOT + "/poc"
sys.path.insert(0, POC)
sys.path.insert(0, ROOT)  # so `poc.xxx` deep imports resolve

import detect.deberta_signal as ds  # noqa: E402
_real_band = ds.band_for_sentence
SENTINEL = "zzsat"  # invisible marker word placed at the end of sentences we force-saturate

def _fake_compose_from_sentences(sentences):
    out = []
    for s in sentences:
        text = str(s.get("text") or "").strip()
        too_short = len(text.split()) < 8
        if too_short:
            score = None
        else:
            score = 0.999 if SENTINEL in text else 0.02
        out.append({
            "sentence_id": s.get("sentence_id"),
            "paragraph_id": s.get("paragraph_id"),
            "score": None if score is None else round(score, 3),
            "band": _real_band(score) if score is not None else "clean",
        })
    return {"sentence_scores": out, "available": True, "model_version": ds.MODEL_VERSION}

ds.compose_from_sentences = _fake_compose_from_sentences

from detect_pipeline import run_detect  # noqa: E402  (imported AFTER patch)

# 11 anchored human sentences (each >=8 words). The sentinel is appended to N of
# them to force saturation. The word "zzsat" is stripped from display by nothing —
# it will appear in text, which is fine for this internal gate test.
HUMAN = [
    "When I was nineteen my grandmother taught me to make her sourdough bread",
    "She kept the starter in a chipped blue jam jar on the windowsill",
    "She scolded me the first time I fed it with cold tap water",
    "I remember the flour dust settling on the black and white kitchen lino",
    "The dough stuck to my knuckles no matter how much I dusted my hands",
    "My first three loaves came out as flat as frisbees on the tray",
    "I carried the fourth loaf home on the number forty two bus that evening",
    "My flatmate ate half of it before it had properly cooled on the rack",
    "That jar of starter is fifteen years old now and has moved four times",
    "One winter the boiler broke and I kept it warm under my woollen jumper",
    "The fermentation process plays a crucial role in developing complex flavor here",
]

def build_text(n_saturated):
    out = []
    for i, s in enumerate(HUMAN):
        out.append(s + (" " + SENTINEL if i < n_saturated else "") + ".")
    return " ".join(out)

RED_HEX = "#dc2626"
GOLD_HEX = "#c99a3b"
RED_MARKERS = ["High-confidence AI signal", "high_confidence_ai"]

def analyze(label, text, outdir):
    os.makedirs(outdir, exist_ok=True)
    run_detect(text, outdir, verbose=False)
    j = sorted(glob.glob(os.path.join(outdir, "*.json")))[-1]
    d = json.load(open(j))
    badge = (d.get("ai_risk_badge") or {})
    tier = str(badge.get("tier") or "").lower()
    md = open(sorted(glob.glob(os.path.join(outdir, "*.md")))[-1]).read()
    red = md.count(RED_HEX) + sum(md.count(m) for m in RED_MARKERS)
    gold = md.count(GOLD_HEX) + md.count("Possible AI — review")
    parity = "Where the risk sits" in md and "Submission and policy view" not in md
    pdf = (sorted(glob.glob(os.path.join(outdir, "*.pdf"))) or [None])[-1]
    print(f"\n== {label} ==  badge={tier}  page_parity={parity}  RED={red}  GOLD={gold}")
    print(f"   pdf: {pdf}")
    return {"tier": tier, "red": red, "gold": gold, "parity": parity, "pdf": pdf}


if __name__ == "__main__":
    base = "/tmp/gate_determ"
    gs = analyze("GREEN+SATURATED (1/11 ~ 9%)", build_text(1), os.path.join(base, "green_sat"))
    cr = analyze("CRITICAL (9/11 ~ 82%)", build_text(9), os.path.join(base, "critical"))

    print("\n===== VERDICT =====")
    p1 = gs["tier"] in ("green", "clean") and gs["parity"] and gs["red"] == 0 and gs["gold"] > 0
    p2 = cr["tier"] in ("amber", "orange", "red") and cr["parity"] and cr["red"] > 0
    print(f"GREEN+SAT: green badge, gate FIRED (gold>0), ZERO red : {'PASS' if p1 else 'FAIL'}  "
          f"(tier={gs['tier']} red={gs['red']} gold={gs['gold']})")
    print(f"CRITICAL : red badge, red PRESERVED                   : {'PASS' if p2 else 'FAIL'}  "
          f"(tier={cr['tier']} red={cr['red']})")
    print(f"\nInspect:\n  green_sat: {gs['pdf']}\n  critical : {cr['pdf']}")
