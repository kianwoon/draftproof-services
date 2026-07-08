"""FAITHFUL render-validation harness — reproduces the REAL production section
layout (page_parity=True) for FREE.

Why the earlier version was worthless: it ran with deep-scan OFF and NO
authorship-breakdown flag, so `_lead` was empty -> page_parity=False -> the
LEGACY "1. Submission and policy view" branch, which is NOT what production
renders. Production sets DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN + DEBERTA_AUTHORITATIVE
(+ paid deep-scan), giving page_parity=True ("1. Where the risk sits").

run_v7_breakdown() degrades gracefully when Modal deep-scan is absent (it fuses
on the composite score alone), so authorship_breakdown — and thus the real
page_parity=True layout — is produced with just the two FREE env flags below.
The local fakespot per-sentence classifier saturates the same way desklib does,
so this faithfully reproduces the red-on-green wall on the REAL branch.

Asserts, per document:
  - page_parity is TRUE (section "Where the risk sits" present, legacy branch absent)
  - document badge tier
  - count of RED surfaces (#dc2626 / "High-confidence AI signal" / "high_confidence_ai")
    -> MUST be 0 on a green-verdict doc, > 0 preserved on an amber/red doc
"""
import os, sys, json, glob, re

# REAL production branch, FREE: breakdown + local authoritative saturation, NO paid Modal.
os.environ["DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN"] = "1"
os.environ["DRAFTPROOF_DEBERTA_AUTHORITATIVE"] = "1"
os.environ["DRAFTPROOF_V7_DEEP_SCAN"] = "0"          # free — no Modal
os.environ.setdefault("DRAFTPROOF_V7_TIER_AUTHORITY", "1")

POC = "/Users/kianwoonwong/Downloads/draftproof_services/.claude/worktrees/ai-detection-accuracy-assessment-1238c8/poc"
sys.path.insert(0, POC)

from detect_pipeline import run_detect  # noqa: E402

GREEN_TEXT = """When I was nineteen my grandmother taught me to make her sourdough
in the cramped kitchen of her flat in Leeds. She kept the starter in a chipped blue
jam jar on the windowsill above the radiator, and she scolded me the first time I
fed it with cold tap water straight from the pipe. "It's alive," she said, "you
wouldn't pour ice down your own throat first thing." I remember the flour dust
settling on the black-and-white lino and the way the dough stuck to my knuckles
no matter how much I dusted my hands. My first three loaves came out flat as
frisbees. The fourth one rose. I carried it home on the 42 bus wrapped in a tea
towel and my flatmate ate half of it before it had properly cooled. That jar of
starter is fifteen years old now and it moved house with me four times, including
the winter the boiler broke and I had to keep it under my jumper on the walk to
the launderette so it wouldn't die in the cold."""

RED_TEXT = """In today's rapidly evolving world, sourdough bread has become an
increasingly popular choice among individuals seeking a healthier and more
sustainable lifestyle. It is important to note that the process of making
sourdough involves a variety of key factors that must be carefully considered.
Firstly, the fermentation process plays a crucial role in developing the complex
flavors and textures that characterize this beloved staple. Furthermore, many
experts agree that sourdough offers numerous potential benefits, including improved
digestibility and a lower glycemic index. Additionally, the cultivation of a
healthy starter is essential to achieving optimal results. In conclusion,
sourdough bread represents a multifaceted and rewarding endeavor that continues to
capture the interest of enthusiasts across the globe, highlighting the enduring
appeal of traditional culinary practices in a modern context."""

# The CRITICAL case: a doc that verdicts GREEN overall but contains a few
# sentences that SATURATE the per-sentence classifier (raw >=0.99). This is the
# exact production artifact (report (7).pdf): mostly-human text where a handful
# of generic/formal sentences trip the detector. On a green verdict those
# saturated sentences MUST render muted gold ("review"), never red. Mostly the
# anchored human narrative + two generic sentences appended.
GREEN_MIXED_TEXT = GREEN_TEXT + """ It is important to note that the fermentation
process plays a crucial role in developing the complex flavors and textures that
characterize this beloved staple."""

RED_HEX = "#dc2626"
RED_MARKERS = ["High-confidence AI signal", "high_confidence_ai"]


def analyze(label, text, outdir):
    os.makedirs(outdir, exist_ok=True)
    res = run_detect(text, outdir, verbose=False)
    jpath = sorted(glob.glob(os.path.join(outdir, "*.json")))[-1]
    with open(jpath) as f:
        data = json.load(f)
    badge = data.get("ai_risk_badge") or {}
    badge_tier = str(badge.get("tier") or "").lower()
    has_breakdown = bool(badge.get("authorship_breakdown"))

    mpath = sorted(glob.glob(os.path.join(outdir, "*.md")))[-1]
    with open(mpath) as f:
        md = f.read()

    # page_parity signal: production layout header vs legacy layout header
    page_parity = "Where the risk sits" in md
    legacy_layout = "Submission and policy view" in md

    red_hex_hits = md.count(RED_HEX)
    red_marker_hits = sum(md.count(m) for m in RED_MARKERS)
    # muted gold review pill (the CORRECT green-doc rendering)
    gold_hits = md.count("#c99a3b")

    print(f"\n===== {label} =====")
    print(f"  badge tier            : {badge_tier}")
    print(f"  authorship_breakdown  : {has_breakdown}")
    print(f"  page_parity (real)    : {page_parity}   legacy_layout: {legacy_layout}")
    print(f"  RED hex #dc2626 hits  : {red_hex_hits}")
    print(f"  RED marker hits       : {red_marker_hits}")
    print(f"  gold 'review' hits    : {gold_hits}")
    print(f"  md: {mpath}")
    print(f"  pdf: {sorted(glob.glob(os.path.join(outdir, '*.pdf')) or ['<none>'])[-1]}")
    return {
        "label": label, "badge_tier": badge_tier, "page_parity": page_parity,
        "legacy": legacy_layout, "red_hex": red_hex_hits, "red_marker": red_marker_hits,
        "gold": gold_hits, "md": mpath,
        "pdf": (sorted(glob.glob(os.path.join(outdir, "*.pdf"))) or [None])[-1],
    }


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/tmp/render_validate_out"
    g = analyze("GREEN (human narrative)", GREEN_TEXT, os.path.join(base, "green"))
    gm = analyze("GREEN+SATURATED (the production artifact)", GREEN_MIXED_TEXT, os.path.join(base, "green_mixed"))
    r = analyze("RED (generic AI essay)", RED_TEXT, os.path.join(base, "red"))

    print("\n===== VERDICT =====")
    parity_ok = g["page_parity"] and not g["legacy"]
    green_is_green = g["badge_tier"] in ("green", "clean")
    green_no_red = (g["red_hex"] == 0 and g["red_marker"] == 0)
    red_shows = (r["red_hex"] > 0 or r["red_marker"] > 0) if r["badge_tier"] not in ("green", "clean") else None

    print(f"page_parity=True on real branch     : {'PASS' if parity_ok else 'FAIL'}")
    print(f"GREEN doc badge is green/clean       : {'YES' if green_is_green else 'NO (tier=%s -> red is CORRECT)' % g['badge_tier']}")
    if green_is_green:
        print(f"GREEN doc shows NO red surfaces      : {'PASS' if green_no_red else 'FAIL (%d hex, %d marker)' % (g['red_hex'], g['red_marker'])}")
    if red_shows is not None:
        print(f"RED doc still shows red surfaces     : {'PASS' if red_shows else 'FAIL (red doc must highlight)'}")
    # The decisive case: green badge WITH saturated sentences -> must be gold, no red.
    print("\n----- DECISIVE: green-verdict WITH saturated sentences -----")
    gm_green = gm["badge_tier"] in ("green", "clean")
    gm_exercised = gm["gold"] > 0  # gold 'review' pills prove the high->review gate fired
    gm_no_red = gm["red_hex"] == 0 and gm["red_marker"] == 0
    print(f"  badge green/clean          : {'YES' if gm_green else 'NO (tier=%s)' % gm['badge_tier']}")
    print(f"  gate EXERCISED (gold>0)    : {'YES (%d gold)' % gm['gold'] if gm_exercised else 'NO — no saturation, gate untested'}")
    print(f"  NO red on green+saturated  : {'PASS' if (gm_green and gm_no_red) else 'FAIL (%d hex, %d marker)' % (gm['red_hex'], gm['red_marker'])}")

    print(f"\nInspect PDFs:\n  green:       {g['pdf']}\n  green_mixed: {gm['pdf']}\n  red:         {r['pdf']}")
