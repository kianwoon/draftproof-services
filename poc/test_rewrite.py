"""Test rewrite pipeline — force all 3 passes, full report."""

import sys
import os
import time

# Set up import paths before any project imports
_POC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _POC_DIR)
sys.path.insert(0, os.path.join(_POC_DIR, "rewriter"))
sys.path.insert(0, os.path.join(_POC_DIR, "predictability"))
sys.path.insert(0, os.path.join(_POC_DIR, "report"))

import rewriter as rw_module
from scanner import PredictabilityScanner
from pipeline import run_pipeline
from report.report import ReportBuilder
from report.render import render_markdown

multi_pass_rewrite = rw_module.multi_pass_rewrite

TEXT = """Hairdressing sits at the intersection of chemistry, geometry, and gut instinct — a trade where a 2 mm trimming error can turn a client into an ex-client. From the curled wigs of Versailles courtiers to the fade cuts of 1990s Brooklyn barbershops, hair has carried meaning far beyond aesthetics. Today, a stylist who bleaches a client's hair to platinum on Monday must still ensure it doesn't snap off by Thursday — equal parts chemist and sculptor.

The craft rests on a paradox. Hold a section at 90 degrees with too much tension and the graduation disappears; hold it at 45 with too little and the client gets a shelf instead of layers. I once watched a junior stylist learn this the hard way on a Friday afternoon — the client was her landlord. A bleach bath left on two minutes too long at pH 11 will lift pigment and melt the disulfide bonds in the same stroke. Whether performing a complex color correction or a chemical straightening treatment, the professional must possess a deep understanding of how different products interact with the hair's keratin structure. Miss that balance and the hair goes gummy — technically achieved but structurally wrecked.

Strip away the foils and blow-dryers and you are left with two strangers in a mirror, one of them holding scissors. A client will say just a trim and mean make me look like I did at 22. Perhaps the hardest skill in the trade is hearing the gap between those two sentences. Fine, colour-treated hair won't hold a bleached pixie cut no matter how many reference photos arrive on Instagram — and saying so without losing the booking is its own art. Few other professionals stand close enough to notice a client's hands shaking after a breakup — and the chair doesn't allow small talk to stay small for long. A stylist who remembers that a client's mother preferred a side part — and why — keeps that chair booked for years.

The trade reinvents itself roughly every decade. A bob that takes off on TikTok on Monday can fill appointment books by Friday. To remain relevant, a professional must commit to lifelong learning. This involves staying updated on the latest tools, such as ergonomic shears or high-tech heat styling equipment, as well as mastering new techniques like balayage, foliage, or precision barbering. Sustainability has moved from marketing copy to licensing requirements in parts of the EU. Many modern professionals are moving toward green salon practices, utilizing biodegradable products and reducing water waste to minimize their environmental footprint.

The real metric? Whether the client sits straighter in the car on the way home. A colour correction after a botched box dye job has been known to change which meetings someone volunteers to speak at. It is one of the few trades where steel meets keratin meets psychology — and has done since barbers also pulled teeth. A fringe trimmed 5 mm too short or a copper toner left on 30 seconds too long can alter how a stranger reads your face"""


# ── Per-pass rewrite functions ────────────────────────────────────────

def rewrite_pass_1(text, span_info):
    """Pass 1: Highest-flagged sentences (top-10 > 65%)."""
    return (text
        .replace(
            "Strip away the foils and blow-dryers and you are left with two strangers in a mirror, one of them holding scissors.",
            "Peel back the foils and the chair becomes what it always was: two people who barely know each other, one of them holding scissors.")
        .replace(
            "Many modern professionals are moving toward green salon practices, utilizing biodegradable products and reducing water waste to minimize their environmental footprint.",
            "Salons from Utrecht to Ljubljana are swapping ammonia-based colour for plant-derived lines, and at least three EU member states now tie water-recycling to licensing.")
        .replace(
            "Whether the client sits straighter in the car on the way home.",
            "Does the client sit a little taller behind the wheel on the drive back?")
    )


def rewrite_pass_2(text, span_info):
    """Pass 2: Mid-tier flagged (top-10 55-65%)."""
    return (text
        .replace(
            "Hairdressing sits at the intersection of chemistry, geometry, and gut instinct — a trade where a 2 mm trimming error can turn a client into an ex-client.",
            "Hairdressing trades in chemistry, geometry, and nerve — misjudge by two millimetres and the client walks.")
        .replace(
            "To remain relevant, a professional must commit to lifelong learning.",
            "A stylist who stops learning after qualifying might as well hand back the licence.")
        .replace(
            "This involves staying updated on the latest tools, such as ergonomic shears or high-tech heat styling equipment, as well as mastering new techniques like balayage, foliage, or precision barbering.",
            "That means understanding why a balayage pattern behaves differently on coarse versus fine hair, not just buying the newest ergonomic shears on the market.")
        .replace(
            "a stylist who bleaches a client's hair to platinum on Monday must still ensure it doesn't snap off by Thursday — equal parts chemist and sculptor.",
            "a stylist who lifts a client's hair to platinum on Monday still has to guarantee it flexes on Thursday — half bench chemist, half sculptor.")
    )


def rewrite_pass_3(text, span_info):
    """Pass 3: Final polish on remaining medium-flagged."""
    return (text
        .replace(
            "Whether performing a complex color correction or a chemical straightening treatment, the professional must possess a deep understanding of how different products interact with the hair's keratin structure.",
            "Running a complex colour correction or a chemical relaxer without knowing how the product attacks keratin is not bold — it is negligent.")
        .replace(
            "The trade reinvents itself roughly every decade.",
            "The trade pivots hard roughly once a decade.")
        .replace(
            "It is one of the few trades where steel meets keratin meets psychology — and has done since barbers also pulled teeth.",
            "Steel on keratin on human psychology — few trades run all three at once, and barbers have done it since they also pulled teeth.")
    )


# ── Run ────────────────────────────────────────────────────────────────

_pass_idx = 0
_pass_fns = [rewrite_pass_1, rewrite_pass_2, rewrite_pass_3]

def rewrite_agent(text, span_info):
    global _pass_idx
    if _pass_idx >= len(_pass_fns):
        return text
    fn = _pass_fns[_pass_idx]
    _pass_idx += 1
    return fn(text, span_info)


print("=" * 72)
print("DRAFTPROOF — Full Rewrite Pipeline (3 forced passes)")
print("=" * 72)

# 1. Scan original
print("\n[1/3] Scanning original text...")
t0 = time.time()
orig_report = run_pipeline(TEXT, do_rewrite=False)
elapsed_orig = time.time() - t0
print(f"  Done in {elapsed_orig:.1f}s — {orig_report.finding_count} findings, tier={orig_report.overall_tier.value.upper()}")

# 2. Run 3-pass rewrite (force all passes: target=0, min_improvement=0)
print("\n[2/3] Running 3-pass rewrite...")
scanner = PredictabilityScanner()
mp_result = multi_pass_rewrite(
    text=TEXT,
    scanner=scanner,
    max_passes=3,
    target_top10=0.0,
    min_improvement=0.0,
    rewrite_fn=rewrite_agent,
)
rewritten_text = mp_result.final_text
print(f"  Passes: {len(mp_result.passes)}, Converged: {mp_result.converged}")
print(f"  Reason: {mp_result.convergence_reason}")
print(f"  Risk: {mp_result.original_metrics.risk:.4f} → {mp_result.final_metrics.risk:.4f}")
print(f"  Top-10: {mp_result.original_metrics.top10_ratio:.1%} → {mp_result.final_metrics.top10_ratio:.1%}")

# 3. Scan rewritten text
print("\n[3/3] Scanning rewritten text...")
t0 = time.time()
rw_report = run_pipeline(rewritten_text, do_rewrite=False)
elapsed_rw = time.time() - t0
print(f"  Done in {elapsed_rw:.1f}s — {rw_report.finding_count} findings, tier={rw_report.overall_tier.value.upper()}")

# 4. Build combined report
builder = ReportBuilder()
builder.set_meta(
    scan_time=elapsed_orig + elapsed_rw,
    original_text=TEXT,
    rewritten_text=rewritten_text,
)
builder._pred_summary = orig_report.predictability

# Copy original findings
for tier_key in ["critical", "high", "medium", "low"]:
    for finding in orig_report.findings_by_tier.get(tier_key, []):
        builder._findings.append(finding)

# Add rewrite data
builder.add_rewrite(mp_result)

# Populate rewritten detection stats
if builder._rewrite_summary and rw_report.predictability:
    builder._rewrite_summary.rewritten_tier = rw_report.overall_tier.value.upper()
    builder._rewrite_summary.rewritten_findings = rw_report.finding_count
    builder._rewrite_summary.rewritten_distribution = dict(rw_report.predictability.risk_distribution)

report = builder.build()

# 5. Save markdown report
md = render_markdown(report, verbose=True)
with open(os.path.join(_POC_DIR, "test_rewrite_report.md"), "w") as f:
    f.write(md)

# 6. Print comparison summary
print()
print("=" * 72)
print("COMPARISON SUMMARY")
print("=" * 72)
print(f"  {'Metric':<25} {'Original':>10} {'Rewritten':>10} {'Change':>10}")
print(f"  {'─' * 55}")
print(f"  {'Overall Tier':<25} {orig_report.overall_tier.value.upper():>10} {rw_report.overall_tier.value.upper():>10}")
o_risk = orig_report.predictability.overall_risk
r_risk = rw_report.predictability.overall_risk
print(f"  {'Predictability Risk':<25} {o_risk:>10.4f} {r_risk:>10.4f} {r_risk - o_risk:>+10.4f}")
print(f"  {'Total Findings':<25} {orig_report.finding_count:>10} {rw_report.finding_count:>10} {rw_report.finding_count - orig_report.finding_count:>+10}")
bd = orig_report.predictability.risk_distribution
rd = rw_report.predictability.risk_distribution
print(f"  {'HIGH findings':<25} {bd.get('high', 0):>10} {rd.get('high', 0):>10} {rd.get('high', 0) - bd.get('high', 0):>+10}")
print(f"  {'MEDIUM findings':<25} {bd.get('medium', 0):>10} {rd.get('medium', 0):>10} {rd.get('medium', 0) - bd.get('medium', 0):>+10}")
print(f"  {'LOW findings':<25} {bd.get('low', 0):>10} {rd.get('low', 0):>10} {rd.get('low', 0) - bd.get('low', 0):>+10}")

# 7. Print the rewrite section of the markdown
print()
print("=" * 72)
print("MARKDOWN REWRITE SECTION")
print("=" * 72)
idx = md.find("## Rewrite")
if idx >= 0:
    print(md[idx:])
else:
    print("No Rewrite section found in markdown report")

print()
print(f"Report saved to test_rewrite_report.md")
