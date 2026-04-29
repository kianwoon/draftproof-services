"""Citation Checker -- PoC Demo

Tests the citation cross-checker against realistic scenarios:
  1. Clean APA: all citations match bibliography
  2. Mismatch APA: cited but not in bib + in bib but not cited
  3. Numeric/IEEE: [1]-style cross-check
  4. Uncited claims: sentences with claim indicators but no citation

Run:  cd poc/citation && python demo.py
"""

from scanner import CitationScanner

# ── Test scenarios ──────────────────────────────────────────────────

APA_CLEAN_BODY = """Artificial intelligence has transformed banking operations.
Smith et al. (2022) found that AI improves efficiency in document checking by 40%.
Recent work (Jones & Lee, 2023) confirms these findings in Asian markets.
However, the evidence is mixed (Park, 2021)."""

APA_CLEAN_BIB = """Smith, A., Brown, B., and Chen, C. (2022). AI adoption in financial services. Journal of FinTech, 15(3), 112-128.
Jones, D. and Lee, E. (2023). Machine learning in Asian banking. International Journal of Banking Technology, 8(1), 45-67.
Park, S. (2021). Limitations of AI in regulatory compliance. FinReg Review, 4(2), 89-102."""

APA_MISMATCH_BODY = """AI has transformed banking (Smith, 2022). The evidence is growing (Jones, 2023).
Williams (2020) argued that compliance risks remain underexplored.
Recent meta-analyses (Chen et al., 2024) confirm these concerns."""

APA_MISMATCH_BIB = """Smith, A. (2022). AI adoption in financial services. Journal of FinTech, 15(3), 112-128.
Jones, D. (2023). Machine learning in banking. IJBT, 8(1), 45-67.
Park, S. (2021). Limitations of AI in compliance. FinReg Review, 4(2), 89-102."""

IEEE_BODY = """Artificial intelligence has transformed banking operations [1].
AI improves document checking efficiency by 40% [2].
These findings are confirmed in Asian markets [2,3].
However, compliance risks remain [4]."""

IEEE_BIB = """[1] Smith, A., Brown, B., and Chen, C. (2022). AI adoption in financial services. Journal of FinTech, 15(3), 112-128.
[2] Jones, D. and Lee, E. (2023). Machine learning in Asian banking. IJBT, 8(1), 45-67.
[3] Park, S. (2021). Cross-border AI regulation. FinReg Review, 4(2), 89-102.
Note: [4] is cited in text but missing from bibliography.
[5] Kumar, R. (2020). Blockchain in trade finance. JFT, 12(4), 201-215."""

UNCITED_CLAIMS_BODY = """Artificial intelligence has transformed banking operations.
Studies show that AI improves efficiency in document checking by 40%.
According to the World Bank, over 60% of financial institutions use some form of AI.
This technology reduces operational costs significantly.
Research demonstrates that AI-driven trading algorithms outperform human traders in high-frequency scenarios.
Personal observations suggest that the industry is moving toward full automation."""

UNCITED_CLAIMS_BIB = """Smith, A. (2022). AI in financial services. Journal of FinTech, 15(3), 112-128."""


# ── Display ─────────────────────────────────────────────────────────

RISK_ICON = {"high": "[H]", "medium": "[M]", "low": "[L]"}


def print_case(name: str, result) -> None:
    print(f"\n{'=' * 72}")
    print(f"  CASE: {name}")
    print(f"  Style: {result.citation_style}  |  In-text: {result.stats['in_text_count']}  |  Bib entries: {result.stats['bib_count']}")
    print(f"{'=' * 72}")

    # Summary
    print(f"  Missing from bib: {result.stats['missing_from_bib']}")
    print(f"  Uncited in body:  {result.stats['uncited_in_body']}")
    print(f"  Uncited claims:   {result.stats['uncited_claims']}")

    if not result.findings:
        print("  No issues found.\n")
        return

    for f in result.findings:
        icon = RISK_ICON.get(f.risk_level, "[?]")
        print(f"\n  {icon} {f.risk_level.upper():6s}  {f.finding_type}")
        print(f"    Detail: {f.detail}")
        print(f"    Evidence: {f.evidence[:100]}{'...' if len(f.evidence) > 100 else ''}")
        print(f"    -> {f.recommendation}")

    print()


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("DraftProof Citation Checker -- PoC\n")

    scanner = CitationScanner()

    # Summary table
    cases = [
        ("APA clean (all match)", APA_CLEAN_BODY, APA_CLEAN_BIB),
        ("APA mismatch (citations vs bib)", APA_MISMATCH_BODY, APA_MISMATCH_BIB),
        ("IEEE numeric", IEEE_BODY, IEEE_BIB),
        ("Uncited claims", UNCITED_CLAIMS_BODY, UNCITED_CLAIMS_BIB),
    ]

    print(f"  {'Case':<35s} {'Style':>8s} {'InT':>4s} {'Bib':>4s} {'M2B':>4s} {'B2I':>4s} {'Claims':>7s}")
    print(f"  {'-'*35} {'-'*8} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*7}")

    results = {}
    for name, body, bib in cases:
        result = scanner.scan(body, bib)
        results[name] = result
        s = result.stats
        print(f"  {name:<35s} {result.citation_style:>8s} {s['in_text_count']:>4d} {s['bib_count']:>4d} "
              f"{s['missing_from_bib']:>4d} {s['uncited_in_body']:>4d} {s['uncited_claims']:>7d}")

    # Detailed findings
    for name, result in results.items():
        print_case(name, result)

    # ── Takeaway ────────────────────────────────────────────────────
    print("=" * 72)
    print("  TAKEAWAY")
    print("=" * 72)
    print("""
  The citation checker detects:
    1. Cited-in-text but missing from bibliography (HIGH risk)
    2. In bibliography but never cited in body (LOW risk)
    3. Claim-like sentences without any citation (MEDIUM/HIGH)
    4. Works with APA/Harvard, IEEE/numeric, and narrative styles

  Key design choices:
    - Normalised match keys (surname+year) for APA cross-checking
    - Numeric reference tracking for IEEE style
    - Claim indicators flag sentences that need citations
    - Citation style auto-detected from in-text patterns

  Limitations (PoC):
    - Simple sentence splitting (production: use spaCy/nltk)
    - Author name matching is exact (production: fuzzy match)
    - No DOI/URL validation
    - No integration with reference managers (Zotero, Mendeley)

  Next: Wire into mini-pipeline with predictability + similarity scanners.
""")


if __name__ == "__main__":
    main()
