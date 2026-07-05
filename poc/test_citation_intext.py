"""In-text citation detection: page-locator + 'and'-joiner support.

Guards the 2026-07-05 fix to citation/scanner.py APA_MULTI_RE (the pattern
_detect_in_text actually uses). Before the fix, common real citations were
missed: author-year with a page locator ("(Author, 2023, p.41)") and the "and"
author joiner ("(Acemoglu and Restrepo, 2020)"). Non-citations must still be
rejected.

allow-hardcode: the string literals below are TEST FIXTURES (example citation /
non-citation spans asserted against the regex), not a scoring or matching oracle
compared against document text.
"""
from citation.scanner import APA_MULTI_RE, APA_RE


def test_apa_multi_matches_page_locators_and_and_joiner():
    should_match = [
        "(Siemens, 2023)",
        "(Siemens, 2023, p.41)",
        "(Smith, 2020, pp. 12-15)",
        "(Jones et al., 2019, p. 7)",
        "(Acemoglu & Restrepo, 2020)",
        "(Acemoglu and Restrepo, 2020)",
        "(Smith & Jones, 2022; Lee, 2021)",
        "(Lee, 2021a; Park, 2021b, p.3)",
    ]
    for c in should_match:
        assert APA_MULTI_RE.search(c), f"expected APA_MULTI_RE to match {c!r}"


def test_apa_multi_rejects_non_citations():
    should_not_match = [
        "(see figure 2)",
        "(the year 2023)",
        "(Company, Inc.)",
        "(edited by Smith)",
    ]
    for c in should_not_match:
        assert not APA_MULTI_RE.search(c), f"expected NO match for {c!r}"


def test_apa_single_supports_page_locator():
    assert APA_RE.search("(Siemens, 2023, p.41)")
    assert APA_RE.search("(Siemens, 2023)")
    assert not APA_RE.search("(see figure 2)")


def test_regex_counts_two_inline_cites():
    text = ("Output rose sharply (Siemens, 2023, p.41). "
            "Displacement effects are contested (Acemoglu and Restrepo, 2020).")
    hits = APA_MULTI_RE.findall(text)
    assert len(hits) == 2, f"expected 2 inline cites, got {len(hits)}: {hits}"
