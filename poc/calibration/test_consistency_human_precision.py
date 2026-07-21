"""Unit tests for the pure aggregation math in consistency_human_precision.py.

Mirrors test_consistency_esl_rates.py's convention: cover only the group-summary
arithmetic with synthetic counts, so these run in CI without any local corpus.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.consistency_human_precision import (
    combine_sources,
    outlier_distribution,
    summarize_source,
)


def test_outlier_distribution_buckets_at_three_plus():
    dist = outlier_distribution([0, 0, 1, 2, 3, 4, 5])
    assert dist == {"0": 2, "1": 1, "2": 1, "3+": 3}


def test_outlier_distribution_empty():
    assert outlier_distribution([]) == {"0": 0, "1": 0, "2": 0, "3+": 0}


def test_summarize_source_basic_rates():
    # 4 docs: 3 eligible (counts 0, 1, 3), 1 ineligible (short doc, excluded).
    s = summarize_source(
        outlier_counts=[0, 1, 3, 0],
        eligibles=[True, True, True, False],
        paragraph_counts=[8, 10, 12, 3],
    )
    assert s["n_docs"] == 4
    assert s["n_eligible"] == 3
    assert s["eligibility_rate_pct"] == 75.0
    assert s["flagged_doc_rate_pct"] == round(2 / 3 * 100, 2)
    assert s["mean_outliers_per_eligible_doc"] == round(4 / 3, 3)
    assert s["mean_paragraphs_per_eligible_doc"] == round(30 / 3, 3)
    assert s["distribution"] == {"0": 1, "1": 1, "2": 0, "3+": 1}


def test_summarize_source_no_eligible_docs():
    s = summarize_source(outlier_counts=[0, 0], eligibles=[False, False], paragraph_counts=[2, 3])
    assert s["n_docs"] == 2
    assert s["n_eligible"] == 0
    assert s["flagged_doc_rate_pct"] is None
    assert s["mean_outliers_per_eligible_doc"] is None
    assert s["mean_paragraphs_per_eligible_doc"] is None
    assert s["distribution"] == {"0": 0, "1": 0, "2": 0, "3+": 0}


def test_summarize_source_empty_input():
    s = summarize_source(outlier_counts=[], eligibles=[], paragraph_counts=[])
    assert s["n_docs"] == 0
    assert s["n_eligible"] == 0
    assert s["eligibility_rate_pct"] is None


def test_combine_sources_merges_raw_lists():
    per_source = {
        "gutenberg": ([0, 1], [True, True], [8, 9]),
        "raid_human": ([2], [True], [10]),
    }
    combined = combine_sources(per_source)
    assert combined["n_docs"] == 3
    assert combined["n_eligible"] == 3
    assert combined["flagged_doc_rate_pct"] == round(2 / 3 * 100, 2)
    assert combined["distribution"] == {"0": 1, "1": 1, "2": 1, "3+": 0}


def test_combine_sources_empty_dict():
    combined = combine_sources({})
    assert combined["n_docs"] == 0
    assert combined["n_eligible"] == 0
