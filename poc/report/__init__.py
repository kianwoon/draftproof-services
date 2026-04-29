"""DraftProof Report — __init__.py"""

from .report import (
    Tier,
    TIER_ORDER,
    TIER_ICON,
    Finding,
    PredictabilitySummary,
    SimilaritySummary,
    CitationSummary,
    RewriteSummary,
    DraftReport,
    ReportBuilder,
    report_to_dict,
)
from .render import render_report, print_report, render_markdown, print_markdown

__all__ = [
    "Tier",
    "TIER_ORDER",
    "TIER_ICON",
    "Finding",
    "PredictabilitySummary",
    "SimilaritySummary",
    "CitationSummary",
    "RewriteSummary",
    "DraftReport",
    "ReportBuilder",
    "report_to_dict",
    "render_report",
    "print_report",
    "render_markdown",
    "print_markdown",
]
