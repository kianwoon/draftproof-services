"""Markdown → PDF renderer using WeasyPrint."""

import re

import markdown as md_lib
from weasyprint import HTML


_CSS = """
@page {
    size: A4;
    margin: 2cm 2.5cm;
    @bottom-center {
        content: "DraftProof Report | page " counter(page) " of " counter(pages);
        font-size: 8pt;
        color: #999;
    }
}

body {
    font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #222;
}

h1 { font-size: 18pt; color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 4pt; }
h2 { font-size: 14pt; color: #333; border-bottom: 1px solid #eee; padding-bottom: 3pt; }
h3 { font-size: 12pt; color: #444; }

table {
    width: 100%;
    table-layout: auto;
    border-collapse: collapse;
    margin: 8pt 0;
    font-size: 9pt;
}
th {
    background: #f5f5f5;
    font-weight: 600;
    text-align: left;
    padding: 4pt 6pt;
    border: 1px solid #ddd;
    overflow-wrap: break-word;
    word-break: break-word;
}
td {
    padding: 3pt 6pt;
    border: 1px solid #ddd;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: break-word;
    word-break: break-word;
    -webkit-hyphens: auto;
    hyphens: auto;
}

code {
    background: #f4f4f4;
    padding: 1pt 3pt;
    border-radius: 2pt;
    font-size: 9pt;
    font-family: "SF Mono", "Menlo", "Monaco", monospace;
}

mark.placeholder {
    background: #fef08a;
    color: #1f2937;
    padding: 0 2pt;
    border-radius: 2pt;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}

pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 3pt;
    padding: 8pt 10pt;
    margin: 6pt 0;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.4;
}

pre code {
    background: none;
    padding: 0;
    border-radius: 0;
    font-size: 8.5pt;
}

blockquote {
    border-left: 3px solid #ccc;
    margin: 8pt 0;
    padding: 4pt 12pt;
    color: #555;
    background: #fafafa;
}

hr { border: none; border-top: 1px solid #ddd; margin: 12pt 0; }

img[src^="https://img.shields.io"] { height: 18px; }

.dp-executive-chart {
    margin: 4pt 0 12pt;
}

.dp-summary-bar {
    display: flex;
    flex-wrap: nowrap;
    align-items: stretch;
    gap: 0;
    margin: 0 0 12pt;
    border: 1px solid #d8e1ea;
    border-radius: 8pt;
    overflow: hidden;
    background: #fff;
}

.dp-summary-stat {
    flex: 1 1 0;
    min-width: 0;
    padding: 8pt 7pt;
    border-right: 1px solid #e2e8f0;
}

.dp-summary-stat:last-child {
    border-right: none;
}

.dp-risk-stat {
    display: flex;
    align-items: center;
    gap: 8pt;
    flex: 1.25 1 0;
    min-width: 0;
}

.dp-risk-icon {
    display: inline-block;
    width: 18pt;
    height: 18pt;
    font-size: 16pt;
    line-height: 18pt;
    text-align: center;
}

.dp-summary-value {
    display: block;
    color: #111827;
    font-size: 10.5pt;
    font-weight: 800;
    line-height: 1.05;
}

.dp-summary-label {
    display: block;
    margin-top: 4pt;
    color: #4b5563;
    font-size: 6.8pt;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.dp-signal-card {
    border: 1px solid #d8e1ea;
    border-radius: 8pt;
    background: #fff;
    overflow: visible;
    break-inside: auto;
}

.dp-signal-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12pt;
    padding: 13pt;
    border-bottom: 1px solid #e2e8f0;
}

.dp-signal-title-row {
    display: flex;
    gap: 10pt;
    align-items: flex-start;
}

.dp-signal-icon {
    display: inline-block;
    width: 32pt;
    height: 32pt;
    color: #0f766e;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 7pt;
    font-size: 18pt;
    line-height: 31pt;
    text-align: center;
}

.dp-kicker,
.dp-ai-score span,
.dp-scan-head span,
.dp-ratio-copy span {
    display: block;
    color: #64748b;
    font-size: 6.8pt;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.dp-signal-header h3 {
    margin: 3pt 0 7pt;
    color: #0f172a;
    font-size: 16pt;
    font-weight: 800;
    line-height: 1.15;
}

.dp-pill-row,
.dp-chip-row,
.dp-evidence-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4pt;
}

.dp-pill-row span,
.dp-chip-row strong,
.dp-evidence-row span {
    display: inline-block;
    min-height: 13pt;
    padding: 3pt 5pt;
    border-radius: 5pt;
    font-size: 6.8pt;
    font-weight: 800;
}

.dp-pill-row span {
    color: #0f766e;
    background: #f0fdfa;
    border: 1px solid #ccfbf1;
    text-transform: capitalize;
}

.dp-ai-score {
    min-width: 72pt;
    text-align: right;
}

.dp-ai-score strong {
    display: block;
    margin-top: 4pt;
    color: #b45309;
    font-size: 16pt;
    font-weight: 900;
    line-height: 1;
}

.dp-scan-head {
    display: flex;
    justify-content: space-between;
    gap: 12pt;
    padding: 12pt 13pt 6pt;
}

.dp-scan-head strong {
    display: block;
    margin-top: 3pt;
    color: #0f172a;
    font-size: 10.5pt;
    font-weight: 800;
}

.dp-scan-head em {
    color: #b45309;
    font-size: 12pt;
    font-style: normal;
    font-weight: 900;
}

.dp-ratio-card {
    display: block;
    margin: 4pt 13pt 12pt;
    padding: 10pt;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 7pt;
}

.dp-ratio-copy p {
    margin: 4pt 0 7pt;
    color: #1f2937;
    font-size: 9pt;
    font-weight: 650;
    line-height: 1.4;
}

.dp-chip-row strong {
    color: #334155;
    background: #fff;
    border: 1px solid #dbe4ea;
}

.dp-ratio-bars {
    display: block;
    margin-top: 8pt;
}

.dp-ratio-row,
.dp-core-row {
    display: block;
    margin-bottom: 7pt;
    break-inside: avoid;
}

.dp-ratio-row > span,
.dp-ratio-row > strong,
.dp-core-label span,
.dp-core-label strong {
    font-size: 7.8pt;
    font-weight: 800;
}

.dp-ratio-row > span,
.dp-core-label span {
    color: #475569;
}

.dp-ratio-row > strong,
.dp-core-label strong {
    float: right;
    color: #0f172a;
}

.dp-bar-track {
    clear: both;
    height: 6pt;
    margin-top: 3pt;
    overflow: hidden;
    border-radius: 999pt;
    background: #e8eef5;
}

.dp-ratio-fill,
.dp-core-fill {
    height: 100%;
    min-width: 2pt;
    border-radius: inherit;
}

.dp-human { background: #16a34a; }
.dp-ai { background: #c2410c; }

.dp-core-heading {
    margin: 8pt 13pt 7pt;
    color: #0f172a;
    font-size: 10pt;
    font-weight: 800;
}

.dp-core-bars {
    padding: 0 13pt 4pt;
}

.dp-core-label {
    min-height: 12pt;
}

.dp-evidence-row {
    margin: 8pt 13pt 0;
    padding-top: 8pt;
    border-top: 1px solid #e2e8f0;
}

.dp-evidence-row span {
    color: #334155;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.dp-confidence-note {
    margin: 8pt 13pt 12pt;
    color: #475569;
    font-size: 8pt;
}
"""


def render_pdf(markdown_text: str, output_path: str) -> str:
    """Convert markdown text to a styled PDF file.

    Args:
        markdown_text: Full markdown content.
        output_path: Where to write the .pdf file.

    Returns:
        Absolute path to the generated PDF.
    """
    extensions = ["tables", "fenced_code"]
    html_body = md_lib.markdown(markdown_text, extensions=extensions)

    # Inject colgroup into wide tables (6-column: findings & false-positives)
    colgroup_6 = '<colgroup><col style="width:3%"/><col style="width:4%"/><col style="width:4%"/><col style="width:36%"/><col style="width:26%"/><col style="width:27%"/></colgroup>'
    # Only tables whose header row starts with # (findings/false-positives pattern)
    html_body = re.sub(
        r'(<table>\s*<thead>\s*<tr>\s*<th[^>]*>\s*#\s*</th>)',
        lambda m: m.group(1).replace('<table>', '<table>' + colgroup_6),
        html_body,
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><style>{_CSS}</style></head>
<body>{html_body}</body>
</html>"""

    HTML(string=html_doc).write_pdf(output_path)
    return output_path
