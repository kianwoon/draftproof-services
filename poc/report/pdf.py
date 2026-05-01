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
