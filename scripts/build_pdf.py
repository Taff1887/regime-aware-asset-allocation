"""Render the Markdown reports to PDF (no system dependencies).

Uses ``markdown`` + ``xhtml2pdf`` so it works anywhere with the ``docs`` extra::

    uv run --extra docs python scripts/build_pdf.py

Produces ``reports/report.pdf`` and ``reports/executive_summary.pdf`` with the
figures embedded.
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

CSS = """
@page { size: A4; margin: 1.6cm 1.7cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9.5pt; line-height: 1.4; color: #222; }
h1 { font-size: 20pt; color: #14315a; }
h2 { font-size: 14pt; color: #14315a; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 16px; }
h3 { font-size: 11.5pt; color: #2a4a73; }
h4 { font-size: 10pt; color: #2a4a73; }
p, li { font-size: 9.5pt; }
code { font-family: Courier, monospace; background: #f3f3f3; font-size: 8.5pt; }
pre { background: #f3f3f3; padding: 6px; font-size: 8pt; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; }
th, td { border: 0.5px solid #aaa; padding: 3px 5px; font-size: 8.5pt; text-align: right; }
th { background: #e8eef6; color: #14315a; text-align: center; }
td:first-child, th:first-child { text-align: left; }
img { width: 16cm; }
blockquote { color: #444; border-left: 3px solid #bbb; padding-left: 8px; font-style: italic; }
"""


def link_callback(uri: str, rel: str) -> str:
    """Resolve relative image URIs (../figures/...) against the reports dir."""
    if uri.startswith(("http://", "https://", "data:")):
        return uri
    p = (REPORTS / uri).resolve()
    return str(p)


def convert(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    # constrain image widths for the page
    html_body = re.sub(r"<img ", '<img width="600" ', html_body)
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    with open(pdf_path, "wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, link_callback=link_callback, encoding="utf-8")
    status = "OK" if not result.err else f"{result.err} errors"
    print(f"{md_path.name} -> {pdf_path.name}: {status}")


def main() -> None:
    convert(REPORTS / "report.md", REPORTS / "report.pdf")
    convert(REPORTS / "executive_summary.md", REPORTS / "executive_summary.pdf")


if __name__ == "__main__":
    main()
