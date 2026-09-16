#!/usr/bin/env python3
"""Quick, high-yield proofing scan for scientific manuscripts.

Goal: catch *high-confidence* defects that are easy to miss in a manual pass:
- duplicated punctuation (", ,", ",,", "..", '" , ,')
- semicolon-capitalization anomalies ('; A' etc.)
- equation-adjacent malformed frames ("into ... into ...", "plugging ... into together")
- common product/language capitalization (javascript/iOS/iPhone)
- arctan(x/y) quadrant ambiguity patterns

Usage:
  python scripts/proofing_scan.py <path-to-pdf-or-text> [--max-hits 80]

Output:
  One line per hit:
    [RULE_ID] p<page>: <snippet>

Notes:
- Page numbers are 1-indexed.
- Snippets are best-effort and may include line breaks collapsed to spaces.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


def _read_pdf_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        txt = re.sub(r"\s+", " ", txt).strip()
        pages.append(txt)
    return pages


def _read_text_as_single_page(path: Path) -> list[str]:
    txt = path.read_text(errors="ignore")
    txt = re.sub(r"\s+", " ", txt).strip()
    return [txt]


def _snip(text: str, start: int, end: int, window: int = 60) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    snippet = text[lo:hi]
    return snippet.strip()


def _find_regex(
    rule_id: str,
    pattern: re.Pattern,
    pages: list[str],
    max_hits: int,
) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for i, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        for m in pattern.finditer(page_text):
            hits.append((rule_id, i, _snip(page_text, m.start(), m.end())))
            if len(hits) >= max_hits:
                return hits
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="PDF or text file")
    ap.add_argument(
        "--max-hits",
        type=int,
        default=80,
        help="Cap total hits across all rules",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    if path.suffix.lower() == ".pdf":
        pages = _read_pdf_pages(path)
    else:
        pages = _read_text_as_single_page(path)

    rules: list[tuple[str, re.Pattern]] = [
        ("PUNC_DOUBLE_COMMA", re.compile(r",\s*,")),
        ("PUNC_DOUBLE_PERIOD", re.compile(r"\.\s*\.")),
        ("PUNC_QUOTE_DOUBLE_COMMA", re.compile(r"\"\s*,\s*,")),
        ("SEMICOLON_CAP", re.compile(r";\s+[A-Z]")),
        ("FRAME_INTO_INTO", re.compile(r"\binto\b.{0,80}?\binto\b", re.IGNORECASE)),
        (
            "FRAME_PLUGGING_INTO_TOGETHER",
            re.compile(r"plugging\s+into\s+together", re.IGNORECASE),
        ),
        ("CAP_JAVASCRIPT", re.compile(r"\bjavascript\b")),
        ("CAP_IOS", re.compile(r"\bios\b")),
        ("CAP_IPHONE", re.compile(r"\biphone\b")),
        (
            "ARCTAN_DIV",
            re.compile(
                r"\barctan\s*\(\s*[^()]{0,40}?/[^()]{0,40}?\)",
                re.IGNORECASE,
            ),
        ),
    ]

    remaining = args.max_hits
    out: list[tuple[str, int, str]] = []
    for rule_id, pat in rules:
        if remaining <= 0:
            break
        hits = _find_regex(rule_id, pat, pages, remaining)
        out.extend(hits)
        remaining = args.max_hits - len(out)

    if not out:
        print("[OK] No high-confidence pattern-scan hits found.")
        return

    for rule_id, page, snippet in out:
        print(f"[{rule_id}] p{page}: {snippet}")


if __name__ == "__main__":
    main()