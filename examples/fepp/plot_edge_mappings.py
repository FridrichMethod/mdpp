#!/usr/bin/env python3
"""Render this example's extracted FEP+ mappings using mdpp.plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from mdpp.plots.fepp import save_fep_mappings


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-j",
        "--json",
        type=Path,
        default=here / "tmp" / "open_map_mappings.json",
        help="Extractor JSON (default: tmp/open_map_mappings.json).",
    )
    parser.add_argument(
        "-o",
        "--pdf",
        type=Path,
        default=None,
        help="Output PDF (default: <json-stem>.pdf).",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Output overview PNG (default: <json-stem>.png).",
    )
    args = parser.parse_args()
    pdf_path = args.pdf or args.json.with_suffix(".pdf")
    png_path = args.png or args.json.with_suffix(".png")
    data = json.loads(args.json.read_text())
    data.setdefault("fmp", args.json.stem)
    n = save_fep_mappings(data, pdf_path=pdf_path, png_path=png_path)
    print(f"rendered {n} edge mappings\n  PDF: {pdf_path}\n  PNG: {png_path}")


if __name__ == "__main__":
    main()
