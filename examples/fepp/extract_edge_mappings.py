#!/usr/bin/env python3
"""Extract FEP+ mappings under Schrodinger Suite Python.

This CLI loads the bundled standalone worker by file path because Suite 2025-4
uses Python 3.11 while importing mdpp requires Python 3.12 or newer.
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--fmp",
        type=Path,
        default=here / "tmp" / "open_map.fmp",
        help="FEP+ map file to read (default: tmp/open_map.fmp).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: <fmp-stem>_mappings.json).",
    )
    args = parser.parse_args()
    out_json = args.output or args.fmp.with_name(f"{args.fmp.stem}_mappings.json")
    worker = Path(__file__).resolve().parents[2] / "src/mdpp/core/_fepp_mapping_worker.py"
    extract = runpy.run_path(str(worker))["extract_edge_mappings"]
    n = extract(args.fmp, out_json)
    print(f"extracted {n} edge mappings -> {out_json}")


if __name__ == "__main__":
    main()
