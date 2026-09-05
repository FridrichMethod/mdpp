#!/usr/bin/env python3
"""Export raw Bennett edge results from a completed FEP+ output map.

The reusable extraction and provenance validation live in ``mdpp.core.fepp``.
Run this command in the mdpp environment; vendor reads use Suite subprocesses.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mdpp.core.fepp import extract_fep_results


def main() -> None:
    """Run the command-line extractor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fmp", type=Path, help="Completed FEP+ *_out.fmp file.")
    parser.add_argument("--state", choices=("open", "closed"), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Run manifest (default: manifest.tsv beside the FMP).",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--schrodinger",
        type=Path,
        default=Path(os.environ.get("SCHRODINGER", "/apps/schrodinger2025-4")),
        help="Schrodinger Suite root (default: $SCHRODINGER).",
    )
    args = parser.parse_args()
    manifest = args.manifest if args.manifest is not None else args.fmp.parent / "manifest.tsv"
    try:
        count = extract_fep_results(
            args.fmp,
            args.output,
            state=args.state,
            schrodinger=args.schrodinger,
            manifest=manifest,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {count} raw Bennett edges to {args.output}")


if __name__ == "__main__":
    main()
