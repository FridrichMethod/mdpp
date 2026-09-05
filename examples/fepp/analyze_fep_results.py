#!/usr/bin/env python3
"""Analyze paired open/closed FEP+ RBFE networks from raw Bennett edge results.

The two RBFE networks do not identify an absolute open-versus-closed binding
preference.  With reference ligand ``r``, this script estimates

    D_i^(r) = [G_open(i) - G_open(r)] - [G_closed(i) - G_closed(r)].

Negative values mean ligand ``i`` is more open-selective than the reference.
An optional, independently determined open-minus-closed anchor for the
reference converts these relative values into an absolute anchored quantity.

Paired analysis requires the provenance-bearing normalized schema written by
``extract_fep_results.py``.  Its low-level reader also accepts the raw
``*_ddG.csv`` written by Schrodinger's ``fmp2excel.py`` so that the extractor
can validate vendor output.  Only raw Bennett edge values and uncertainties
are fitted.  Vendor cycle-closure-adjusted values and ``ccc_ddg_error`` are
intentionally not statistical inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdpp.analysis.fepp import compute_fep_selectivity, write_fep_selectivity
from mdpp.core.fepp_results import SUPPORTED_UNITS


def main() -> None:
    """Run the command-line analyzer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open",
        type=Path,
        action="append",
        required=True,
        help="Open-state edge CSV; repeat the option for independent repeats.",
    )
    parser.add_argument(
        "--closed",
        type=Path,
        action="append",
        required=True,
        help="Closed-state edge CSV; repeat the option for independent repeats.",
    )
    parser.add_argument("--reference", required=True, help="Explicit reference ligand title.")
    parser.add_argument(
        "--unit",
        choices=tuple(SUPPORTED_UNITS),
        default="kcal/mol",
        help="Output energy unit (default: kcal/mol).",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        help="Optional independent absolute open-minus-closed reference anchor JSON.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for analysis.json and diagnostic CSVs.",
    )
    args = parser.parse_args()
    try:
        result = compute_fep_selectivity(
            args.open,
            closed_paths=args.closed,
            reference=args.reference,
            output_unit=args.unit,
            anchor_path=args.anchor,
        )
        write_fep_selectivity(result, output_dir=args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"wrote FEP+ open/closed analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
