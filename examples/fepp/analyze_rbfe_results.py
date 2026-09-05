#!/usr/bin/env python3
"""Analyze one-conformation, multi-ligand FEP+ RBFE networks.

With reference ligand ``r``, this script estimates the identifiable quantity

    DeltaDeltaG_i,r = DeltaG_bind(i) - DeltaG_bind(r).

Negative values mean stronger predicted binding than the reference under the
usual binding-free-energy sign convention. The reference value of zero and its
zero uncertainty fix the network gauge; they are not an absolute binding-free-
energy measurement. Production analysis requires normalized, provenance-
bearing exports written by ``extract_fep_results.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdpp.analysis.rbfe import compute_rbfe, write_rbfe
from mdpp.core.fepp_results import SUPPORTED_UNITS


def main() -> None:
    """Run the standalone RBFE command-line analyzer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Normalized edge CSV; repeat for independent repeats.",
    )
    parser.add_argument("--state", choices=("open", "closed"), required=True)
    parser.add_argument("--reference", required=True, help="Explicit reference ligand title.")
    parser.add_argument(
        "--unit",
        choices=tuple(SUPPORTED_UNITS),
        default="kcal/mol",
        help="Output energy unit (default: kcal/mol).",
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
        result = compute_rbfe(
            args.input,
            state=args.state,
            reference=args.reference,
            output_unit=args.unit,
        )
        write_rbfe(result, output_dir=args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"wrote FEP+ {args.state} RBFE analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
