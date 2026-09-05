#!/usr/bin/env python3
"""Validate example FEP+ SDF inputs against their SMILES provenance.

This phosphate dataset explicitly ignores phosphorus stereo tags introduced by
coordinate import; the reusable API preserves phosphorus stereochemistry by default.
"""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

# Load the exact source fingerprinted by build_fepp_inputs.sh. Suite Python 3.11
# supplies RDKit but cannot import mdpp, which requires Python 3.12 or newer.
_MODULE = Path(__file__).resolve().parents[2] / "src/mdpp/chem/validation.py"
validate_ligand_inputs = runpy.run_path(str(_MODULE))["validate_ligand_inputs"]


def main() -> None:
    """Run the command-line validator."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smiles",
        type=Path,
        default=here / "inputs" / "ligands_amp_fep.smi",
    )
    parser.add_argument(
        "--ligand-dir",
        type=Path,
        default=here / "inputs" / "ligands",
    )
    parser.add_argument("--expected-charge", type=int, default=-1)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate_ligand_inputs(
            args.smiles,
            args.ligand_dir,
            expected_charge=args.expected_charge,
            ignore_phosphorus_stereo=True,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"validated {report['ligand_count']} ligand inputs")


if __name__ == "__main__":
    main()
