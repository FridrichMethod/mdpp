"""CLI compatibility tests for the refactored FEP+ example helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

FEPP_DIR = Path(__file__).parents[3] / "examples" / "fepp"


def test_ligand_validation_cli_preserves_example_defaults(tmp_path: Path) -> None:
    report = tmp_path / "validation.json"
    completed = subprocess.run(
        [sys.executable, str(FEPP_DIR / "validate_ligand_inputs.py"), "-o", str(report)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(report.read_text())
    assert data["ligand_count"] == 12
    assert {item["formal_charge"] for item in data["ligands"]} == {-1}


def test_plotting_clis_work_outside_example_directory(tmp_path: Path) -> None:
    edge = tmp_path / "map.edge"
    edge.write_text("a:b # A -> B\n")
    records = {}
    for name, smiles in [("A", "CC"), ("B", "CO")]:
        molecule = Chem.MolFromSmiles(smiles)
        molecule.SetProp("_Name", name)
        Chem.MolToMolFile(molecule, str(tmp_path / f"{name}.sdf"))
        records[name] = Chem.MolToMolBlock(molecule)
    network = tmp_path / "map.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(FEPP_DIR / "plot_fep_map.py"),
            "-e",
            str(edge),
            "-l",
            str(tmp_path),
            "-o",
            str(network),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert network.read_bytes().startswith(b"\x89PNG")
    mappings = tmp_path / "map_mappings.json"
    mappings.write_text(
        json.dumps({
            "fmp": "test",
            "edges": [
                {
                    "name_a": "A",
                    "name_b": "B",
                    "similarity": 0.5,
                    "molblock_a": records["A"],
                    "molblock_b": records["B"],
                    "core_a": [1, 2],
                    "core_b": [1, 2],
                    "dummy_a": [],
                    "dummy_b": [],
                }
            ],
        })
    )
    completed = subprocess.run(
        [sys.executable, str(FEPP_DIR / "plot_edge_mappings.py"), "--json", str(mappings)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert mappings.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    assert mappings.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
