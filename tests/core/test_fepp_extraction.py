"""Standalone FEP+ extraction and source-cohort provenance regression."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mdpp.core import fepp as extractor
from mdpp.core import fepp_results as reader


def test_extractor_validates_standalone_manifest_and_state_cohort(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "version.txt").write_text("Suite TEST\n")
    runner = suite / "run"
    runner.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
output=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then
        output="$2"
        shift 2
    else
        shift
    fi
done
if [[ "${output}" == *mappings.json ]]; then
    printf '{"schema_version":2,"mapping_fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","environment_fingerprint":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","edges":[]}\n' >"${output}"
else
    printf 'Ligand1,Ligand2,bennett_ddg,bennett_ddg_error,convergence,ligand RMSD,REST exchange,CCC Conv\nB,A,-1.25,0.2,Good,Good,Good,Good\n' >"${output}_ddG.csv"
fi
"""
    )
    runner.chmod(0o755)
    fmp = tmp_path / "standalone_out.fmp"
    fmp.write_text("completed map\n")
    mapping_fingerprint = "a" * 64
    snapshot_contents = {
        "input_map.fmp": "input map\n",
        "input_map.edge": "abc:def  # A -> B\n",
        "input_map.provenance": "map provenance\n",
        "input_map_mappings.json": json.dumps({
            "schema_version": 2,
            "mapping_fingerprint": mapping_fingerprint,
            "edges": [{"name_a": "A", "name_b": "B"}],
        }),
        "input_ligands.maegz": "ligands\n",
        "input_ligand_validation.json": '{"valid":true}\n',
    }
    for filename, content in snapshot_contents.items():
        (tmp_path / filename).write_text(content)
    hashes = {
        filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        for filename in snapshot_contents
    }
    state_values = {
        "schema_version": "1",
        "build_schema": "fepp-input-build-v5",
        "suite_release": "Suite TEST",
        "state": "open",
        "receptor_sha256": "b" * 64,
        "receptor_provenance_sha256": "c" * 64,
        "pose_viewer_sha256": "d" * 64,
        "pose_viewer_provenance_sha256": "e" * 64,
        "ligand_bundle_sha256": hashes["input_ligands.maegz"],
        "ligand_validation_sha256": hashes["input_ligand_validation.json"],
        "map_sha256": hashes["input_map.fmp"],
        "edge_sha256": hashes["input_map.edge"],
        "map_provenance_sha256": hashes["input_map.provenance"],
        "atom_mapping_fingerprint": mapping_fingerprint,
        "atom_mapping_json_sha256": hashes["input_map_mappings.json"],
    }
    state_path = tmp_path / "input_state_inputs.tsv"
    state_path.write_text(
        "\n".join(f"{key}\t{value}" for key, value in state_values.items()) + "\n"
    )
    state_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest_values = {
        "schema_version": "2",
        "workflow_type": "single_rbfe",
        "state": "open",
        "seed": "2041",
        "jobname": "standalone",
        "prepare_only": "0",
        "suite_release": "Suite TEST",
        "forcefield": "OPLS4",
        "custom_charge_mode": "assign",
        "water": "SPC",
        "ensemble": "muVT",
        "time_ps": "5000",
        "equilibration_time_ps": "20",
        "lambda_windows": "12",
        "salt_molar": "0.0",
        "input_his267_state": "HIE",
        "input_prep_ph": "7.0",
        "input_prep_rmsd_A": "0.3",
        "input_map_topology": "normal",
        "map_sha256": hashes["input_map.fmp"],
        "edge_sha256": hashes["input_map.edge"],
        "map_provenance_sha256": hashes["input_map.provenance"],
        "atom_mapping_fingerprint": mapping_fingerprint,
        "atom_mapping_json_sha256": hashes["input_map_mappings.json"],
        "ligand_bundle_sha256": hashes["input_ligands.maegz"],
        "ligand_validation_sha256": hashes["input_ligand_validation.json"],
        "state_inputs_sha256": state_hash,
    }
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\n".join(f"{key}\t{value}" for key, value in manifest_values.items()) + "\n"
    )

    run = extractor._read_manifest(
        manifest,
        state="open",
        fmp=fmp,
        schrodinger=suite,
    )
    assert run.workflow_type == "single_rbfe"
    assert run.normalized_schema_version == 3
    assert run.input_lineage["input_state_inputs_sha256"] == state_hash
    assert run.expected_edge_pairs == frozenset({("A", "B")})

    output = tmp_path / "standalone.csv"
    assert (
        extractor.extract_fep_results(
            str(fmp),
            str(output),
            state="open",
            schrodinger=str(suite),
            manifest=str(manifest),
        )
        == 1
    )
    parsed = reader.read_edge_file(output, expected_state="open")
    assert parsed.schema == "normalized_v3"
    assert parsed.workflow_type == "single_rbfe"
    assert parsed.input_lineage is not None
    assert parsed.input_lineage["input_state_inputs_sha256"] == state_hash

    state_values["state"] = "closed"
    state_path.write_text(
        "\n".join(f"{key}\t{value}" for key, value in state_values.items()) + "\n"
    )
    manifest_values["state_inputs_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest.write_text(
        "\n".join(f"{key}\t{value}" for key, value in manifest_values.items()) + "\n"
    )
    with pytest.raises(ValueError, match="state does not match"):
        extractor._read_manifest(
            manifest,
            state="open",
            fmp=fmp,
            schrodinger=suite,
        )
