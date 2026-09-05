"""End-to-end tests of the thin FEP+ extraction and analysis CLIs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from mdpp.core import fepp_results as reader

FEPP_DIR = Path(__file__).parents[3] / "examples" / "fepp"


def test_vendor_adapter_normalizes_results_and_rejects_blank_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(FEPP_DIR.parents[1] / "src"))
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "version.txt").write_text("Suite TEST\n")
    runner = suite / "run"
    runner.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${1-}" == "python3" && "$(basename "${2-}")" == "_fepp_mapping_worker.py" ]]; then
    shift 2
    output=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f) shift 2 ;;
            -o) output="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    fingerprint="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    environment="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    if [[ "${FAKE_WRONG_MAPPING:-0}" == "1" && "${output}" == *completed_mappings.json ]]; then
        fingerprint="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    fi
    if [[ "${FAKE_WRONG_ENVIRONMENT:-0}" == "1" && "${output}" == *completed_mappings.json ]]; then
        environment="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
    fi
    printf '{"schema_version":2,"mapping_fingerprint":"%s","environment_fingerprint":"%s","edges":[]}\n' \
        "${fingerprint}" "${environment}" >"${output}"
    exit 0
fi
output=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then
        output="$2"
        shift 2
    else
        shift
    fi
done
if [[ "${FAKE_BLANK:-0}" == "1" ]]; then
    printf 'Ligand1,Ligand2,bennett_ddg,bennett_ddg_error,convergence,ligand RMSD,REST exchange,CCC Conv\nA,B,,,Good,Good,Good,Good\n' >"${output}_ddG.csv"
elif [[ "${FAKE_WRONG_EDGE:-0}" == "1" ]]; then
    printf 'Ligand1,Ligand2,bennett_ddg,bennett_ddg_error,convergence,ligand RMSD,REST exchange,CCC Conv\nA,C,1.25,0.2,Good,Good,Good,Good\n' >"${output}_ddG.csv"
else
    printf 'Ligand1,Ligand2,bennett_ddg,bennett_ddg_error,convergence,ligand RMSD,REST exchange,CCC Conv\nB,A,-1.25,0.2,Good,Good,Good,Good\n' >"${output}_ddG.csv"
fi
"""
    )
    runner.chmod(0o755)
    fmp = tmp_path / "completed_out.fmp"
    fmp.write_text("synthetic completed map\n")
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
        "input_ligands.maegz": "ligand bundle\n",
        "input_ligand_validation.json": '{"status":"valid"}\n',
        "input_receptor_microstates.json": '{"identical":true}\n',
    }
    for filename, content in snapshot_contents.items():
        (tmp_path / filename).write_text(content)
    snapshot_hashes = {
        filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        for filename in snapshot_contents
    }
    paired_values = {
        "schema_version": "1",
        "build_schema": "fepp-input-build-v5",
        "suite_release": "Suite TEST",
        "allow_microstate_mismatch": "0",
        "open_receptor_sha256": "c" * 64,
        "closed_receptor_sha256": "d" * 64,
        "ligand_bundle_sha256": snapshot_hashes["input_ligands.maegz"],
        "ligand_validation_sha256": snapshot_hashes["input_ligand_validation.json"],
        "receptor_microstates_sha256": snapshot_hashes["input_receptor_microstates.json"],
        "open_map_sha256": snapshot_hashes["input_map.fmp"],
        "closed_map_sha256": "e" * 64,
        "open_edge_sha256": snapshot_hashes["input_map.edge"],
        "closed_edge_sha256": snapshot_hashes["input_map.edge"],
        "open_map_provenance_sha256": snapshot_hashes["input_map.provenance"],
        "closed_map_provenance_sha256": "f" * 64,
        "open_atom_mapping_fingerprint": mapping_fingerprint,
        "closed_atom_mapping_fingerprint": mapping_fingerprint,
    }
    paired_path = tmp_path / "input_paired_inputs.tsv"
    paired_path.write_text(
        "\n".join(f"{key}\t{value}" for key, value in paired_values.items()) + "\n"
    )
    snapshot_hashes[paired_path.name] = hashlib.sha256(paired_path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\n".join([
            "schema_version\t1",
            "state\topen",
            "seed\t2041",
            "jobname\tcompleted",
            "prepare_only\t0",
            "suite_release\tSuite TEST",
            "forcefield\tOPLS4",
            "custom_charge_mode\tassign",
            "water\tSPC",
            "ensemble\tmuVT",
            "time_ps\t5000",
            "equilibration_time_ps\t20",
            "lambda_windows\t12",
            "salt_molar\t0.0",
            "input_his267_state\tHIE",
            "input_prep_ph\t7.0",
            "input_prep_rmsd_A\t0.3",
            "input_map_topology\tnormal",
            "input_allow_microstate_mismatch\t0",
            f"map_sha256\t{snapshot_hashes['input_map.fmp']}",
            f"edge_sha256\t{snapshot_hashes['input_map.edge']}",
            f"map_provenance_sha256\t{snapshot_hashes['input_map.provenance']}",
            f"atom_mapping_fingerprint\t{mapping_fingerprint}",
            f"atom_mapping_json_sha256\t{snapshot_hashes['input_map_mappings.json']}",
            f"ligand_bundle_sha256\t{snapshot_hashes['input_ligands.maegz']}",
            f"ligand_validation_sha256\t{snapshot_hashes['input_ligand_validation.json']}",
            f"receptor_microstates_sha256\t{snapshot_hashes['input_receptor_microstates.json']}",
            f"paired_inputs_sha256\t{snapshot_hashes['input_paired_inputs.tsv']}",
        ])
        + "\n"
    )
    output = tmp_path / "open.csv"
    command = [
        sys.executable,
        str(FEPP_DIR / "extract_fep_results.py"),
        str(fmp),
        "--state",
        "open",
        "--schrodinger",
        str(suite),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    parsed = reader.read_edge_file(output, expected_state="open")
    assert len(parsed.edges) == 1
    assert parsed.edges[0].source == "A"
    assert parsed.edges[0].target == "B"
    assert parsed.edges[0].ddg == pytest.approx(1.25)
    assert parsed.schema == "normalized_v2"
    assert parsed.run_seed == 2041
    assert parsed.run_manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    provenance_path = output.with_suffix(".csv.provenance.json")
    assert provenance_path.is_file()
    provenance = json.loads(provenance_path.read_text())
    assert provenance["schema_version"] == 3
    assert provenance["publication_status"] == "complete_commit_marker"
    assert provenance["input_environment_fingerprint"] == "c" * 64
    assert provenance["completed_environment_fingerprint"] == "c" * 64
    assert output.with_suffix(".csv.lock").is_file()
    assert output.with_name("open.vendor_edges.csv").is_file()

    stale_nodes = output.with_name("open.vendor_nodes.csv")
    stale_nodes.write_text("stale previous export\n")
    repeated = subprocess.run(command, text=True, capture_output=True, check=False)
    assert repeated.returncode == 0, repeated.stderr
    assert not stale_nodes.exists()

    blank_output = tmp_path / "blank.csv"
    blank_env = dict(os.environ, FAKE_BLANK="1")
    blank = subprocess.run(
        [*command[:-1], str(blank_output)],
        text=True,
        capture_output=True,
        check=False,
        env=blank_env,
    )
    assert blank.returncode != 0
    assert "blank required field 'bennett_ddg'" in blank.stderr
    assert not blank_output.exists()

    wrong_output = tmp_path / "wrong.csv"
    wrong_env = dict(os.environ, FAKE_WRONG_EDGE="1")
    wrong = subprocess.run(
        [*command[:-1], str(wrong_output)],
        text=True,
        capture_output=True,
        check=False,
        env=wrong_env,
    )
    assert wrong.returncode != 0
    assert "completed FMP edge set does not match" in wrong.stderr
    assert not wrong_output.exists()

    wrong_mapping_output = tmp_path / "wrong_mapping.csv"
    wrong_mapping_env = dict(os.environ, FAKE_WRONG_MAPPING="1")
    wrong_mapping = subprocess.run(
        [*command[:-1], str(wrong_mapping_output)],
        text=True,
        capture_output=True,
        check=False,
        env=wrong_mapping_env,
    )
    assert wrong_mapping.returncode != 0
    assert "completed FMP atom mapping does not match" in wrong_mapping.stderr
    assert not wrong_mapping_output.exists()

    wrong_environment_output = tmp_path / "wrong_environment.csv"
    wrong_environment_env = dict(os.environ, FAKE_WRONG_ENVIRONMENT="1")
    wrong_environment = subprocess.run(
        [*command[:-1], str(wrong_environment_output)],
        text=True,
        capture_output=True,
        check=False,
        env=wrong_environment_env,
    )
    assert wrong_environment.returncode != 0
    assert "completed FMP receptor/environment does not match" in wrong_environment.stderr
    assert not wrong_environment_output.exists()

    (suite / "version.txt").unlink()
    missing_release_output = tmp_path / "missing_release.csv"
    missing_release = subprocess.run(
        [*command[:-1], str(missing_release_output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_release.returncode != 0
    assert "Suite release metadata not found" in missing_release.stderr
    assert not missing_release_output.exists()


def test_paired_cli_preserves_selectivity_and_anchor_outputs(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    open_path = tmp_path / "open.csv"
    closed_path = tmp_path / "closed.csv"
    write_edges(open_path, "open", [("A", "B", 1.25, 0.2)])
    write_edges(closed_path, "closed", [("A", "B", 0.5, 0.3)])
    anchor = tmp_path / "anchor.json"
    anchor.write_text(
        json.dumps({
            "reference": "A",
            "quantity": "binding_free_energy_open_minus_closed",
            "sign_convention": "open_minus_closed",
            "estimate": 0.5,
            "standard_uncertainty": 0.1,
            "unit": "kcal/mol",
        })
    )
    output = tmp_path / "analysis"
    result = subprocess.run(
        [
            sys.executable,
            str(FEPP_DIR / "analyze_fep_results.py"),
            "--open",
            str(open_path),
            "--closed",
            str(closed_path),
            "--reference",
            "A",
            "--anchor",
            str(anchor),
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env=dict(os.environ, PYTHONPATH=str(FEPP_DIR.parents[1] / "src")),
    )
    assert result.returncode == 0, result.stderr
    assert "wrote FEP+ open/closed analysis" in result.stdout
    summary = json.loads((output / "analysis.json").read_text())
    assert summary["unit"] == "kcal/mol"
    assert summary["relative_selectivity"]["value"] == pytest.approx([0.0, 0.75])
    assert summary["anchored"]["value"] == pytest.approx([0.5, 1.25])
    marker = json.loads((output / "analysis.provenance.json").read_text())
    assert set(marker["files"]) == {
        "analysis.json",
        "selectivity.csv",
        "open_edge_residuals.csv",
        "closed_edge_residuals.csv",
        "open_cycles.csv",
        "closed_cycles.csv",
    }
