"""Prepared receptor chemistry comparison and isolated Suite reader tests."""

from __future__ import annotations

import ast
import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mdpp.prep import _fepp_receptor_worker as worker
from mdpp.prep.fepp import (
    compare_receptor_microstates,
    compare_receptor_summaries,
    enforce_receptor_microstate_policy,
    summarize_receptor,
)

FEPP_DIR = Path(__file__).parents[2] / "examples" / "fepp"


def _summary(*, label: str = "HIE", charge: int = 0) -> dict[str, Any]:
    return {
        "path": "receptor.mae",
        "atom_count": 2,
        "formal_charge": charge,
        "residues": {
            "A:10": {
                "label": label,
                "base_residue": "HIS",
                "formal_charge": charge,
                "heavy_atom_topology": [{"atom": "NE2", "element": "N", "bonded_heavy_atoms": []}],
                "heavy_atoms": [
                    {
                        "atom": "NE2",
                        "element": "N",
                        "formal_charge": charge,
                        "attached_hydrogens": ["HE2"],
                    }
                ],
            }
        },
    }


def _mock_reader(monkeypatch: pytest.MonkeyPatch, structures: list[Any]) -> None:
    vendor = ModuleType("schrodinger")
    vendor.structure = SimpleNamespace(StructureReader=lambda _path: iter(structures))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "schrodinger", vendor)


def _mock_receptor() -> SimpleNamespace:
    hydrogen = SimpleNamespace(element="H", pdbname=" HE2 ", formal_charge=0)
    nitrogen = SimpleNamespace(
        element="N", pdbname=" NE2 ", formal_charge=1, bonded_atoms=[hydrogen]
    )
    residue = SimpleNamespace(
        chain=" A ", resnum=10, inscode=" B ", pdbres=" HIP ", atom=[nitrogen, hydrogen]
    )
    return SimpleNamespace(residue=[residue], atom=[nitrogen, hydrogen])


def test_worker_summarizes_actual_atoms_and_normalizes_residue_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_reader(monkeypatch, [_mock_receptor()])
    report = worker.summarize(Path("receptor.mae"))
    assert report["atom_count"] == 2
    assert report["formal_charge"] == 1
    residue = report["residues"]["A:10B"]
    assert residue["label"] == "HIP"
    assert residue["base_residue"] == "HIS"
    assert residue["heavy_atoms"][0]["attached_hydrogens"] == ["HE2"]
    assert residue["heavy_atoms"][0]["formal_charge"] == 1


@pytest.mark.parametrize("structure_count", [0, 2])
def test_worker_requires_one_receptor(
    monkeypatch: pytest.MonkeyPatch, structure_count: int
) -> None:
    _mock_reader(monkeypatch, [_mock_receptor()] * structure_count)
    with pytest.raises(ValueError, match="expected one receptor structure"):
        worker.summarize(Path("receptor.mae"))


def test_worker_rejects_duplicate_residue_identifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    receptor = _mock_receptor()
    receptor.residue *= 2
    _mock_reader(monkeypatch, [receptor])
    with pytest.raises(ValueError, match="duplicate residue identifier"):
        worker.summarize(Path("receptor.mae"))


def test_heavy_atom_topology_detects_cross_residue_rewiring() -> None:
    residue_a = SimpleNamespace(chain="A", resnum=10, inscode="")
    residue_b = SimpleNamespace(chain="A", resnum=20, inscode="")
    neighbor_a = SimpleNamespace(pdbname=" SG ", element="S", getResidue=lambda: residue_a)
    neighbor_b = SimpleNamespace(pdbname=" SG ", element="S", getResidue=lambda: residue_b)
    atom_a = SimpleNamespace(pdbname=" SG ", element="S", bonded_atoms=[neighbor_a])
    atom_b = SimpleNamespace(pdbname=" SG ", element="S", bonded_atoms=[neighbor_b])
    topology_a = worker._heavy_atom_topology(atom_a)
    topology_b = worker._heavy_atom_topology(atom_b)
    assert topology_a != topology_b
    assert topology_a["bonded_heavy_atoms"][0]["residue"] == "A:10"
    assert topology_b["bonded_heavy_atoms"][0]["residue"] == "A:20"


def test_summary_comparison_uses_chemistry_instead_of_residue_labels() -> None:
    open_summary = _summary(label="HIE")
    closed_summary = _summary(label="HIS")
    expected = copy.deepcopy((open_summary, closed_summary))
    report = compare_receptor_summaries(open_summary, closed_summary)
    assert report["schema_version"] == 1
    assert report["identical_declared_microstates"]
    assert report["formal_charge_difference_open_minus_closed"] == 0
    assert (open_summary, closed_summary) == expected
    enforce_receptor_microstate_policy(report)


@pytest.mark.parametrize("change", ["missing_residue", "sequence", "composition"])
def test_microstate_override_cannot_allow_sequence_or_connectivity_changes(change: str) -> None:
    open_summary = _summary()
    closed_summary = _summary()
    if change == "missing_residue":
        closed_summary["residues"].clear()
    elif change == "sequence":
        closed_summary["residues"]["A:10"]["base_residue"] = "LYS"
    else:
        closed_summary["residues"]["A:10"]["heavy_atom_topology"][0]["bonded_heavy_atoms"] = [
            {"residue": "A:20", "atom": "C", "element": "C"}
        ]
    report = compare_receptor_summaries(open_summary, closed_summary)
    field = "composition_mismatches" if change == "composition" else "sequence_mismatches"
    assert len(report[field]) == 1
    with pytest.raises(ValueError, match="mismatches cannot be overridden"):
        enforce_receptor_microstate_policy(report, allow_microstate_mismatch=True)


@pytest.mark.parametrize("change", ["charge", "hydrogen_connectivity", "total_charge"])
def test_protonation_difference_requires_explicit_override(change: str) -> None:
    open_summary = _summary()
    closed_summary = _summary()
    if change == "charge":
        closed_summary = _summary(charge=1)
    elif change == "hydrogen_connectivity":
        closed_summary["residues"]["A:10"]["heavy_atoms"][0]["attached_hydrogens"] = []
    else:
        closed_summary["formal_charge"] = 1
    report = compare_receptor_summaries(open_summary, closed_summary)
    assert not report["identical_declared_microstates"]
    assert bool(report["microstate_mismatches"]) == (change != "total_charge")
    assert report["formal_charge_difference_open_minus_closed"] == (
        0 if change == "hydrogen_connectivity" else -1
    )
    with pytest.raises(ValueError, match="protonation/formal-charge signatures"):
        enforce_receptor_microstate_policy(report)
    enforce_receptor_microstate_policy(report, allow_microstate_mismatch=True)


def test_public_reader_uses_explicit_suite_and_standalone_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen_commands.append(command)
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        assert command[:2] == [str(tmp_path / "Suite root" / "run"), "python3"]
        assert Path(command[2]).name == "_fepp_receptor_worker.py"
        assert command[3:5] == ["--summary", "receptor with spaces.mae"]
        Path(command[-1]).write_text(json.dumps(_summary()))
        return subprocess.CompletedProcess(command, 0, "vendor log", "")

    monkeypatch.setattr("mdpp.prep.fepp.subprocess.run", run)
    assert (
        summarize_receptor("receptor with spaces.mae", schrodinger_root=tmp_path / "Suite root")
        == _summary()
    )
    assert len(seen_commands) == 1
    assert not Path(seen_commands[0][-1]).exists()


def test_public_reader_preserves_suite_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = subprocess.CalledProcessError(2, ["suite-run"], stderr="invalid receptor")

    def run(*_args: Any, **_kwargs: Any) -> None:
        raise failure

    monkeypatch.setattr("mdpp.prep.fepp.subprocess.run", run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        summarize_receptor("bad.mae", schrodinger_root="suite")
    assert error.value is failure
    assert error.value.stderr == "invalid receptor"


def test_public_reader_rejects_non_object_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command: list[str], **_kwargs: Any) -> None:
        Path(command[-1]).write_text("[]")

    monkeypatch.setattr("mdpp.prep.fepp.subprocess.run", run)
    with pytest.raises(ValueError, match="must be a JSON object"):
        summarize_receptor("bad.mae", schrodinger_root="suite")


def test_public_file_comparison_uses_both_paths_and_keeps_differences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_paths: list[str] = []

    def summarize(path: str, *, schrodinger_root: str) -> dict[str, Any]:
        assert schrodinger_root == "explicit-suite"
        seen_paths.append(path)
        return _summary(charge=int(path == "closed.mae"))

    monkeypatch.setattr("mdpp.prep.fepp.summarize_receptor", summarize)
    report = compare_receptor_microstates(
        "open.mae", "closed.mae", schrodinger_root="explicit-suite"
    )
    assert seen_paths == ["open.mae", "closed.mae"]
    assert report["formal_charge_difference_open_minus_closed"] == -1


def test_example_cli_writes_rejected_report_and_honors_allow_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_receptor = _mock_receptor()
    closed_receptor = _mock_receptor()
    closed_receptor.atom[0].formal_charge = 0
    vendor = ModuleType("schrodinger")
    vendor.structure = SimpleNamespace(  # type: ignore[attr-defined]
        StructureReader=lambda path: iter([
            open_receptor if path == "open.mae" else closed_receptor
        ])
    )
    monkeypatch.setitem(sys.modules, "schrodinger", vendor)
    output = tmp_path / "nested" / "report.json"
    script = FEPP_DIR / "compare_receptor_microstates.py"
    arguments = [str(script), "open.mae", "closed.mae", "-o", str(output)]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2
    assert json.loads(output.read_text())["formal_charge_difference_open_minus_closed"] == 1
    monkeypatch.setattr(sys, "argv", [*arguments, "--allow-mismatch"])
    runpy.run_path(str(script), run_name="__main__")
    assert not json.loads(output.read_text())["identical_declared_microstates"]


def test_worker_summary_cli_and_python311_syntax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_reader(monkeypatch, [_mock_receptor()])
    output = tmp_path / "summary.json"
    worker_path = Path(worker.__file__)
    monkeypatch.setattr(
        sys, "argv", [str(worker_path), "--summary", "receptor.mae", "-o", str(output)]
    )
    worker.main()
    assert json.loads(output.read_text())["formal_charge"] == 1
    for path in (
        worker_path,
        FEPP_DIR / "compare_receptor_microstates.py",
        FEPP_DIR / "validate_ligand_inputs.py",
        worker_path.parents[1] / "chem" / "validation.py",
    ):
        ast.parse(path.read_text(), feature_version=(3, 11))
