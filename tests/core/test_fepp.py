"""Reusable FEP+ mapping and vendor-interpreter boundary regressions."""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from rdkit import Chem

from mdpp.core import _fepp_mapping_worker as worker
from mdpp.core.fepp import extract_edge_mappings, fep_mapping_fingerprint, read_fep_edges


def test_read_fep_edges_preserves_direction_and_order(tmp_path: Path) -> None:
    path = tmp_path / "network.edge"
    path.write_text("\nabc:def  # B -> A\nxy:z # A -> C\n")
    assert read_fep_edges(str(path)) == [("B", "A"), ("A", "C")]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not an edge\n", "malformed edge line"),
        ("a:b # A -> A\n", "self-edge"),
        ("a:b # A -> B\nb:a # B -> A\n", "duplicate ligand pair"),
        ("\n", "no edges parsed"),
    ],
)
def test_read_fep_edges_rejects_invalid_networks(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "network.edge"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        read_fep_edges(path)


def mapping_record() -> dict[str, Any]:
    """Return one mapping with deliberately unsorted atom correspondence."""
    return {
        "name_a": "A",
        "name_b": "B",
        "similarity": 0.5,
        "molblock_a": "coordinates A",
        "molblock_b": "coordinates B",
        "core_a": [1, 3],
        "core_b": [2, 1],
        "dummy_a": [2],
        "dummy_b": [3],
    }


def test_mapping_fingerprint_tracks_correspondence_not_coordinates() -> None:
    original = mapping_record()
    moved = {**original, "molblock_a": "translated", "molblock_b": "rotated"}
    remapped = {**original, "core_b": [1, 2]}
    assert fep_mapping_fingerprint([original]) == fep_mapping_fingerprint([moved])
    assert fep_mapping_fingerprint([original]) != fep_mapping_fingerprint([remapped])
    assert len(fep_mapping_fingerprint([original])) == 64


@pytest.fixture
def fake_vendor_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install small vendor API fixtures without loading licensed libraries."""
    names = [
        "schrodinger",
        "schrodinger.structure",
        "schrodinger.application",
        "schrodinger.application.scisol",
        "schrodinger.application.scisol.packages",
        "schrodinger.application.scisol.packages.fep",
        "schrodinger.application.scisol.packages.fep.graph",
        "schrodinger.rdkit",
        "schrodinger.rdkit.rdkit_adapter",
    ]
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(modules["schrodinger.structure"], "write_ct_to_string", str, raising=False)
    monkeypatch.setattr(
        modules["schrodinger.rdkit.rdkit_adapter"],
        "to_rdkit",
        lambda structure: structure.mol,
        raising=False,
    )
    atom_a = SimpleNamespace(title="A", mol=Chem.MolFromSmiles("CCO"))
    atom_b = SimpleNamespace(title="B", mol=Chem.MolFromSmiles("NCC"))
    edge = SimpleNamespace(
        nodes=[SimpleNamespace(struc=atom_a), SimpleNamespace(struc=atom_b)],
        core_atoms=([3, 1], [1, 2]),
        dummy_atoms=([2], [3]),
        similarity=0.5,
    )
    graph = SimpleNamespace(edges_iter=lambda: iter([edge]), environment_struc=["protein", None])
    monkeypatch.setattr(
        modules["schrodinger.application.scisol.packages.fep.graph"],
        "Graph",
        SimpleNamespace(deserialize=lambda _path: graph),
        raising=False,
    )
    return graph


def test_worker_preserves_one_based_pairs_and_molblock_atom_order(
    tmp_path: Path, fake_vendor_modules: SimpleNamespace
) -> None:
    output = tmp_path / "mappings.json"
    assert worker.extract_edge_mappings(tmp_path / "input.fmp", output) == 1
    payload = json.loads(output.read_text())
    first = copy.deepcopy(payload)
    record = payload["edges"][0]
    assert payload["schema_version"] == 2
    assert record["core_a"] == [1, 3]
    assert record["core_b"] == [2, 1]
    assert record["dummy_a"] == [2]
    assert record["dummy_b"] == [3]
    mol_a = Chem.MolFromMolBlock(record["molblock_a"])
    mol_b = Chem.MolFromMolBlock(record["molblock_b"])
    assert [atom.GetSymbol() for atom in mol_a.GetAtoms()] == ["C", "C", "O"]
    assert [atom.GetSymbol() for atom in mol_b.GetAtoms()] == ["N", "C", "C"]
    edge = next(fake_vendor_modules.edges_iter())
    edge.core_atoms = ([1, 3], [2, 1])
    worker.extract_edge_mappings(tmp_path / "input.fmp", output)
    assert json.loads(output.read_text()) == first


def test_environment_fingerprint_preserves_slot_boundaries(
    fake_vendor_modules: SimpleNamespace,  # noqa: ARG001
) -> None:
    assert worker.environment_fingerprint(["ab", "c"]) != worker.environment_fingerprint([
        "a",
        "bc",
    ])
    assert worker.environment_fingerprint(["protein", None]) != worker.environment_fingerprint([
        None,
        "protein",
    ])
    assert worker.environment_fingerprint([None]) != worker.environment_fingerprint([])


@pytest.mark.parametrize(
    "path",
    [
        Path(worker.__file__),
        Path(__file__).parents[2] / "examples/fepp/extract_edge_mappings.py",
    ],
)
def test_suite_worker_and_cli_remain_python311_compatible(path: Path) -> None:
    ast.parse(path.read_text(), feature_version=(3, 11))
    # -I ignores the checkout and user site: --help must work without mdpp
    # or any vendor import, even in a separate interpreter environment.
    completed = subprocess.run(
        [sys.executable, "-I", str(path), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--fmp" in completed.stdout


def test_public_mapping_extractor_dispatches_bundled_worker(tmp_path: Path) -> None:
    suite = tmp_path / "suite with spaces"
    suite.mkdir()
    runner = suite / "run"
    output = tmp_path / "out dir" / "mappings.json"
    fmp = tmp_path / "input.fmp"
    fmp.write_text("map")
    runner.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "assert sys.argv[1] == 'python3'\n"
        "worker = pathlib.Path(sys.argv[2])\n"
        "assert worker.name == '_fepp_mapping_worker.py'\n"
        "assert worker.is_file()\n"
        "assert 'examples' not in worker.parts\n"
        "assert pathlib.Path(sys.argv[4]).read_text() == 'map'\n"
        "output = pathlib.Path(sys.argv[6])\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "output.write_text(json.dumps({'schema_version': 2, "
        "'mapping_fingerprint': 'a' * 64, 'environment_fingerprint': 'b' * 64, "
        "'edges': [{'name_a': 'A', 'name_b': 'B'}]}))\n"
    )
    runner.chmod(0o755)
    assert extract_edge_mappings(str(fmp), str(output), schrodinger=str(suite)) == 1
    assert json.loads(output.read_text())["mapping_fingerprint"] == "a" * 64
    with pytest.raises(ValueError, match="must differ"):
        extract_edge_mappings(fmp, fmp, schrodinger=suite)


def test_suite_example_cli_loads_worker_from_another_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_vendor_modules: SimpleNamespace,  # noqa: ARG001
) -> None:
    import runpy

    script = Path(__file__).parents[2] / "examples/fepp/extract_edge_mappings.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script), "-f", "input.fmp"])
    runpy.run_path(str(script), run_name="__main__")
    payload = json.loads((tmp_path / "input_mappings.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["edges"][0]["core_b"] == [2, 1]


def test_mapping_extractor_reports_vendor_failure(tmp_path: Path) -> None:
    runner = tmp_path / "run"
    runner.write_text("#!/usr/bin/env bash\nprintf 'vendor failure' >&2\nexit 7\n")
    runner.chmod(0o755)
    fmp = tmp_path / "input.fmp"
    fmp.write_text("map")
    with pytest.raises(RuntimeError, match="exit code 7: vendor failure"):
        extract_edge_mappings(fmp, tmp_path / "output.json", schrodinger=tmp_path)


def test_public_imports_and_compute_do_not_require_posix_file_locks(tmp_path: Path) -> None:
    code = """
import builtins
original = builtins.__import__
def import_without_fcntl(name, *args, **kwargs):
    if name == 'fcntl':
        raise ModuleNotFoundError('fcntl unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = import_without_fcntl
from mdpp.core import extract_fep_results, read_edge_file
from mdpp.analysis import CombinedEdge, compute_fep_network
result = compute_fep_network([CombinedEdge('A', 'B', 1., 1., 1, 0., 0)], reference='A')
assert result.relative_free_energy_kcal_mol.tolist() == [0., 1.]
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
