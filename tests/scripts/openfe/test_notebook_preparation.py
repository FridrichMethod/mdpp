"""Scientific input/cache regressions without requiring OpenFE or GPU simulations."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "openfe"
SPEC = importlib.util.spec_from_file_location("workflow_utils", EXAMPLES / "workflow_utils.py")
assert SPEC is not None and SPEC.loader is not None
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)


def test_fingerprint_tracks_contents_and_settings_not_enumeration_order(tmp_path: Path) -> None:
    a, b = tmp_path / "A.sdf", tmp_path / "B.sdf"
    a.write_text("pose A")
    b.write_text("pose B")
    settings = {"method": "am1bcc", "version": "1"}
    initial = UTILS.input_digest([a, b], settings=settings)
    assert UTILS.input_digest([b, a], settings=dict(reversed(list(settings.items())))) == initial
    a.write_text("different pose A")
    assert UTILS.input_digest([a, b], settings=settings) != initial
    a.write_text("pose A")
    assert UTILS.input_digest([a, b], settings={**settings, "method": "nagl"}) != initial
    assert UTILS.input_digest([a, b], settings={**settings, "version": "2"}) != initial


@pytest.mark.parametrize("notebook", ["rbfe", "rbfe_open_closed"])
def test_charge_cell_reuses_exact_inputs_and_regenerates_changed_inputs(
    notebook: str, tmp_path: Path
) -> None:
    """Same ligand count must not hide a changed pose/chemistry or charge method."""
    nb = json.loads((EXAMPLES / f"{notebook}.ipynb").read_text())
    ligand_dir = tmp_path / "ligands"
    ligand_dir.mkdir()
    source = ligand_dir / "A.sdf"
    source.write_text("original pose")
    calls = []
    settings = {"method": "am1bcc"}

    @dataclass
    class ChargedMol:
        name: str

        def to_sdf(self) -> str:
            return source.read_text()

    class ChargeSettings:
        partial_charge_method = "am1bcc"
        off_toolkit_backend = "ambertools"
        number_of_conformers = None
        nagl_model = None

        def model_dump(self, **kwargs: object) -> dict[str, str]:  # noqa: ARG002
            return settings.copy()

    def charge(**kwargs: object) -> list[ChargedMol]:
        calls.append(kwargs)
        return [ChargedMol("A")]

    namespace = {
        "OpenFFPartialChargeSettings": lambda **_kwargs: ChargeSettings(),
        "input_digest": UTILS.input_digest,
        "LIGAND_DIR": ligand_dir,
        "smiles_dict": {"A": object()},
        "SOFTWARE_VERSIONS": {"openfe": "test"},
        "CHARGED_LIGAND_ROOT": tmp_path / "charges",
        "ligands": [ChargedMol("A")],
        "SmallMoleculeComponent": SimpleNamespace(
            from_sdf_file=lambda path: Path(path).read_text()
        ),
        "bulk_assign_partial_charges": charge,
        "mp": SimpleNamespace(cpu_count=lambda: 1),
    }
    code = compile("".join(nb["cells"][11]["source"]), f"{notebook}:charging", "exec")
    exec(code, namespace)
    original_cache = namespace["CHARGED_LIGAND_DIR"]
    exec(code, namespace)
    assert len(calls) == 1
    assert namespace["charged_ligands"] == ["original pose"]

    source.write_text("new pose / new chemical identity, same name and count")
    exec(code, namespace)
    assert len(calls) == 2
    assert namespace["CHARGED_LIGAND_DIR"] != original_cache
    settings["method"] = "nagl"
    exec(code, namespace)
    assert len(calls) == 3
    # An incomplete cache must not be accepted merely because its directory exists.
    (namespace["CHARGED_LIGAND_DIR"] / "A.sdf").unlink()
    exec(code, namespace)
    assert len(calls) == 4


@pytest.mark.parametrize("notebook", ["rbfe", "rbfe_open_closed"])
def test_notebook_cells_compile_without_stale_outputs(notebook: str) -> None:
    nb = json.loads((EXAMPLES / f"{notebook}.ipynb").read_text())
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{notebook}:cell{index}", "exec")
            assert not cell["outputs"]
            assert cell["execution_count"] is None
