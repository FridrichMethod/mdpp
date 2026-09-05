"""Executable checks of APBS/BrownDye example handoffs, without simulations."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_NOTEBOOK = _EXAMPLES / "browndye/browndye_prep.ipynb"


def _cell(index: int) -> str:
    return "".join(json.loads(_NOTEBOOK.read_text())["cells"][index]["source"])


@pytest.fixture()
def apbs_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stage independent component maps with a shared continuum model."""
    examples = tmp_path / "examples"
    settings = {
        "ionic_strength_m": 0.150,
        "solute_dielectric": 2.0,
        "solvent_dielectric": 78.54,
        "solvent_radius_a": 1.4,
        "temperature_k": 298.0,
    }
    for name in ("protein", "ligand"):
        base = examples / "apbs" / name / "tmp"
        (base / "ambertools").mkdir(parents=True)
        (base / "apbs").mkdir()
        (base / "apbs" / f"{name}.pqr").write_text("fixture PQR\n")
        (base / "ambertools" / f"{name}.pqr").write_text("new unmatched AmberTools PQR\n")
        (base / "apbs" / f"{name}.dx").write_text("fixture potential\n")
        (base / "apbs" / f"{name}.apbs.log").write_text("Debye length: 7.85 A\n")
        (base / "apbs" / f"{name}.settings.json").write_text(json.dumps(settings))
    (examples / "browndye").mkdir()
    monkeypatch.chdir(examples / "browndye")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    return examples


def test_notebook_preserves_matching_electrostatics_and_snapshot(apbs_fixtures: Path) -> None:
    namespace: dict = {}
    exec(_cell(1), namespace)
    exec(_cell(3), namespace)
    exec(_cell(11), namespace)
    staged = namespace["BDPREP_INTERMEDIATE"]
    root = ET.parse(staged / "input.xml").getroot()
    assert float(root.findtext("system/solvent/dielectric", default="nan")) == pytest.approx(78.54)
    assert float(root.findtext("system/solvent/debye_length", default="nan")) == pytest.approx(7.85)
    assert [
        float(core.findtext("dielectric", default="nan"))
        for core in root.findall("system/group/core")
    ] == [
        2.0,
        2.0,
    ]
    original = apbs_fixtures / "apbs/protein/tmp/apbs/protein.dx"
    original.write_text("later incompatible potential\n")
    assert (staged / "protein.dx").read_text() == "fixture potential\n"
    assert not (staged / "protein.dx").is_symlink()
    # A successful reparameterization followed by APBS failure must not leak
    # its newer, incompatible AmberTools PQR into this completed APBS bundle.
    assert (staged / "protein.pqr").read_text() == "fixture PQR\n"


@pytest.mark.parametrize(
    ("key", "value"),
    [("ionic_strength_m", 0.05), ("solvent_dielectric", 80.0), ("temperature_k", 310.0)],
)
def test_notebook_rejects_incompatible_component_maps(
    apbs_fixtures: Path, key: str, value: float
) -> None:
    path = apbs_fixtures / "apbs/ligand/tmp/apbs/ligand.settings.json"
    settings = json.loads(path.read_text())
    settings[key] = value
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match=f"inconsistent {key}"):
        exec(_cell(1), {})


def test_notebook_checks_both_debye_logs(apbs_fixtures: Path) -> None:
    namespace: dict = {}
    exec(_cell(1), namespace)
    log = apbs_fixtures / "apbs/ligand/tmp/apbs/ligand.apbs.log"
    log.write_text("Debye length: 15.0 A\n")
    with pytest.raises(ValueError, match="Inconsistent Debye lengths"):
        exec(_cell(3), namespace)


def test_impossible_contact_criterion_aborts_preparation(tmp_path: Path) -> None:
    for tool, output in (
        ("make_rxn_pairs", "<pairs><pair>1</pair><pair>2</pair></pairs>"),
        ("make_rxn_file", "<reactions/>"),
    ):
        path = tmp_path / tool
        path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{output}'\n")
        path.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    env.update(
        BDPREP_INTERMEDIATE=str(tmp_path),
        CORE0="protein",
        CORE1="ligand",
        RXN_SEARCH_DISTANCE="5.5",
        RXN_DISTANCE="10",
        RXN_NEEDED="3",
    )
    script = _cell(9).removeprefix("%%bash\n")
    result = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, env=env, check=False
    )
    assert result.returncode != 0
    assert "Insufficient bound-pose contact pairs" in result.stderr


def test_all_electrostatics_notebook_cells_have_valid_syntax() -> None:
    notebooks = [*_EXAMPLES.glob("apbs/*/*.ipynb"), _NOTEBOOK]
    for path in notebooks:
        for index, cell in enumerate(json.loads(path.read_text())["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.startswith("%%bash\n"):
                result = subprocess.run(
                    ["bash", "-n"],
                    input=source.removeprefix("%%bash\n"),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                assert result.returncode == 0, (path, index, result.stderr)
            else:
                compile(source, f"{path}:cell-{index}", "exec")


@pytest.mark.parametrize("succeeds", [False, True])
def test_apbs_publishes_matching_pqr_only_after_success(tmp_path: Path, succeeds: bool) -> None:
    """A failed solver must leave the preceding published PQR/map pair intact."""
    published = tmp_path / "published"
    intermediate = tmp_path / "intermediate"
    published.mkdir()
    intermediate.mkdir()
    for suffix in ("pqr", "dx", "settings.json"):
        (published / f"protein.{suffix}").write_text(f"old {suffix}\n")
    (intermediate / "protein.pqr").write_text("new PQR\n")
    (intermediate / "protein.in").write_text("new input\n")
    (intermediate / "protein.settings.json").write_text("new settings\n")
    solver = tmp_path / "apbs"
    solver.write_text("#!/bin/sh\n" + ("echo new >protein-PE0.dx\n" if succeeds else "exit 1\n"))
    solver.chmod(0o755)
    notebook = json.loads((_EXAMPLES / "apbs/protein/protein_apbs.ipynb").read_text())
    script = next(
        "".join(cell["source"]).removeprefix("%%bash\n")
        for cell in notebook["cells"]
        if 'apbs "$stem.in"' in "".join(cell["source"])
    )
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        APBS_DIR=str(published),
        APBS_INTERMEDIATE=str(intermediate),
    )
    result = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, env=env, check=False
    )
    assert (result.returncode == 0) == succeeds
    assert (published / "protein.pqr").read_text() == ("new PQR\n" if succeeds else "old pqr\n")
    assert (published / "protein.dx").read_text() == ("new\n" if succeeds else "old dx\n")
    assert (published / "protein.settings.json").read_text() == (
        "new settings\n" if succeeds else "old settings.json\n"
    )
