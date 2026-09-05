"""Execute analysis templates on synthetic data to detect workflow/API errors.

The seeded peptide is deliberately synthetic and is not a validation of physical
conformations, thermodynamic convergence, or actual simulation results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pytest

NOTEBOOKS = Path(__file__).resolve().parents[2] / "examples/gromacs"


@pytest.fixture(scope="module")
def notebook_inputs(tmp_path_factory):
    """Create a bonded 60-residue periodic peptide with reproducible fluctuations."""
    root = tmp_path_factory.mktemp("gromacs_notebook_inputs")
    topology = md.Topology()
    chain = topology.add_chain()
    base = []
    atoms = [
        ("N", md.element.nitrogen, [-0.12, 0.0, 0.0]),
        ("H", md.element.hydrogen, [-0.18, 0.03, 0.0]),
        ("CA", md.element.carbon, [0.0, 0.0, 0.0]),
        ("C", md.element.carbon, [0.12, 0.02, 0.0]),
        ("O", md.element.oxygen, [0.18, 0.10, 0.0]),
    ]
    for i in range(60):
        residue = topology.add_residue("ALA", chain, resSeq=i + 1)
        origin = np.array([0.4 * np.cos(i * np.pi / 2), 0.4 * np.sin(i * np.pi / 2), 0.075 * i])
        for name, element, offset in atoms:
            topology.add_atom(name, element, residue)
            base.append(origin + offset)
    topology.create_standard_bonds()
    rng = np.random.default_rng(31)
    xyz = np.array(base)[None] + rng.normal(scale=0.015, size=(150, topology.n_atoms, 3))
    traj = md.Trajectory(
        xyz,
        topology,
        time=np.arange(150) * 10.0,
        unitcell_lengths=np.tile([12, 12, 12], (150, 1)),
        unitcell_angles=np.tile([90, 90, 90], (150, 1)),
    )
    traj[0].save_pdb(str(root / "topology.pdb"))
    traj.save_xtc(str(root / "trajectory.xtc"))
    return root


@pytest.mark.parametrize("notebook", sorted(p.name for p in NOTEBOOKS.glob("*.ipynb")))
def test_gromacs_notebook_executes(notebook, notebook_inputs, tmp_path, monkeypatch) -> None:
    """All code cells should execute after replacing only template input paths."""
    monkeypatch.chdir(tmp_path)
    cells = json.loads((NOTEBOOKS / notebook).read_text())["cells"]
    context: dict[str, Any] = {}
    try:
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            source = source.replace("/path/to/topology.pdb", str(notebook_inputs / "topology.pdb"))
            source = source.replace(
                "/path/to/trajectory.xtc", str(notebook_inputs / "trajectory.xtc")
            )
            exec(compile(source, f"{notebook}:cell{index}", "exec"), context)
    finally:
        plt.close("all")
    assert context["traj"].n_frames == 30
