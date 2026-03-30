"""Tests for mdpp.plots.three_d."""

from __future__ import annotations

from unittest.mock import Mock

import mdtraj as md
import nglview
import py3Dmol
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mdpp.plots import make_atom_labels_3d, view_mol_3d, view_traj_3d


@pytest.fixture()
def ethanol_3d() -> Chem.rdchem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    return mol


def _conformer_positions(mol: Chem.rdchem.Mol) -> list[tuple[float, float, float]]:
    conformer = mol.GetConformer()
    return [
        (
            conformer.GetAtomPosition(atom_idx).x,
            conformer.GetAtomPosition(atom_idx).y,
            conformer.GetAtomPosition(atom_idx).z,
        )
        for atom_idx in range(mol.GetNumAtoms())
    ]


class TestViewMol3D:
    def test_returns_py3dmol_viewer(self, ethanol_3d):
        viewer = view_mol_3d(ethanol_3d)
        assert isinstance(viewer, py3Dmol.view)

    def test_raises_without_conformer(self):
        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError, match="at least one conformer"):
            view_mol_3d(mol)

    def test_does_not_modify_input_conformer(self, ethanol_3d):
        before_positions = _conformer_positions(ethanol_3d)
        ethanol_3d.GetAtomWithIdx(0).SetDoubleProp("PartialCharge", -0.12)

        viewer = view_mol_3d(ethanol_3d)

        assert isinstance(viewer, py3Dmol.view)
        assert _conformer_positions(ethanol_3d) == before_positions
        assert ethanol_3d.GetAtomWithIdx(0).GetDoubleProp("PartialCharge") == pytest.approx(-0.12)

    def test_accepts_position_labels(self, ethanol_3d):
        labels = [{"text": "C0", "position": {"x": 0.0, "y": 0.0, "z": 0.0}}]
        viewer = view_mol_3d(ethanol_3d, labels=labels)
        assert isinstance(viewer, py3Dmol.view)

    def test_resolves_atom_index_labels(self, ethanol_3d):
        labels = [{"text": "O", "atom_index": 2, "fontColor": "red"}]
        viewer = view_mol_3d(ethanol_3d, labels=labels)
        assert isinstance(viewer, py3Dmol.view)

    def test_rejects_invalid_labels(self, ethanol_3d):
        labels = [{"text": "bad"}]
        with pytest.raises(ValueError, match="exactly one of 'atom_index' or 'position'"):
            view_mol_3d(ethanol_3d, labels=labels)


class TestMakeAtomLabels3D:
    def test_requires_text_fn(self, ethanol_3d):
        with pytest.raises(ValueError, match="requires a text_fn"):
            make_atom_labels_3d(ethanol_3d)

    def test_returns_one_label_per_atom(self, ethanol_3d):
        labels = make_atom_labels_3d(
            ethanol_3d,
            text_fn=lambda atom: atom.GetSymbol(),
            base_style={"fontSize": 10},
        )

        assert len(labels) == ethanol_3d.GetNumAtoms()
        assert all("text" in label for label in labels)
        assert all("position" in label for label in labels)
        assert all(label["fontSize"] == 10 for label in labels)

    def test_supports_atom_subset(self, ethanol_3d):
        labels = make_atom_labels_3d(
            ethanol_3d,
            atom_indices=[0, 2],
            text_fn=lambda atom: str(atom.GetIdx()),
        )

        assert [label["text"] for label in labels] == ["0", "2"]

    def test_supports_charge_style_callbacks(self, ethanol_3d):
        for atom in ethanol_3d.GetAtoms():
            atom.SetDoubleProp("PartialCharge", (-1.0) ** atom.GetIdx() * 0.1)

        labels = make_atom_labels_3d(
            ethanol_3d,
            text_fn=lambda atom: f"{atom.GetDoubleProp('PartialCharge'):+.2f}",
            color_fn=lambda atom: "red" if atom.GetDoubleProp("PartialCharge") < 0 else "blue",
            base_style={"showBackground": True},
        )

        assert labels[0]["text"].startswith("+")
        assert labels[1]["fontColor"] == "red"
        assert labels[0]["showBackground"] is True


class TestViewTraj3D:
    def test_returns_ngl_widget(self, two_atom_trajectory: md.Trajectory):
        widget = view_traj_3d(two_atom_trajectory)
        assert isinstance(widget, nglview.NGLWidget)

    def test_uses_nglview_defaults_when_representations_omitted(
        self,
        two_atom_trajectory: md.Trajectory,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_widget = Mock()
        monkeypatch.setattr(nglview, "show_mdtraj", lambda _traj: fake_widget)

        widget = view_traj_3d(two_atom_trajectory)

        assert widget is fake_widget
        fake_widget.clear_representations.assert_not_called()
        fake_widget.add_representation.assert_not_called()

    def test_empty_representations_are_allowed(self, two_atom_trajectory: md.Trajectory):
        widget = view_traj_3d(two_atom_trajectory, representations=[])
        assert isinstance(widget, nglview.NGLWidget)

    def test_accepts_custom_representations(self, two_atom_trajectory: md.Trajectory):
        widget = view_traj_3d(
            two_atom_trajectory,
            representations=[
                {"type": "ball+stick", "selection": "all", "color": "element"},
            ],
        )
        assert isinstance(widget, nglview.NGLWidget)

    def test_requires_representation_type(self, two_atom_trajectory: md.Trajectory):
        with pytest.raises(ValueError, match="must include a 'type' field"):
            view_traj_3d(two_atom_trajectory, representations=[{"selection": "all"}])
