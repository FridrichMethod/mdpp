"""Regression tests for ligand graph and stereochemistry preparation."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mdpp.prep.ligand import assign_topology

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "openfe"


def test_assign_topology_restores_bond_orders_and_preserves_coordinates() -> None:
    template = Chem.MolFromSmiles("c1ccccc1O")
    source = Chem.AddHs(Chem.Mol(template))
    assert AllChem.EmbedMolecule(source, randomSeed=13) == 0
    source = Chem.RemoveHs(source)
    for bond in source.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in source.GetAtoms():
        atom.SetIsAromatic(False)
    source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    source.SetProp("_Name", "phenol")
    coords = source.GetConformer().GetPositions().copy()

    result = assign_topology(source, template)

    assert Chem.MolToSmiles(Chem.RemoveHs(result)) == Chem.MolToSmiles(template)
    assert result.GetProp("_Name") == "phenol"
    np.testing.assert_allclose(result.GetConformer().GetPositions()[: len(coords)], coords)
    np.testing.assert_allclose(source.GetConformer().GetPositions(), coords)
    assert result.GetNumAtoms() > source.GetNumAtoms()


@pytest.mark.parametrize("conformation", ["3a7r", "1x2h"])
@pytest.mark.parametrize(
    "ligand_name",
    [mol.GetProp("_Name") for mol in Chem.SmilesMolSupplier(str(EXAMPLES / "ligands_amp_fep.smi"))],
)
def test_bundled_ligand_stereo_survives_topology_assignment(
    conformation: str, ligand_name: str, tmp_path: Path
) -> None:
    import MDAnalysis as mda

    template = next(
        mol
        for mol in Chem.SmilesMolSupplier(str(EXAMPLES / "ligands_amp_fep.smi"))
        if mol.GetProp("_Name") == ligand_name
    )
    complex_path = EXAMPLES / "pdbs" / f"FLE03_{ligand_name}_{conformation}_model_0.pdb"
    ligand_path = tmp_path / "ligand.pdb"
    mda.Universe(complex_path).select_atoms("chainID B").write(ligand_path)
    source = Chem.MolFromPDBFile(str(ligand_path), removeHs=True)

    result = assign_topology(source, template)

    assert Chem.MolToSmiles(Chem.RemoveHs(result)) == Chem.MolToSmiles(template)
    np.testing.assert_allclose(
        result.GetConformer().GetPositions()[: source.GetNumAtoms()],
        source.GetConformer().GetPositions(),
    )
    assert result.GetAtomWithIdx(0).GetPDBResidueInfo().GetName() == (
        source.GetAtomWithIdx(0).GetPDBResidueInfo().GetName()
    )


@pytest.mark.parametrize(
    ("source_smiles", "template_smiles"),
    [("C/C=C/C", "C/C=C\\C"), ("C[C@H](O)F", "C[C@@H](O)F")],
)
def test_conflicting_3d_stereochemistry_is_rejected(
    source_smiles: str, template_smiles: str
) -> None:
    source = Chem.AddHs(Chem.MolFromSmiles(source_smiles))
    assert AllChem.EmbedMolecule(source, randomSeed=21) == 0
    # Mimic PDB inputs with coordinates but no authoritative stereo tags.
    Chem.RemoveStereochemistry(source)
    with pytest.raises(ValueError, match="stereochemistry"):
        assign_topology(source, Chem.MolFromSmiles(template_smiles))


def test_template_stereo_without_coordinates_is_preserved() -> None:
    source = Chem.MolFromSmiles("CC=CC")
    template = Chem.MolFromSmiles("C/C=C/C")
    result = assign_topology(source, template)
    assert Chem.MolToSmiles(Chem.RemoveHs(result)) == Chem.MolToSmiles(template)


def test_partial_template_is_rejected() -> None:
    with pytest.raises(ValueError, match="heavy-atom graph"):
        assign_topology(Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC"))
