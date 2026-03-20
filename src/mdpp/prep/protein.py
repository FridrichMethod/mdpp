"""Protein structure preparation and manipulation utilities."""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
from Bio.PDB import Select
from openmm.app import PDBFile
from pdbfixer import PDBFixer

from mdpp._types import StrPath


class ChainSelect(Select):
    """Biopython ``Select`` subclass that accepts only specified PDB chains.

    Args:
        chain_ids: One or more chain identifiers to keep.

    Example::

        from Bio.PDB import PDBIO, PDBParser
        from mdpp.prep import ChainSelect

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", "complex.pdb")
        io = PDBIO()
        io.set_structure(structure)
        io.save("protein.pdb", ChainSelect("A"))
    """

    def __init__(self, chain_ids: str | list[str]) -> None:
        """Initialize the ChainSelect object.

        Args:
            chain_ids (str | list[str]): The chain IDs to keep.
        """
        self.chain_ids = {chain_ids} if isinstance(chain_ids, str) else set(chain_ids)

    def accept_chain(self, chain) -> int:  # type: ignore[override]
        """Return 1 if the chain should be kept, 0 otherwise."""
        return int(chain.id in self.chain_ids)


def fix_pdb(pdb_path: StrPath, fixed_pdb_path: StrPath, pH: float = 7.0) -> None:
    """Fix a PDB file by adding missing residues, atoms, and hydrogens.

    Removes heterogens (excluding water by default), identifies missing
    residues and atoms, then adds them back along with hydrogens at the
    specified pH.

    Args:
        pdb_path: Path to the input PDB file.
        fixed_pdb_path: Path where the fixed PDB will be written.
        pH: pH value for hydrogen placement.
    """
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=pH)

    with Path(fixed_pdb_path).open("w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)


def strip_solvent(
    traj: md.Trajectory,
    *,
    keep_ions: bool = False,
) -> md.Trajectory:
    """Remove solvent molecules from a trajectory.

    Args:
        traj: Input trajectory.
        keep_ions: If ``True``, retain common ions (Na+, Cl-, K+, etc.)
            while still removing water.

    Returns:
        A new trajectory with solvent removed.
    """
    selection = "not water" if keep_ions else "not water and not (resname NA CL K MG CA ZN CU FE)"
    atom_indices = traj.topology.select(selection)
    if atom_indices.size == 0:
        raise ValueError("No atoms remain after stripping solvent.")
    return traj.atom_slice(atom_indices)


def extract_chain(traj: md.Trajectory, chain_id: int) -> md.Trajectory:
    """Extract a single chain from a trajectory.

    Args:
        traj: Input trajectory.
        chain_id: Zero-based chain index to extract.

    Returns:
        A new trajectory containing only the specified chain.

    Raises:
        ValueError: If ``chain_id`` is out of range.
    """
    chains = list(traj.topology.chains)
    if not 0 <= chain_id < len(chains):
        raise ValueError(f"chain_id must be in [0, {len(chains) - 1}], got {chain_id}.")

    atom_indices = [atom.index for atom in chains[chain_id].atoms]
    return traj.atom_slice(atom_indices)
