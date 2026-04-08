"""Protein structure preparation and manipulation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mdtraj as md
from Bio.PDB import Select

from mdpp._types import StrPath

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PropkaResidue:
    """PROPKA pKa prediction for a single titratable residue.

    Attributes:
        residue_type: Group label (e.g. ``ASP``, ``HIS``, ``N+``, ``C-``).
        res_num: Residue sequence number.
        chain_id: PDB chain identifier.
        pka: PROPKA-predicted pKa value.
        model_pka: Reference model pKa value.
    """

    residue_type: str
    res_num: int
    chain_id: str
    pka: float
    model_pka: float

    @property
    def label(self) -> str:
        """Formatted residue label matching PROPKA output style."""
        return f"{self.residue_type:>3s} {self.res_num:>4d} {self.chain_id}"

    def is_protonated_at(self, pH: float) -> bool:
        """Whether PROPKA predicts the residue to be protonated at the given pH."""
        return self.pka > pH

    def is_default_protonated_at(self, pH: float) -> bool:
        """Whether the model pKa predicts the residue to be protonated at the given pH."""
        return self.model_pka > pH


@dataclass(frozen=True, slots=True)
class PropkaResult:
    """PROPKA pKa prediction results for all titratable residues.

    Attributes:
        residues: pKa predictions for each titratable residue.
    """

    residues: tuple[PropkaResidue, ...]

    def get_nonstandard(self, pH: float) -> tuple[PropkaResidue, ...]:
        """Return residues where PROPKA and model pKa disagree on protonation state.

        A residue is "non-standard" when ``pKa > pH`` and ``model_pKa <= pH``
        (or vice versa), meaning PDBFixer would assign a different protonation
        state than what PROPKA predicts.

        Args:
            pH: pH value for protonation state comparison.

        Returns:
            Residues with non-standard predicted protonation.
        """
        return tuple(
            r for r in self.residues if r.is_protonated_at(pH) != r.is_default_protonated_at(pH)
        )


def run_propka(pdb_path: StrPath) -> PropkaResult:
    """Run PROPKA to predict pKa values for titratable protein residues.

    Args:
        pdb_path: Path to the input PDB file.

    Returns:
        pKa predictions for all titratable residues found.
    """
    import propka.run

    mol = propka.run.single(str(pdb_path), write_pka=False)
    conf = next(iter(mol.conformations.values()))

    residues: list[PropkaResidue] = []
    for group in conf.get_titratable_groups():
        if not group.model_pka_set:
            continue
        residues.append(
            PropkaResidue(
                residue_type=group.residue_type.strip(),
                res_num=group.atom.res_num,
                chain_id=group.atom.chain_id.strip(),
                pka=group.pka_value,
                model_pka=group.model_pka,
            )
        )

    return PropkaResult(residues=tuple(residues))


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

    Runs PROPKA to check for residues whose environment-shifted pKa predicts
    a different protonation state than the model-pKa default used by PDBFixer,
    and logs a warning for each such residue.

    Args:
        pdb_path: Path to the input PDB file.
        fixed_pdb_path: Path where the fixed PDB will be written.
        pH: pH value for hydrogen placement.
    """
    result = run_propka(pdb_path)
    nonstandard = result.get_nonstandard(pH)
    if nonstandard:
        lines = "\n".join(
            f"  {r.label}: pKa={r.pka:5.2f} (model={r.model_pka:5.2f})"
            f" -> PROPKA: {'protonated' if r.is_protonated_at(pH) else 'deprotonated'}"
            f", PDBFixer: {'protonated' if r.is_default_protonated_at(pH) else 'deprotonated'}"
            for r in nonstandard
        )
        logger.warning(
            "PROPKA predicts non-standard protonation for %d residue(s) at pH %.1f.\n"
            "PDBFixer uses model pKa and will assign different protonation:\n%s",
            len(nonstandard),
            pH,
            lines,
        )

    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

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
