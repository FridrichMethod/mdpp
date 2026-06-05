"""Protein structure preparation and manipulation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mdtraj as md
from Bio.PDB import Select

from mdpp._types import StrPath

if TYPE_CHECKING:
    from openmm.app import Topology

logger = logging.getLogger(__name__)

# OpenMM Modeller.addHydrogens variant names used to apply PROPKA-predicted
# protonation. A residue is overridden only where PROPKA disagrees with the
# model-pKa default (PropkaResult.get_nonstandard); every other residue keeps
# OpenMM's default pH-based selection. Residue types not listed here (e.g.
# termini "N+"/"C-") are left to OpenMM and logged.
_PROTONATED_VARIANT: dict[str, str] = {"ASP": "ASH", "GLU": "GLH", "HIS": "HIP"}
_DEPROTONATED_VARIANT: dict[str, str] = {"LYS": "LYN", "CYS": "CYX", "HIS": "HIE"}


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


def _propka_variants(
    topology: Topology,
    nonstandard: tuple[PropkaResidue, ...],
    pH: float,
) -> list[str | None]:
    """Build a per-residue protonation variant list for ``Modeller.addHydrogens``.

    Only residues in ``nonstandard`` (where PROPKA disagrees with the model-pKa
    default) are overridden to PROPKA's predicted state; every other residue
    maps to ``None`` so OpenMM's default pH-based rule applies. Residues whose
    type has no supported variant (e.g. termini ``N+``/``C-``) are skipped and
    logged.

    Args:
        topology: OpenMM topology the variant list must align with (one entry
            per residue, in topology order).
        nonstandard: Residues whose PROPKA-predicted protonation differs from
            the model-pKa default.
        pH: pH at which protonation states are evaluated.

    Returns:
        One entry per topology residue: the OpenMM variant name to force, or
        ``None`` to keep OpenMM's default selection.
    """
    overrides: dict[tuple[str, str, str], str] = {}
    labels: dict[tuple[str, str, str], str] = {}
    for residue in nonstandard:
        table = _PROTONATED_VARIANT if residue.is_protonated_at(pH) else _DEPROTONATED_VARIANT
        variant = table.get(residue.residue_type)
        if variant is None:
            logger.warning(
                "PROPKA protonation for %s has no OpenMM variant; keeping default.",
                residue.label,
            )
            continue
        key = (residue.chain_id, str(residue.res_num), residue.residue_type)
        overrides[key] = variant
        labels[key] = residue.label

    matched: set[tuple[str, str, str]] = set()
    variants: list[str | None] = []
    for res in topology.residues():
        key = (res.chain.id, res.id, res.name)
        if overrides.get(key) is not None:
            matched.add(key)
        variants.append(overrides.get(key))

    for key in matched:
        logger.info("Applying PROPKA protonation %s -> %s", labels[key], overrides[key])
    unmatched = set(overrides) - matched
    if unmatched:
        logger.warning(
            "%d PROPKA override(s) matched no topology residue (chain/residue id "
            "mismatch); protonation NOT applied: %s",
            len(unmatched),
            sorted(labels[key] for key in unmatched),
        )
    return variants


def fix_pdb(
    pdb_path: StrPath,
    fixed_pdb_path: StrPath,
    pH: float = 7.0,
    *,
    protonation: Literal["model", "propka"] = "model",
) -> None:
    """Fix a PDB file by adding missing residues, atoms, and hydrogens.

    Removes all heterogens including water, identifies missing residues and
    atoms, then adds them back along with hydrogens at the specified pH.

    Runs PROPKA to check for residues whose environment-shifted pKa predicts
    a different protonation state than the model-pKa default used by PDBFixer,
    and logs a warning for each such residue.

    Args:
        pdb_path: Path to the input PDB file.
        fixed_pdb_path: Path where the fixed PDB will be written.
        pH: pH value for hydrogen placement.
        protonation: Protonation policy. ``"model"`` (default) uses PDBFixer's
            built-in model pKa values. ``"propka"`` keeps the model default for
            most residues but overrides the residues where PROPKA disagrees
            (``PropkaResult.get_nonstandard``) with PROPKA's predicted state,
            applied via OpenMM ``Modeller`` variants. Supported overrides are
            ASP/GLU/LYS/HIS/CYS (a neutral histidine uses the HIE tautomer);
            unsupported residue types (e.g. termini) keep the default and are
            logged.
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

    from openmm.app import Modeller, PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    if protonation == "propka" and nonstandard:
        # Apply PROPKA-predicted states for the disagreeing residues; OpenMM's
        # default pH rule still handles every other residue (None variant).
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(pH=pH, variants=_propka_variants(modeller.topology, nonstandard, pH))
        topology, positions = modeller.topology, modeller.positions
    else:
        fixer.addMissingHydrogens(pH=pH)
        topology, positions = fixer.topology, fixer.positions

    with Path(fixed_pdb_path).open("w") as f:
        PDBFile.writeFile(topology, positions, f)


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
