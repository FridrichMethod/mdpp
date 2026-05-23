"""APBS preparation for biotinol-AMP complexes 2ewn.pdb and 29xd.pdb.

Workflow per PDB (no BrownDye prep):

1. Split chain A into protein-only and BTX-only PDBs.
2. PropKa check + PDBFixer hydrogen placement on the protein.
3. Assign ligand bond orders from the biotinol-AMP SMILES; write SDF.
4. AmberTools parameterisation (pdb4amber -> obabel -> antechamber ->
   parmchk2 -> tleap), ParmEd PQR export for protein, ligand, and complex.
5. APBS Poisson-Boltzmann on the complex PQR -> ``complex.dx``.

Outputs land in ``tmp/<stem>/complex/{prep,ambertools,apbs}/`` next to this
script.

Run from the ambertools conda environment (which also has mdpp installed)::

    conda run -n ambertools python apbs_btx_complex.py
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import parmed as pmd
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Residue import Residue
from rdkit import Chem

from mdpp.prep import assign_topology, fix_pdb, run_propka, write_apbs_input

# Ligand: biotinol-AMP (alcohol linker, not the ester used in complex.pdb).
LIGAND_SMILES = (
    r"Nc1ncnc2n(cnc12)[C@@H]3O[C@H](CO[P]([O-])(=O)OCCCCC[C@@H]4SC[C@@H]5"
    r"NC(=O)N[C@H]45)[C@@H](O)[C@H]3O"
)

# 2ewn.pdb and 29xd.pdb both keep protein + ligand in chain A,
# with a single BTX HETATM.
PROTEIN_CHAIN_ID = "A"
LIGAND_RESNAME = "BTX"
PH = 7.4

# AmberTools.
PROTEIN_FF = "leaprc.protein.ff19SB"
LIGAND_FF = "leaprc.gaff2"
PB_RADII = "mbondi3"
STRIP_PROTEIN_H = True


class _ProteinAtomSelect(Select):
    """Keep only standard (non-hetero) residues from a chosen chain."""

    def __init__(self, chain_id: str) -> None:
        self.chain_id = chain_id

    def accept_chain(self, chain) -> int:  # type: ignore[override]
        return int(chain.id == self.chain_id)

    def accept_residue(self, residue: Residue) -> int:  # type: ignore[override]
        hetflag = residue.id[0].strip()
        return int(hetflag == "")


class _LigandResnameSelect(Select):
    """Keep only residues whose resname matches ``resname`` (e.g. BTX)."""

    def __init__(self, resname: str) -> None:
        self.resname = resname

    def accept_residue(self, residue: Residue) -> int:  # type: ignore[override]
        return int(residue.get_resname().strip() == self.resname)


def _run(cmd: Iterable[str], cwd: Path, log_path: Path | None = None) -> None:
    """Run ``cmd`` from ``cwd``; if ``log_path`` given, tee stdout+stderr there."""
    cmd_list = [str(c) for c in cmd]
    print("$ (cd", cwd, "&&", " ".join(cmd_list), ")")
    if log_path is None:
        subprocess.run(cmd_list, cwd=str(cwd), check=True)
        return
    result = subprocess.run(
        cmd_list,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd_list)


_PML_TEMPLATE = """# Visualize the full complex APBS map in PyMOL.
#
# Run from this directory:
#   pymol viz_complex_apbs.pml

load complex/ambertools/complex.pqr, complex
load complex/apbs/complex.dx, complex_potential

show cartoon, complex
show sticks, organic
show surface, complex

ramp_new complex_esp, complex_potential, [-5, 0, 5], [red, white, blue]
set surface_color, complex_esp, complex

isomesh complex_pos_mesh, complex_potential, 1.0
isomesh complex_neg_mesh, complex_potential, -1.0
color blue, complex_pos_mesh
color red, complex_neg_mesh

orient complex
zoom complex, 8
"""


def write_viz_pml(target_dir: Path) -> Path:
    """Write the PyMOL APBS visualisation script into ``target_dir``."""
    pml_path = target_dir / "viz_complex_apbs.pml"
    pml_path.write_text(_PML_TEMPLATE, encoding="utf-8")
    return pml_path


def _stage_apbs_dx(apbs_intermediate: Path, stem: str) -> Path:
    """Rename whichever ``.dx`` variant apbs produced to ``{stem}.dx``."""
    for src_name in (f"{stem}-PE0.dx", f"{stem}.pqr.dx", f"{stem}.dx"):
        src = apbs_intermediate / src_name
        if src.is_file() and src.stat().st_size > 0:
            target = apbs_intermediate / f"{stem}.dx"
            if src != target:
                src.rename(target)
            return target
    raise RuntimeError(f"No {stem}.dx produced in {apbs_intermediate}")


def process_pdb(input_pdb: Path, out_root: Path) -> None:
    """Run the prep + APBS workflow for one biotinol-AMP complex PDB."""
    stem = input_pdb.stem
    print(f"\n{'=' * 60}\nProcessing {stem}\n{'=' * 60}")

    base = out_root / stem / "complex"
    prep_dir = base / "prep"
    prep_intermediate = prep_dir / "intermediate"
    amber_dir = base / "ambertools"
    amber_intermediate = amber_dir / "intermediate"
    apbs_dir = base / "apbs"
    apbs_intermediate = apbs_dir / "intermediate"
    for path in (
        prep_dir,
        prep_intermediate,
        amber_dir,
        amber_intermediate,
        apbs_dir,
        apbs_intermediate,
    ):
        path.mkdir(parents=True, exist_ok=True)

    # ---- 1. Split protein and ligand ----
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(stem, str(input_pdb))
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)

    protein_pdb = prep_intermediate / "protein.pdb"
    pdb_io.save(str(protein_pdb), _ProteinAtomSelect(PROTEIN_CHAIN_ID))
    ligand_pdb = prep_intermediate / "ligand.pdb"
    pdb_io.save(str(ligand_pdb), _LigandResnameSelect(LIGAND_RESNAME))

    propka_result = run_propka(protein_pdb)
    nonstandard = propka_result.get_nonstandard(PH)
    propka_report = prep_dir / "protein_propka.tsv"
    with propka_report.open("w") as handle:
        handle.write(
            "residue_type\tres_num\tchain_id\tpka\tmodel_pka\tpropka_protonated\tmodel_protonated\n"
        )
        for residue in propka_result.residues:
            handle.write(
                f"{residue.residue_type}\t{residue.res_num}\t{residue.chain_id}\t"
                f"{residue.pka:.3f}\t{residue.model_pka:.3f}\t"
                f"{residue.is_protonated_at(PH)}\t{residue.is_default_protonated_at(PH)}\n"
            )
    print(f"PropKa report -> {propka_report}")
    if nonstandard:
        print(f"PropKa flags {len(nonstandard)} non-standard state(s) at pH {PH}:")
        for residue in nonstandard:
            print(f"  {residue.label}: pKa={residue.pka:.2f}, model={residue.model_pka:.2f}")
    else:
        print(f"PropKa agrees with model-pKa defaults at pH {PH}.")

    protein_fixed_pdb = prep_dir / "protein_fixed.pdb"
    fix_pdb(protein_pdb, protein_fixed_pdb, pH=PH)
    print(f"Fixed protein -> {protein_fixed_pdb}")

    # ---- 2. Assign ligand topology from SMILES ----
    template_mol = Chem.MolFromSmiles(LIGAND_SMILES, sanitize=True)
    if template_mol is None:
        raise RuntimeError("Failed to parse LIGAND_SMILES")
    ligand_net_charge = Chem.GetFormalCharge(template_mol)
    print(f"Ligand net charge: {ligand_net_charge}")

    mol = Chem.MolFromPDBFile(str(ligand_pdb), sanitize=True, removeHs=True)
    if mol is None:
        raise RuntimeError(f"RDKit could not parse {ligand_pdb}")
    mol_assigned = assign_topology(mol, template_mol)
    mol_assigned.SetProp("_Name", LIGAND_RESNAME)

    ligand_sdf = prep_dir / "ligand.sdf"
    with Chem.SDWriter(str(ligand_sdf)) as writer:
        writer.write(mol_assigned)
    (prep_dir / "ligand_charge.txt").write_text(f"{ligand_net_charge}\n")
    (prep_dir / "ligand_resname.txt").write_text(f"{LIGAND_RESNAME}\n")
    print(f"Ligand SDF -> {ligand_sdf}")

    # ---- 3. AmberTools ----
    shutil.copy(protein_fixed_pdb, amber_intermediate / "protein_fixed.pdb")
    shutil.copy(ligand_sdf, amber_intermediate / "ligand.sdf")

    pdb4amber_args = [
        "pdb4amber",
        "-i",
        "protein_fixed.pdb",
        "-o",
        "protein_amber.pdb",
        "-d",
        "--no-conect",
    ]
    if STRIP_PROTEIN_H:
        pdb4amber_args.append("-y")
    _run(pdb4amber_args, cwd=amber_intermediate)

    _run(
        ["obabel", "ligand.sdf", "-O", "ligand_seed.mol2"],
        cwd=amber_intermediate,
    )

    ligand_seed_mol2 = amber_intermediate / "ligand_seed.mol2"
    text = ligand_seed_mol2.read_text()
    for old in ("UNL1", "UNL", "UNK"):
        text = text.replace(old, LIGAND_RESNAME)
    if LIGAND_RESNAME not in text:
        raise RuntimeError(
            f"Residue name {LIGAND_RESNAME!r} not found in {ligand_seed_mol2} after patch"
        )
    ligand_seed_mol2.write_text(text)

    _run(
        [
            "antechamber",
            "-i",
            "ligand_seed.mol2",
            "-fi",
            "mol2",
            "-o",
            "ligand_amber.mol2",
            "-fo",
            "mol2",
            "-c",
            "bcc",
            "-s",
            "2",
            "-at",
            "gaff2",
            "-nc",
            str(ligand_net_charge),
            "-rn",
            LIGAND_RESNAME,
        ],
        cwd=amber_intermediate,
    )
    _run(
        [
            "parmchk2",
            "-i",
            "ligand_amber.mol2",
            "-f",
            "mol2",
            "-o",
            "ligand.frcmod",
        ],
        cwd=amber_intermediate,
    )

    tleap_in = amber_intermediate / "tleap.in"
    tleap_in.write_text(
        f"""source {PROTEIN_FF}
source {LIGAND_FF}

{LIGAND_RESNAME} = loadmol2 ligand_amber.mol2
loadamberparams ligand.frcmod
protein = loadpdb protein_amber.pdb
complex = combine {{protein {LIGAND_RESNAME}}}

set default PBRadii {PB_RADII}
saveamberparm protein protein.prmtop protein.rst7
saveamberparm {LIGAND_RESNAME} ligand.prmtop ligand.rst7
saveamberparm complex complex.prmtop complex.rst7
quit
"""
    )
    _run(["tleap", "-f", "tleap.in"], cwd=amber_intermediate)

    for body in ("protein", "ligand", "complex"):
        parm = pmd.load_file(
            str(amber_intermediate / f"{body}.prmtop"),
            xyz=str(amber_intermediate / f"{body}.rst7"),
        )
        parm.save(str(amber_intermediate / f"{body}.pqr"), overwrite=True)
        for suffix in ("prmtop", "rst7", "pqr"):
            shutil.copy(
                amber_intermediate / f"{body}.{suffix}",
                amber_dir / f"{body}.{suffix}",
            )

    # ---- 4. APBS on the complex PQR ----
    shutil.copy(amber_dir / "complex.pqr", apbs_intermediate / "complex.pqr")
    apbs_in = write_apbs_input("complex", apbs_intermediate)
    print(f"APBS input -> {apbs_in}")

    apbs_log = apbs_intermediate / "complex.apbs.log"
    _run(["apbs", "complex.in"], cwd=apbs_intermediate, log_path=apbs_log)
    dx_path = _stage_apbs_dx(apbs_intermediate, "complex")

    shutil.copy(apbs_intermediate / "complex.in", apbs_dir / "complex.in")
    shutil.copy(apbs_log, apbs_dir / "complex.apbs.log")
    shutil.copy(dx_path, apbs_dir / "complex.dx")
    shutil.copy(apbs_intermediate / "complex.pqr", apbs_dir / "complex.pqr")

    pml_path = write_viz_pml(base.parent)
    print(f"Complex DX  -> {apbs_dir / 'complex.dx'}")
    print(f"Complex PQR -> {apbs_dir / 'complex.pqr'}")
    print(f"Viz script  -> {pml_path}")


def main() -> None:
    """Process 2ewn.pdb and 29xd.pdb in the example directory."""
    example_dir = Path(__file__).resolve().parent
    out_root = example_dir / "tmp"
    for pdb_name in ("2ewn.pdb", "29xd.pdb"):
        process_pdb(example_dir / pdb_name, out_root)


if __name__ == "__main__":
    main()
