# Preparation

The `mdpp.prep` subpackage provides utilities for system preparation: fixing protein structures, parameterizing ligands, and manipulating trajectories.

## Protein Structure Preparation

### Fix a PDB file

Add missing residues, atoms, and hydrogens:

```python
from mdpp.prep import fix_pdb

fix_pdb("raw.pdb", "fixed.pdb", pH=7.4)
```

This uses OpenMM's PDBFixer under the hood. Requires the `openmm` optional dependency:

```bash
pip install -e ".[openmm]"
```

### Strip solvent

Remove water and ions from a trajectory:

```python
from mdpp.prep import strip_solvent

dry_traj = strip_solvent(traj)
```

Keep ions while removing water:

```python
dry_traj = strip_solvent(traj, keep_ions=True)
```

### Extract a chain

Extract a single chain by zero-based index:

```python
from mdpp.prep import extract_chain

chain_a = extract_chain(traj, chain_id=0)
```

The `ChainSelect` helper (a `Bio.PDB.Select` subclass) is also available for advanced PDB chain filtering with BioPython.

### Predict pKa values (PROPKA)

Predict titratable residue pKa values:

```python
from mdpp.prep import run_propka

result = run_propka("protein.pdb")
for residue in result.residues:
    print(f"{residue.residue_type} {residue.res_num}{residue.chain_id}: pKa={residue.pka:.2f}")
```

The `PropkaResult` and `PropkaResidue` frozen dataclasses hold the prediction output.

## Ligand Parameterization

### Assign bond orders from a template

When loading a ligand from a PDB file (which lacks bond order information), use a SMILES-derived template to assign correct bond orders:

```python
from rdkit import Chem
from mdpp.prep import assign_topology, constraint_minimization

mol = Chem.MolFromPDBFile("ligand.pdb", removeHs=False)
template = Chem.MolFromSmiles("c1ccccc1")

fixed = assign_topology(mol, template)
minimized = constraint_minimization(fixed)
```

## Trajectory Manipulation

### Merge trajectories

Concatenate multiple runs along the time axis:

```python
from mdpp.core import load_trajectories
from mdpp.prep import merge_trajectories

trajs = load_trajectories(["run1.xtc", "run2.xtc"], topology_paths=["topol.gro"] * 2)
combined = merge_trajectories(trajs)
```

### Slice and subsample

```python
from mdpp.prep import slice_trajectory, subsample_trajectory

# Take frames 100-500 with stride 2
sliced = slice_trajectory(traj, start=100, stop=500, stride=2)

# Evenly subsample to 200 frames
subsampled = subsample_trajectory(traj, n_frames=200)
```

## APBS input generation

Generate an APBS multigrid input file from an existing PQR. Grid dimensions
are derived from the radius-inflated atom bounding box; `dime` is rounded up
to the nearest `c * 2**n + 1` value APBS multigrid requires. Physics defaults
mirror `pdb2pqr --apbs-input` with explicit Na+/Cl- ion lines (so the Debye
length is finite, required by downstream BrownDye simulations).

```python
from mdpp.prep import write_apbs_input

# Writes work_dir/complex.in for work_dir/complex.pqr.
write_apbs_input("complex", work_dir)

# Override physics defaults per call:
write_apbs_input(
    "complex",
    work_dir,
    ionic_strength_m=0.050,
    solute_dielectric=4.0,
    fine_spacing_a=0.5,
)
```

After running APBS itself (`apbs complex.in`), parse the Debye length out of
the log for use in any downstream tool that needs it:

```python
from mdpp.prep import infer_debye_length

debye = infer_debye_length(work_dir / "complex.apbs.log")
```

## BrownDye2 XML generation

Build the two XML inputs consumed by BrownDye2 from Python. These helpers
produce text files only; run BrownDye's own CLI tools (`pqr2xml`,
`make_rxn_pairs`, `make_rxn_file`, `bd_top`, `nam_simulation`) separately.

### Contact types

`contact_types.xml` lists every unique heavy-atom `(atom_name, residue_name)`
pair per body, consumed by `make_rxn_pairs`:

```python
from mdpp.prep import write_contact_types

write_contact_types(
    "complex.pqr",
    "substrate.pqr",
    "contact_types.xml",
)
```

### Top-level input.xml

`input.xml` is the top-level BrownDye configuration consumed by `bd_top`.
Configure two bodies plus a shared solvent block and pass the rest as
keyword arguments:

```python
from mdpp.prep import (
    BrownDyeBody,
    BrownDyeSolvent,
    infer_debye_length,
    write_input_xml,
)

debye = infer_debye_length("complex.apbs.log", "substrate.apbs.log")

write_input_xml(
    "input.xml",
    BrownDyeBody(name="complex", atoms_xml="complex_atoms.xml", grid_dx="complex.dx"),
    BrownDyeBody(name="substrate", atoms_xml="substrate_atoms.xml", grid_dx="substrate.dx"),
    solvent=BrownDyeSolvent(debye_length_a=debye),
    n_trajectories=100_000,
    seed=11111111,
)
```

See `examples/browndye/complex_pqr.ipynb` for a full pipeline that
chains APBS, AmberTools, PDB2PQR, and BrownDye2.
