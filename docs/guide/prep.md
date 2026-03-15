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
