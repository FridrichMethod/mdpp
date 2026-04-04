# Getting Started

## Installation

mdpp requires Python 3.12+ and is installed from source:

```bash
git clone https://github.com/FridrichMethod/mdpp.git
cd mdpp
pip install -e ".[dev]"
```

For conda environments with OpenMM support:

```bash
conda install -c conda-forge pdbfixer
pip install -e ".[openmm,dev]"
```

## Minimal Example

Load a GROMACS trajectory, compute RMSD, and plot it:

```python
from mdpp.core import load_trajectory
from mdpp.analysis.metrics import compute_rmsd
from mdpp.plots import plot_rmsd

traj = load_trajectory("md.xtc", topology_path="topol.gro")
result = compute_rmsd(traj, atom_selection="backbone")
ax = plot_rmsd(result)
```

## Parse GROMACS Output

Read an XVG file produced by `gmx energy`:

```python
from mdpp.core import read_xvg

df = read_xvg("energy.xvg")
print(df.columns)  # ["Time", "Potential", "Kinetic En.", ...]
```

Read a binary EDR file:

```python
from mdpp.core import read_edr

df = read_edr("ener.edr")
df.plot(x="Time", y=["Potential", "Temperature"])
```

## Prepare a Protein

Fix a PDB file (add missing atoms, hydrogens):

```python
from mdpp.prep import fix_pdb

fix_pdb("raw.pdb", "fixed.pdb", pH=7.4)
```

## Cheminformatics

Calculate molecular descriptors and check for PAINS alerts:

```python
from rdkit import Chem
from mdpp.chem import calc_descs, is_pains, gen_fp, calc_sim

mol = Chem.MolFromSmiles("c1ccc(NC(=O)c2ccccc2)cc1")
descs = calc_descs(mol)  # (MW, LogP, HBA, HBD, Fsp3, RotBonds, Rings, TPSA, QED)
print(f"PAINS: {is_pains(mol)}")

fp = gen_fp(mol, fp_type="ecfp4")
```

## MDP Templates

The repository includes GROMACS MDP templates under `scripts/gromacs/mdps/`.
Copy the force-field-specific files you need into your simulation directory:

```bash
cp scripts/gromacs/mdps/charmm/*.mdp ./my_simulation/
```

Utility shell scripts (analysis wrappers, runtime helpers, etc.) live in the top-level `scripts/` directory and can be copied to your working directory as needed.

## Next Steps

- [User Guide: Core](guide/core.md) -- trajectory loading and file parsing
- [User Guide: Analysis](guide/analysis.md) -- structural and dynamic analysis
- [User Guide: Cheminformatics](guide/chem.md) -- descriptors, fingerprints, similarity
- [User Guide: Plots](guide/plots.md) -- visualization
- [User Guide: Preparation](guide/prep.md) -- system setup
- [User Guide: Scripts](guide/scripts.md) -- repository scripts and MDP templates
