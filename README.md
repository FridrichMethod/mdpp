# mdpp

**Molecular Dynamics Pre- & Post-Processing**

*A Python toolkit for MD simulation workflows — trajectory analysis, cheminformatics, publication-ready plots, system preparation, and GROMACS/OpenFE automation.*

[![Documentation](https://readthedocs.org/projects/mdpp/badge/?version=latest)](https://mdpp.readthedocs.io/en/latest/?badge=latest)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

## Table of Contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Structure](#package-structure)
- [Scripts](#scripts)
- [Design Philosophy](#design-philosophy)
- [Dependencies](#dependencies)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Highlights

- **Trajectory analysis** — RMSD, RMSF, delta-RMSF, DCCM, SASA, radius of gyration, hydrogen bonds, native contacts, pairwise distances, DSSP secondary structure
- **Dimensionality reduction** — PCA (with projection), TICA, backbone torsion featurization, free energy surfaces
- **Conformational clustering** — RMSD distance matrix, GROMOS algorithm
- **Pluggable compute backends** — pairwise distances and RMSD matrix ship with `mdtraj` / `numba` / `torch` / `jax` / `cupy` backends; switch via `backend=` (50x+ speedup on multi-core CPU or GPU, no PBC on non-mdtraj backends)
- **Cheminformatics** — molecular descriptors, PAINS filters, fingerprints (Morgan/ECFP), Tanimoto similarity, Butina clustering
- **Publication-ready plots** — one-liner matplotlib figures with proper axis labels and units
- **2D/3D visualization** — molecule structure drawings (RDKit), interactive 3D views (py3Dmol, nglview)
- **System preparation** — PDB fixing (OpenMM), pKa prediction (PROPKA), ligand parameterization (RDKit), trajectory merge/slice/subsample
- **GROMACS automation** — MDP templates plus analysis, runtime, and post-processing helpers in `scripts/gromacs/`
- **OpenFE automation** — RBFE workflow scripts with SLURM array jobs and checkpoint resumption
- **Typed & tested** — full type annotations, frozen dataclass results, Google docstrings

## Installation

```bash
git clone https://github.com/FridrichMethod/mdpp.git
cd mdpp
pip install -e ".[dev]"
```

<details>
<summary><b>Conda environment with OpenMM support</b></summary>

```bash
conda create -n mdpp python=3.12 -y && conda activate mdpp
conda install -c conda-forge pdbfixer -y
pip install -e ".[openmm,dev]"
```

</details>

<details>
<summary><b>Optional GPU backends (cupy / torch / jax)</b></summary>

```bash
pip install -e ".[gpu]"
```

Enables the `backend="cupy"`, `backend="torch"`, and `backend="jax"` options
on `compute_rmsd_matrix`, `compute_distances`, and `featurize_ca_distances`.
These libraries are optional -- the `numba` and `mdtraj` backends work
without any GPU dependencies.

</details>

## Quick Start

### Load, analyze, and plot in 3 lines

```python
from mdpp.core import load_trajectory
from mdpp.analysis import compute_rmsd
from mdpp.plots import plot_rmsd

traj = load_trajectory("md.xtc", topology_path="topol.gro")
result = compute_rmsd(traj, atom_selection="backbone")
ax = plot_rmsd(result)
```

### Multi-panel figures

```python
import matplotlib.pyplot as plt
from mdpp.analysis import compute_rmsd, compute_rmsf, compute_dccm
from mdpp.plots import plot_rmsd, plot_rmsf, plot_dccm, plot_fes

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plot_rmsd(rmsd_result, ax=axes[0, 0])
plot_rmsf(rmsf_result, ax=axes[0, 1])
plot_dccm(dccm_result, ax=axes[1, 0])
plot_fes(fes_result, ax=axes[1, 1])
fig.tight_layout()
```

### Parse GROMACS output

```python
from mdpp.core import read_xvg, read_edr

df = read_xvg("rmsd.xvg")         # XVG → pandas DataFrame
df = read_edr("ener.edr")          # binary EDR → pandas DataFrame
```

### Prepare a protein

```python
from mdpp.prep import fix_pdb, strip_solvent, run_propka

fix_pdb("raw.pdb", "fixed.pdb", pH=7.4)    # add missing atoms & hydrogens
dry = strip_solvent(traj, keep_ions=True)    # remove water
pka = run_propka("protein.pdb")             # predict titratable residue pKa values
```

### Cheminformatics

```python
from mdpp.chem import MolSupplier, calc_descs, gen_fp, calc_sim, is_pains

for mol in MolSupplier("compounds.sdf"):
    descs = calc_descs(mol)                 # molecular descriptors (MW, LogP, TPSA, ...)
    fp = gen_fp(mol, fp_type="ecfp4")       # ECFP4 fingerprint
    print(f"PAINS: {is_pains(mol)}")        # structural alert filter
```

## Package Structure

```
mdpp
├── core         Trajectory I/O · XVG/EDR parsers · atom selection · alignment
├── analysis     RMSD · RMSF · delta-RMSF · DCCM · SASA · Rg · H-bonds · contacts · DSSP · PCA · TICA · FES · clustering
├── chem         Descriptors · PAINS filters · fingerprints · similarity · molecule file I/O
├── plots        Time series · heatmaps · FES contours · scatter · contact maps · 2D/3D molecules
├── prep         PDB fixing · pKa prediction · ligand topology · trajectory merge/slice/subsample
├── constants    Physical constants (gas constant, default temperature)
└── scripts      Repository shell helpers for GROMACS, OpenFE, and BrownDye
```

## Scripts

```bash
cp scripts/gromacs/mdps/charmm/*.mdp ./sim/
cp scripts/gromacs/mdrun/mdprep.sh ./sim/
cp scripts/gromacs/mdrun/mdrun.sh ./sim/
```

Shell scripts live in `scripts/` and are not installed as part of the Python package.

| Category | Contents |
|---|---|
| `gromacs/mdps` | Force-field-specific MDP templates for AMBER and CHARMM workflows |
| `gromacs/analysis` | RMSD, RMSF, DSSP, H-bonds, energy, Rg, SASA, clustering |
| `gromacs/runtime` | Job status monitor, restart, extend, export |
| `gromacs/compilation` | GROMACS build scripts (generic + Sherlock HPC) |
| `gromacs/mdenv` | Environment setup (Sherlock module loads) |
| `gromacs/data_transfer` | DTN download scripts (Sherlock) |
| `gromacs/postprocessing` | Trajectory postprocessing |
| `gromacs/visualization` | PyMOL movie generation |
| `openfe/quickrun` | RBFE SLURM submission (`quickrun.sh`, `quickrun.sbatch`) |
| `openfe/runtime` | Status checking (`check_status.sh`) and periodic monitoring (`monitor.sbatch`) |

## Design Philosophy

Every analysis function follows the same pattern:

```python
result = compute_something(traj, *, keyword_args...)   # → frozen dataclass
ax = plot_something(result, *, ax=None)                 # → matplotlib Axes
```

- **Input**: `md.Trajectory` (from MDTraj) or a feature matrix
- **Output**: frozen `@dataclass` with unit-conversion properties (`.time_ns`, `.rmsd_angstrom`, etc.)
- **Plotting**: pass the result dataclass directly — labels and units are set automatically

## Dependencies

Built on the scientific Python ecosystem:

| Library | Role |
|---|---|
| [MDTraj](https://mdtraj.org) | Trajectory loading & geometry |
| [MDAnalysis](https://mdanalysis.org) | XVG auxiliary reader |
| [panedr](https://github.com/MDAnalysis/panedr) | GROMACS EDR parsing |
| [scikit-learn](https://scikit-learn.org) | PCA, clustering |
| [deeptime](https://deeptime-ml.github.io) | TICA |
| [RDKit](https://rdkit.org) | Cheminformatics & ligand topology |
| [Numba](https://numba.pydata.org) | Parallel CPU kernels: RMSD matrix (QCP), pairwise distances, similarity |
| [PyTorch](https://pytorch.org) · [JAX](https://jax.readthedocs.io) · [CuPy](https://cupy.dev) | Optional GPU backends (install via `[gpu]` extra) |
| [PROPKA](https://github.com/jensengroup/propka) | pKa prediction |
| [BioPython](https://biopython.org) | PDB chain extraction |
| [matplotlib](https://matplotlib.org) | Static 2D plotting |
| [py3Dmol](https://3dmol.csb.pitt.edu) | Interactive 3D molecule views |
| [nglview](https://nglviewer.org) | Interactive 3D trajectory views |

## Documentation

Full documentation at **[mdpp.readthedocs.io](https://mdpp.readthedocs.io)**.

Build locally:

```bash
pip install -e ".[docs]"
mkdocs serve                  # http://127.0.0.1:8000
```

## Contributing

```bash
# Lint & format
ruff check src/ tests/ --fix
ruff format src/ tests/

# Type check
mypy src/mdpp/

# Run tests
pytest

# Full pre-commit suite
pre-commit run --all-files
```

## License

[MIT](LICENSE) — Zhaoyang Li
