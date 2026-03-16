<div align="center">

# mdpp

**Molecular Dynamics Pre- & Post-Processing**

*A Python toolkit for MD simulation workflows — trajectory analysis, publication-ready plots, system preparation, and GROMACS automation.*

[![Documentation](https://readthedocs.org/projects/mdpp/badge/?version=latest)](https://mdpp.readthedocs.io/en/latest/?badge=latest)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

[Documentation](https://mdpp.readthedocs.io) · [Getting Started](https://mdpp.readthedocs.io/en/latest/getting-started/) · [API Reference](https://mdpp.readthedocs.io/en/latest/api/core/)

</div>

______________________________________________________________________

## Highlights

- **Trajectory analysis** — RMSD, RMSF, DCCM, SASA, radius of gyration, hydrogen bonds, native contacts, pairwise distances, DSSP secondary structure
- **Dimensionality reduction** — PCA, TICA, backbone torsion featurization, free energy surfaces
- **Conformational clustering** — RMSD distance matrix, GROMOS algorithm
- **Publication-ready plots** — one-liner matplotlib figures with proper axis labels and units
- **System preparation** — PDB fixing (OpenMM), ligand parameterization (RDKit), trajectory merge/slice/subsample
- **GROMACS automation** — bundled MDP templates, analysis wrappers, runtime monitoring scripts, and a CLI to manage them all
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
from mdpp.prep import fix_pdb, strip_solvent

fix_pdb("raw.pdb", "fixed.pdb", pH=7.4)    # add missing atoms & hydrogens
dry = strip_solvent(traj, keep_ions=True)    # remove water
```

## Package Structure

```
mdpp
├── core         Trajectory I/O · XVG/EDR parsers · atom selection · alignment
├── analysis     RMSD · RMSF · DCCM · SASA · Rg · H-bonds · contacts · DSSP · PCA · TICA · FES · clustering
├── plots        Time series · heatmaps · FES contours · scatter · Ramachandran · contact maps
├── prep         PDB fixing · ligand topology · trajectory merge/slice/subsample
├── data         GROMACS MDP parameter templates
└── scripts      Analysis wrappers · runtime tools · compilation helpers
```

## CLI

```bash
mdpp list                               # list bundled utility scripts
mdpp list gromacs/analysis              # filter by category
mdpp show gromacs/runtime/restart.sh    # view a script
mdpp copy gromacs/analysis ./sim/       # copy scripts to working dir
mdpp mdps ./sim/                        # copy MDP templates
```

<details>
<summary><b>Available script categories</b></summary>

| Category | Contents |
|---|---|
| `gromacs/analysis` | RMSD, RMSF, DSSP, H-bonds, energy, Rg, SASA, clustering |
| `gromacs/runtime` | Job status monitor, restart, extend, export |
| `gromacs/compilation` | GROMACS build scripts (generic + Sherlock HPC) |
| `gromacs/mdenv` | Environment setup (Sherlock module loads) |
| `gromacs/postprocessing` | Trajectory postprocessing |
| `gromacs/visualization` | PyMOL movie generation |

</details>

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
| [RDKit](https://rdkit.org) | Ligand topology |
| [matplotlib](https://matplotlib.org) | Plotting |

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
