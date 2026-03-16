# mdpp

**MD simulation pre- and post-processing.**

mdpp provides Python utilities and shell scripts for molecular dynamics simulation workflows: trajectory loading, structural/dynamic analysis, visualization, system preparation, and GROMACS workflow automation.

## Installation

Requires Python 3.12+.

```bash
pip install -e ".[dev]"
```

For OpenMM-based tools:

```bash
conda install -c conda-forge pdbfixer
pip install -e ".[openmm]"
```

## Quick Start

```python
from mdpp.core import load_trajectory
from mdpp.analysis import compute_rmsd
from mdpp.plots import plot_rmsd

traj = load_trajectory("md.xtc", topology_path="topol.gro")
result = compute_rmsd(traj, atom_selection="backbone")
ax = plot_rmsd(result)
```

## Package Structure

```
mdpp
├── core         Trajectory I/O, XVG/EDR parsers
├── analysis     RMSD, RMSF, DCCM, SASA, contacts, DSSP, FES, PCA/TICA, clustering
├── plots        Publication-ready matplotlib helpers
├── prep         Protein fixing, ligand parameterization, trajectory tools
├── data         MDP parameter templates (bundled)
└── scripts      GROMACS utility scripts (bundled)
```

## CLI

mdpp includes a command-line interface for managing bundled scripts and MDP templates:

```bash
mdpp list                           # list available utility scripts
mdpp list gromacs/analysis          # filter by category
mdpp show gromacs/runtime/restart.sh  # print script to stdout
mdpp copy gromacs/analysis ./sim/   # copy scripts to working dir
mdpp mdps ./sim/                    # copy MDP templates
```

## Documentation

Full documentation: [mdpp.readthedocs.io](https://mdpp.readthedocs.io)

```bash
pip install -e ".[docs]"
mkdocs serve          # local preview at http://127.0.0.1:8000
```

## License

See [LICENSE](LICENSE).
