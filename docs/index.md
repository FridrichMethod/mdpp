# mdpp

**MD simulation pre- and post-processing.**

mdpp provides a collection of Python utilities for molecular dynamics simulation workflows, covering trajectory loading, structural analysis, visualization, and system preparation. It is designed to work with GROMACS, AMBER, OpenFE, and other MD engines.

## Package Structure

```
mdpp
├── core        # Trajectory I/O, XVG/EDR parsers
├── analysis    # RMSD, RMSF, DCCM, SASA, contacts, DSSP, FES, PCA/TICA, clustering
├── plots       # Publication-ready plotting helpers
├── prep        # Protein fixing, ligand parameterization, trajectory manipulation
└── scripts     # Repository shell scripts and GROMACS MDP templates
```

## Quick Install

```bash
pip install -e ".[dev]"
```

For OpenMM-based preparation tools:

```bash
pip install -e ".[openmm]"
```

## Links

- [Getting Started](getting-started.md)
- [User Guide](guide/core.md)
- [API Reference](api/core.md)
- [Scripts](guide/scripts.md)
