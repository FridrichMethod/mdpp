# mdpp

**MD simulation pre- and post-processing.**

mdpp provides a collection of Python utilities for molecular dynamics simulation workflows, covering trajectory loading, structural analysis, visualization, and system preparation. It is designed to work with GROMACS, AMBER, OpenFE, and other MD engines.

## Highlights

- **Trajectory analysis** -- RMSD, RMSF, delta-RMSF (with SEM), DCCM, SASA, Rg, H-bonds, contacts (including native Q(t)), DSSP, FES, pairwise distances.
- **Multi-backend compute** -- `mdtraj` / `numba` / `cupy` / `torch` / `jax` for `compute_distances`, `compute_rmsd_matrix`, and `compute_dccm`. Default backend is `mdtraj` (only one that supports periodic boundary conditions); opt in to faster backends explicitly.
- **Dimensionality reduction & clustering** -- PCA / TICA / 2D FES, plus seven sklearn-style callable clustering classes: `Gromos`, `Hierarchical`, `DBSCAN`, `HDBSCAN`, `KMeans`, `MiniBatchKMeans`, `RegularSpace`.
- **Cheminformatics** -- descriptors, PAINS / Murcko filters, fingerprints, similarity, Butina clustering (single-threaded + Numba-parallel).
- **Visualization** -- publication-ready matplotlib plots, 2D molecule drawing, interactive 3D views via py3Dmol and nglview.
- **System preparation** -- PDBFixer wrappers, PROPKA pKa prediction, ligand topology / minimisation, trajectory slicing / merging.
- **Float32 by default** -- matches mdtraj's coordinate precision; opt-in float64 via `set_default_dtype` or per-call `dtype=`.
- **GROMACS / OpenFE automation** -- shell-script suites in `scripts/` for HPC simulation pipelines.

## Package Structure

```
mdpp
├── core        # Trajectory I/O, XVG/EDR parsers
├── analysis    # RMSD, RMSF, delta-RMSF, DCCM, SASA, contacts, DSSP, FES, PCA/TICA, clustering
│   └── _backends  # mdtraj/numba/cupy/torch/jax backends for distances/RMSD/DCCM
├── chem        # Molecular descriptors, fingerprints, similarity, PAINS filters, file I/O
├── plots       # Publication-ready 2D plots, molecule drawings, interactive 3D views
├── prep        # Protein fixing, pKa prediction, ligand parameterization, trajectory manipulation
├── constants   # Physical constants (gas constant, default temperature)
└── scripts     # Repository shell scripts for GROMACS, OpenFE, and BrownDye
```

## Quick Install

```bash
pip install -e ".[dev]"
```

For OpenMM-based preparation tools:

```bash
pip install -e ".[openmm]"
```

For optional GPU compute backends (`cupy` / `torch` / `jax`):

```bash
pip install -e ".[gpu]"
```

## Links

- [Getting Started](getting-started.md)
- [User Guide](guide/core.md)
- [Cheminformatics](guide/chem.md)
- [API Reference](api/core.md)
- [Scripts](guide/scripts.md)
