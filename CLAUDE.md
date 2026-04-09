# CLAUDE.md

This file provides guidance for Claude Code (claude-code, claude terminal) when working in this repository.

## Project Overview

**mdpp** is a Python package for molecular dynamics (MD) simulation pre- and post-processing. It provides trajectory loading, structural/dynamic analysis, cheminformatics, visualization, and system preparation utilities for GROMACS, AMBER, OpenFE, and other MD engines.

## Repository Structure

```
src/mdpp/
├── _types.py        # shared type aliases (StrPath, PathLike)
├── constants.py     # physical constants (GAS_CONSTANT_KJ_MOL_K, DEFAULT_TEMPERATURE_K)
├── core/            # trajectory I/O, XVG/EDR parsers
│   ├── trajectory.py    # load_trajectory, load_trajectories, align_trajectory,
│   │                     # select_atom_indices, residue_ids_from_indices, trajectory_time_ps
│   └── parsers.py       # read_xvg, read_edr (thin wrappers around panedr/numpy)
├── analysis/        # compute_* functions returning frozen dataclass results
│   ├── metrics.py       # RMSD, RMSF, delta-RMSF, DCCM, SASA, radius of gyration
│   ├── hbond.py         # hydrogen bond detection
│   ├── contacts.py      # inter-residue contacts, native contacts Q(t)
│   ├── distance.py      # pairwise distances, minimum distance
│   ├── dssp.py          # secondary structure (DSSP)
│   ├── decomposition.py # PCA (with projection), TICA, backbone torsion featurization
│   ├── fes.py           # 2D free energy surfaces
│   └── clustering.py    # RMSD matrix, GROMOS clustering
├── chem/            # small-molecule cheminformatics (RDKit-based)
│   ├── descriptors.py   # molecular descriptor calculation and filtering
│   ├── filters.py       # Murcko scaffold extraction, PAINS filters
│   ├── fingerprints.py  # fingerprint generation (Morgan/ECFP), Butina clustering
│   ├── similarity.py    # Tanimoto and other similarity metrics, Numba-parallel kernels
│   └── suppliers.py     # MolSupplier: iterate molecules from SDF/SMILES/MOL2 files
├── plots/           # plot_* / draw_* / view_* functions
│   ├── utils.py         # get_axis helper
│   ├── timeseries.py    # RMSD, RMSF, SASA, Rg, distances, energy, H-bonds, Q(t)
│   ├── matrix.py        # DCCM heatmap
│   ├── fes.py           # FES contour plot
│   ├── scatter.py       # PCA/TICA projection, Ramachandran
│   ├── contacts.py      # contact map heatmap
│   ├── molecules.py     # 2D molecule drawing (draw_mol, draw_mols, get_highlight_bonds)
│   └── three_d.py       # 3D visualization (view_mol_3d, view_traj_3d via py3Dmol/nglview)
├── prep/            # system preparation
│   ├── protein.py       # fix_pdb, strip_solvent, extract_chain, run_propka,
│   │                     # PropkaResult, PropkaResidue, ChainSelect
│   ├── ligand.py        # assign_topology, constraint_minimization
│   └── topology.py      # merge, slice, subsample trajectories
scripts/             # shell scripts (NOT packaged, copy to MD working directories)
├── gromacs/
│   ├── analysis/        # gmx_rmsd.sh, gmx_rmsf.sh, etc.
│   ├── compilation/     # gmx_compile.sh, gmx_mpi_compile.sh, plumed_compile.sh
│   ├── data_transfer/   # dtn_download.sh (DTN rsync from Sherlock)
│   ├── mdps/            # force-field-specific GROMACS MDP templates
│   ├── mdenv/           # environment setup (source_gmx.sh, source_plumed.sh)
│   ├── mdrun/           # mdprep.sh, mdrun.sh, mdrun.sbatch, rest2/
│   ├── postprocessing/  # gmx_postprocessing.sh
│   ├── runtime/         # check_status.sh, restart.sh, extend.sh, export.sh
│   └── visualization/   # pymol_movie.pml
├── openfe/
│   ├── quickrun/        # quickrun.sh, quickrun.sbatch
│   └── runtime/         # check_status.sh, monitor.sbatch
examples/            # worked examples and notebooks
├── gromacs/             # GROMACS analysis notebooks (RMSD, RMSF, DCCM, FES, I/O)
├── openfe/              # OpenFE RBFE workflow notebook + input PDBs
└── browndye/            # BrownDye2 complex PQR preparation

tests/               # mirrors src/ layout (tests/analysis/, tests/plots/, tests/chem/)
docs/                # mkdocs documentation (guide/ and api/)
```

## Build & Test Commands

```bash
# Environment setup (conda)
conda create -n mdpp python=3.12 -y && conda activate mdpp
bash setup.sh

# Run all tests (parallelized)
pytest

# Run a specific test file
pytest tests/analysis/test_metrics.py

# Lint and format
ruff check src/ tests/ --fix
ruff format src/ tests/

# Type checking
mypy src/mdpp/

# Pre-commit (lint + format + typecheck + shellcheck)
pre-commit run --all-files

# Docs preview
pip install -e ".[docs]"
mkdocs serve
```

## Post-Edit Validation

After modifying any production code under `src/mdpp/`, you MUST complete the following loop before considering the task done. Repeat until no CRITICAL issues remain:

1. **Write tests** -- add or update tests for every changed function. Tests live under `tests/` mirroring the `src/mdpp/` layout.
1. **Run tests** -- `conda run -n mdpp pytest <relevant scope>` (use full `pytest` when multiple areas are affected).
1. **Run pre-commit** -- `conda run -n mdpp pre-commit run --all-files` (covers ruff lint, ruff format, mypy type checking, shellcheck).
1. **Run Codex review** -- invoke `/codex:review` to get an independent review of the changes.
1. **Fix issues** -- address any CRITICAL or HIGH issues from tests, pre-commit, or Codex review.
1. **Repeat from step 2** until all checks pass and no CRITICAL issues are found.

Prefer `conda run -n mdpp ...` for all non-interactive checks.

## Coding Conventions

### Python Style

- **Python >= 3.12** required. Use `type` statement for type aliases.
- **Google docstrings** enforced by ruff pydocstyle.
- **Absolute imports only** — `from mdpp.core.trajectory import load_trajectory`.
- **Line length**: 100 characters.
- **No builtin shadowing** — the package uses `core/` (not `io/`), `protein.py` (not `pdb.py`), `_types.py` (not `types.py`).
- **Type hints required** — every function (public and private) must have complete type annotations for all parameters and the return type. Use modern union syntax (`X | None` not `Optional[X]`).
- **No special characters** — production code and comments must use only standard ASCII. No ligatures, emoji, Unicode arrows/symbols, or non-ASCII punctuation. Standard keyboard symbols (`!@#$%^&*()` etc.) are fine.

### Analysis Modules

Every `compute_*` function:

1. Takes `traj: md.Trajectory` as the first argument (or a feature matrix).
1. Uses keyword-only arguments after the first positional arg.
1. Returns a frozen `@dataclass(frozen=True, slots=True)`.
1. Provides unit-conversion properties (`.time_ns`, `.rmsd_angstrom`, etc.).
1. Imports trajectory helpers from `mdpp.core.trajectory`.

### Chem Modules

The `chem/` subpackage provides RDKit-based cheminformatics utilities:

- Functions take `Chem.rdchem.Mol` or SMILES strings as input.
- `MolSupplier` provides an iterator over molecules from SDF/SMILES/MOL2 files.
- Fingerprint generators are registered in the `FP_GENERATORS` dict.
- Similarity kernels use Numba-parallel acceleration for bulk computation.

### Plot Modules

Every `plot_*` function:

1. Takes an analysis result dataclass as the first argument.
1. Accepts `ax: Axes | None = None` and returns `Axes`.
1. Uses `from mdpp.plots.utils import get_axis`.
1. Sets axis labels with display units (Å, ns).

The `molecules.py` module provides 2D structure drawing (`draw_mol`, `draw_mols`).
The `three_d.py` module provides interactive 3D visualization via py3Dmol and nglview.

### Tests

- Mirror `src/mdpp/` structure under `tests/`.
- Shared fixtures in `tests/conftest.py`.
- Use `pytest.approx` for floats.
- Plotting tests: `matplotlib.use("Agg")`, close figures after assertions.

### Shell Scripts

- `set -euo pipefail`, 4-space indent, pass shellcheck.
- All shell scripts live in top-level `scripts/<engine>/<category>/` — not packaged, copy to MD working directories.
- SLURM batch scripts (`.sbatch`) live alongside their `.sh` counterparts in the same directory.

## Dependencies

Core dependencies are in `pyproject.toml` `[project.dependencies]`. Key libraries:

- **mdtraj** — trajectory loading and geometry calculations
- **MDAnalysis** — XVG auxiliary reader
- **panedr** — GROMACS EDR parsing
- **scikit-learn** — PCA, clustering
- **deeptime** — TICA
- **rdkit** — cheminformatics: ligand topology, descriptors, fingerprints, similarity
- **numba** — parallel similarity kernels in `chem/similarity.py`
- **biopython** — PDB chain extraction (`Bio.PDB.Select`)
- **biotite** — structural bioinformatics utilities
- **propka** — pKa prediction (`prep/protein.py`)
- **pdb-tools / pdb2pqr** — PDB/PQR manipulation
- **prody** — protein dynamics and structural analysis
- **ParmEd** — parameter/topology file interconversion
- **openmm + pdbfixer** — PDB fixing (optional, `[openmm]` extra)
- **matplotlib** — static 2D plotting
- **mplplots** — custom matplotlib style/helpers
- **seaborn** — statistical visualization
- **plotly** — interactive plotting
- **py3dmol** — 3D molecule visualization in notebooks
- **nglview** — 3D trajectory visualization in notebooks
- **Pillow** — image handling for molecule drawings
- **numpy / scipy / pandas / polars** — numerical and data handling
- **tqdm** — progress bars

## Adding New Features

### New analysis function

1. Create or extend a module in `src/mdpp/analysis/`.
1. Define a frozen result dataclass following existing patterns.
1. Write the `compute_*` function with keyword-only args.
1. Add re-export to `analysis/__init__.py` and `__all__`.
1. Add corresponding `plot_*` function in `plots/` if applicable.
1. Add the plot re-export to `plots/__init__.py` and `__all__`.
1. Write tests in `tests/analysis/`.

### New cheminformatics function

1. Create or extend a module in `src/mdpp/chem/`.
1. Functions take `Chem.rdchem.Mol` or SMILES strings as input.
1. Add re-export to `chem/__init__.py` and `__all__`.
1. Write tests in `tests/chem/`.

### New parser

1. Prefer wrapping an existing library (panedr, MDAnalysis) over writing custom parsing.
1. Add to `src/mdpp/core/parsers.py`.
1. Re-export in `core/__init__.py`.
