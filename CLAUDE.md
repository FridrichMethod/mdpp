# CLAUDE.md

This file provides guidance for Claude Code (claude-code, claude terminal) when working in this repository.

## Project Overview

**mdpp** is a Python package for molecular dynamics (MD) simulation pre- and post-processing. It provides trajectory loading, structural/dynamic analysis, visualization, and system preparation utilities for GROMACS, AMBER, OpenFE, and other MD engines.

## Repository Structure

```
src/mdpp/
├── _types.py        # shared type aliases (StrPath, PathLike)
├── core/            # trajectory I/O, XVG/EDR parsers
│   ├── trajectory.py    # load_trajectory, align_trajectory, select_atom_indices
│   └── parsers.py       # read_xvg, read_edr (thin wrappers around panedr/numpy)
├── analysis/        # compute_* functions returning frozen dataclass results
│   ├── metrics.py       # RMSD, RMSF, DCCM, SASA, radius of gyration
│   ├── hbond.py         # hydrogen bond detection
│   ├── contacts.py      # inter-residue contacts, native contacts Q(t)
│   ├── distance.py      # pairwise distances, minimum distance
│   ├── dssp.py          # secondary structure (DSSP)
│   ├── decomposition.py # PCA, TICA, backbone torsion featurization
│   ├── fes.py           # 2D free energy surfaces
│   └── clustering.py    # RMSD matrix, GROMOS clustering
├── plots/           # plot_* functions returning matplotlib Axes
│   ├── utils.py         # get_axis helper
│   ├── timeseries.py    # RMSD, RMSF, SASA, Rg, distances, energy, H-bonds, Q(t)
│   ├── matrix.py        # DCCM heatmap
│   ├── fes.py           # FES contour plot
│   ├── scatter.py       # PCA/TICA projection, Ramachandran
│   └── contacts.py      # contact map heatmap
├── prep/            # system preparation
│   ├── protein.py       # fix_pdb, strip_solvent, extract_chain
│   ├── ligand.py        # assign_topology, constraint_minimization
│   └── topology.py      # merge, slice, subsample trajectories
scripts/             # shell scripts (NOT packaged, copy to MD working directories)
├── gromacs/
│   ├── analysis/        # gmx_rmsd.sh, gmx_rmsf.sh, etc.
│   ├── compilation/     # gmx_compile.sh, sherlock/ variants
│   ├── mdps/            # force-field-specific GROMACS MDP templates
│   ├── mdenv/           # environment setup (sherlock/)
│   ├── mdrun/           # mdprep.sh, mdrun.sh, rest2/, sherlock/ sbatch files
│   ├── postprocessing/  # gmx_postprocessing.sh
│   ├── runtime/         # check_status.sh, restart.sh, extend.sh, export.sh
│   └── visualization/   # pymol_movie.pml
└── openfe/              # quickrun.sh, quickrun.sbatch, restart.sh

tests/               # mirrors src/ layout (tests/analysis/, tests/plots/, tests/chem/)
notebooks/           # Jupyter notebooks for interactive analysis
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

After modifying code, always validate in the `mdpp` conda environment before considering the task complete.

- Prefer `conda run -n mdpp ...` for non-interactive checks.
- Run `conda run -n mdpp pre-commit run --all-files` as the standard post-edit gate. This is the default way to cover linting and type checking, rather than relying only on standalone `ruff` or `mypy`.
- Also run the most relevant `pytest` scope for the files you changed. Use full `pytest` when the change affects multiple areas.

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

### Plot Modules

Every `plot_*` function:

1. Takes an analysis result dataclass as the first argument.
1. Accepts `ax: Axes | None = None` and returns `Axes`.
1. Uses `from mdpp.plots.utils import get_axis`.
1. Sets axis labels with display units (Å, ns).

### Tests

- Mirror `src/mdpp/` structure under `tests/`.
- Shared fixtures in `tests/conftest.py`.
- Use `pytest.approx` for floats.
- Plotting tests: `matplotlib.use("Agg")`, close figures after assertions.

### Shell Scripts

- `set -euo pipefail`, 4-space indent, pass shellcheck.
- All shell scripts live in top-level `scripts/<engine>/<category>/` — not packaged, copy to MD working directories.
- SLURM variants in `sherlock/` subdirectories.

## Dependencies

Core dependencies are in `pyproject.toml` `[project.dependencies]`. Key libraries:

- **mdtraj** — trajectory loading and geometry calculations
- **MDAnalysis** — XVG auxiliary reader
- **panedr** — GROMACS EDR parsing
- **scikit-learn** — PCA, clustering
- **deeptime** — TICA
- **rdkit** — ligand topology
- **openmm + pdbfixer** — PDB fixing (optional, `[openmm]` extra)
- **matplotlib** — all plotting

## Adding New Features

### New analysis function

1. Create or extend a module in `src/mdpp/analysis/`.
1. Define a frozen result dataclass following existing patterns.
1. Write the `compute_*` function with keyword-only args.
1. Add re-export to `analysis/__init__.py` and `__all__`.
1. Add corresponding `plot_*` function in `plots/` if applicable.
1. Add the plot re-export to `plots/__init__.py` and `__all__`.
1. Write tests in `tests/analysis/`.

### New parser

1. Prefer wrapping an existing library (panedr, MDAnalysis) over writing custom parsing.
1. Add to `src/mdpp/core/parsers.py`.
1. Re-export in `core/__init__.py`.
