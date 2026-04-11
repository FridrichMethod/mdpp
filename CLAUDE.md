# CLAUDE.md

This file provides guidance for Claude Code (claude-code, claude terminal) when working in this repository.

## Project Overview

**mdpp** is a Python package for molecular dynamics (MD) simulation pre- and post-processing. It provides trajectory loading, structural/dynamic analysis, cheminformatics, visualization, and system preparation utilities for GROMACS, AMBER, OpenFE, and other MD engines.

## Repository Structure

```
src/mdpp/
├── _types.py        # shared type aliases (StrPath, PathLike, DtypeArg)
├── _dtype.py        # float dtype config (get/set_default_dtype, resolve_dtype)
├── constants.py     # physical constants (GAS_CONSTANT_KJ_MOL_K, DEFAULT_TEMPERATURE_K)
├── core/            # trajectory I/O, XVG/EDR parsers
│   ├── trajectory.py    # load_trajectory, load_trajectories, align_trajectory,
│   │                     # select_atom_indices, residue_ids_from_indices, trajectory_time_ps
│   └── parsers.py       # read_xvg, read_edr (thin wrappers around panedr/numpy)
├── analysis/        # compute_* functions returning frozen dataclass results
│   ├── _backends/       # private subpackage: pluggable compute backends
│   │   ├── _registry.py     # BackendRegistry[F] + DistanceBackend/RMSDBackend Literals
│   │   ├── _imports.py      # lazy require_torch/jax/cupy + has_* flags
│   │   ├── _distances.py    # 5 pairwise-distance backends (mdtraj/numba/torch/jax/cupy)
│   │   └── _rmsd_matrix.py  # 5 RMSD matrix backends (same set, QCP kernel for numba)
│   ├── metrics.py       # RMSD, RMSF, delta-RMSF, DCCM, SASA, radius of gyration
│   ├── hbond.py         # hydrogen bond detection
│   ├── contacts.py      # inter-residue contacts, native contacts Q(t)
│   ├── distance.py      # pairwise distances, minimum distance (thin wrapper)
│   ├── dssp.py          # secondary structure (DSSP)
│   ├── decomposition.py # PCA (with projection), TICA, backbone torsion featurization
│   ├── fes.py           # 2D free energy surfaces
│   └── clustering.py    # RMSD matrix, GROMOS clustering (thin wrapper)
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

# Skip benchmarks for fast CI
pytest -m "not benchmark"

# Run only benchmarks
pytest -m benchmark

# Skip slow + benchmark tests
pytest -m "not (slow or benchmark)"

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

### Float Dtype System

The package uses **float32 by default**, matching mdtraj's coordinate storage precision. Float64 is not forced anywhere in the analysis pipeline. Users can override globally or per-function.

**Architecture** (`_dtype.py`):

- `get_default_dtype()` / `set_default_dtype(np.float64)` -- global control.
- `resolve_dtype(dtype)` -- resolves per-function `dtype` arg, falling back to global default.
- `DtypeArg` (`_types.py`) -- shared type alias (`type[np.floating] | np.dtype[np.floating] | None`) used for all `dtype` parameters.

**Rules for new code**:

- Every `compute_*` function accepts `dtype: DtypeArg = None` as the last keyword argument.
- Call `resolved = resolve_dtype(dtype)` at the top, then cast outputs to `resolved`.
- **Never force float64** for "numerical stability". mdtraj coordinates are float32; you cannot recover precision that was never there. Empirical tests confirm float32 is sufficient for RMSF (error ~1e-5 nm), DCCM (error ~4e-6), FES (error ~2e-6 kJ/mol).
- Float64 appears only where **external constraints** produce it:
  - **Numba JIT**: `float()` casts map to double in Numba's type system. Cast the kernel output to `resolved` dtype afterward.
  - **Deeptime TICA**: upcasts to float64 internally for covariance. No explicit pre-cast needed from our side.
  - **`np.histogram2d`**: returns float64 density regardless of input dtype.
  - **`np.mean` on boolean arrays**: NumPy defaults to float64 for boolean reductions.
- Import `DtypeArg` from `mdpp._types` -- do not inline the union type.

### Analysis Modules

Every `compute_*` function:

1. Takes `traj: md.Trajectory` as the first argument (or a feature matrix).
1. Uses keyword-only arguments after the first positional arg.
1. Accepts `dtype: DtypeArg = None` as the last keyword argument.
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

#### Pytest Markers

Three custom markers control test selection (`--strict-markers` enforced):

| Marker | Purpose | Deselect |
|--------|---------|----------|
| `benchmark` | Performance timing tests with printed reports | `-m "not benchmark"` |
| `slow` | Resource-intensive tests (>10s runtime) | `-m "not slow"` |
| `gpu` | Tests that exercise GPU backends cupy/torch/jax | `-m "not gpu"` |

Combine markers with boolean expressions, e.g.:

- `pytest -m "benchmark and not slow"` -- fast benchmarks only
- `pytest -m "not gpu"` -- CPU-only test run (skips all GPU backend coverage)
- `pytest -m "benchmark and gpu and not slow"` -- fast GPU benchmarks

Current benchmark tests:

- `tests/analysis/test_ca_distances.py` -- fast (1K-100/1K-200/2K-200) and slow (3K-200/5K-200) pairwise distance backend tiers, both marked `gpu`.
- `tests/analysis/test_clustering.py` -- fast (100f/200f) and slow (500f/1000f) RMSD matrix backend tiers, both marked `gpu`.
- `tests/core/test_trajectory.py` -- atom selection: direct loading vs load-all+slice memory and timing.

When adding a new benchmark, decorate with `@pytest.mark.benchmark` (and `@pytest.mark.slow` if >10s; add `@pytest.mark.gpu` if it exercises cupy/torch/jax backends). Register any new markers in `pyproject.toml` `[tool.pytest.ini_options].markers`.

### Shell Scripts

- **Shebang**: always use `#!/usr/bin/env bash` (never `#!/bin/bash`).
- **Executable bit**: all `.sh` and `.sbatch` files must have `chmod +x`.
- `set -euo pipefail`, 4-space indent, pass shellcheck.
- All shell scripts live in top-level `scripts/<engine>/<category>/` — not packaged, copy to MD working directories.
- SLURM batch scripts (`.sbatch`) live alongside their `.sh` counterparts in the same directory.
- **Argument parsing** (for scripts accepting flags/options):
  - Use manual `while [[ $# -gt 0 ]]; do case "$1" in ...` loops (not `getopts`) to support both short and long flags.
  - Define a `usage()` function that documents all arguments.
  - Always support `-h` / `--help`.
  - Provide both short and long forms for every flag (e.g. `-j` / `--jobs`, `-n` / `--dry-run`).
  - Validate required arguments and print clear error messages on invalid input.
  - Scripts that only accept simple positional arguments (e.g. `$1`) do not need this treatment.

## Dependencies

Core dependencies are in `pyproject.toml` `[project.dependencies]`. Key libraries:

- **mdtraj** — trajectory loading and geometry calculations
- **MDAnalysis** — XVG auxiliary reader
- **panedr** — GROMACS EDR parsing
- **scikit-learn** — PCA, clustering
- **deeptime** — TICA
- **rdkit** — cheminformatics: ligand topology, descriptors, fingerprints, similarity
- **numba** — parallel CPU kernels: pairwise distances and RMSD matrix (`analysis/_backends/`), similarity (`chem/similarity.py`)
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
- **cupy / torch / jax** — optional GPU backends for pairwise distances (`pip install mdpp[gpu]`)

## Adding New Features

### New analysis function

1. Create or extend a module in `src/mdpp/analysis/`.
1. Define a frozen result dataclass following existing patterns.
1. Write the `compute_*` function with keyword-only args.
1. Add re-export to `analysis/__init__.py` and `__all__`.
1. Add corresponding `plot_*` function in `plots/` if applicable.
1. Add the plot re-export to `plots/__init__.py` and `__all__`.
1. Write tests in `tests/analysis/`.

### Compute backend conventions

**Default backend rule**: every public compute function that accepts a
`backend=` argument MUST default to `"mdtraj"`. Other backends exist
for performance (`"numba"` for CPU-parallel, `"torch"`/`"jax"`/`"cupy"`
for GPU) and MUST be opted into explicitly by the caller.

Rationale:

- **Correctness first**: only `mdtraj` supports periodic boundary
  conditions. Defaulting to anything else would silently drop PBC for
  users who rely on unit cells without reading the backend parameter.
- **API consistency**: all analysis functions share the same default so
  switching between `compute_distances`, `compute_rmsd_matrix`,
  `featurize_ca_distances`, etc. doesn't silently change semantics.
- **No hidden GPU dependency**: defaulting to `numba`/`torch`/`jax`/`cupy`
  would require heavy optional dependencies for the common path.

Users who want performance explicitly pass `backend="numba"` (or a GPU
backend) and accept the PBC limitation.

**Uniform signature rule**: every backend registered in a given
`BackendRegistry` MUST accept the exact same call signature as the
Protocol type parameter on that registry. If one backend needs an
extra keyword argument (e.g. `periodic` on mdtraj), every other
backend in the same registry MUST also accept that keyword, silently
ignoring it if unused (mark `# noqa: ARG001` and document in the
docstring that the arg is accepted for Protocol uniformity and
ignored). This keeps the dispatcher free of per-backend branching
and preserves type inference for callers.

**Registry typing rule**: every `BackendRegistry[F]` instance MUST be
parameterised with an explicit `Protocol` type `F`:

```python
from typing import Protocol

class RMSDMatrixBackendFn(Protocol):
    def __call__(
        self,
        traj: md.Trajectory,
        atom_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]: ...

rmsd_matrix_backends: BackendRegistry[RMSDMatrixBackendFn] = BackendRegistry(default="mdtraj")
```

Never declare a bare `BackendRegistry` without a type parameter --
`registry.get(backend)` would return an unbound `F` and the dispatcher
would lose the signature of `compute_fn` at the call site. The
Protocol lives in the same `_backends/_<kind>.py` file as the backends
it describes (not in the shared `_registry.py`) so the registry module
stays decoupled from any particular backend signature.

**GPU cache cleanup rule**: torch and cupy GPU-backed compute
kernels MUST be decorated with the matching framework-specific
cache-cleanup decorator from `_backends/_imports.py`:

| Backend | Decorator |
|---|---|
| `torch` | `@clean_torch_cache` |
| `cupy` | `@clean_cupy_cache` |
| `jax` | (none) |

JAX kernels are deliberately **not** decorated. `jax.clear_caches()`
clears the JIT compilation cache, not device memory, and wiping the
compilation cache after every call forces a multi-second recompile
on the next invocation. JAX has no public API for returning pooled
device memory to the driver anyway -- XLA manages it directly.

Example:

```python
from mdpp.analysis._backends._imports import clean_torch_cache

@clean_torch_cache
def rmsd_torch(traj, atom_indices):
    ...
    return result.cpu().numpy()
```

The decorators call the framework's cache-clear API
(`torch.cuda.empty_cache()`, `cp.get_default_memory_pool().free_all_blocks()`,
`jax.clear_caches()`) in a `finally` block so pooled memory is
returned to the driver on normal return *and* on exceptions. Apply
decorators to the inner kernel functions, **not** the outer
`compute_*` wrappers (the wrappers are CPU-only and delegate to the
kernel via the registry). The decorators use PEP 695 generic
syntax (`[**P, T]`) so mypy preserves the Protocol signature at
registry call sites.

### New compute backend

To add a new backend (e.g. `cupy`) for an existing compute function like the RMSD matrix or pairwise distances:

1. Add an implementation function in the matching `src/mdpp/analysis/_backends/_<kind>.py` file.
1. Use lazy imports via `require_torch()` / `require_jax()` / `require_cupy()` from `_backends/_imports.py` -- never import optional GPU libraries at module top-level.
1. Decorate torch/cupy GPU kernels with the matching `@clean_torch_cache` / `@clean_cupy_cache` from `_backends/_imports.py` so pooled GPU memory is released in a `finally` block after the kernel runs. **Do not** decorate JAX kernels -- `jax.clear_caches()` trashes JIT compilation caches and forces slow recompiles.
1. Match the `Protocol` type defined at the top of the same file exactly (e.g. `RMSDMatrixBackendFn`, `DistanceBackendFn`). If you introduce a new keyword argument, also retrofit every existing backend in the same registry to accept it (silently ignoring if unused, `# noqa: ARG001`).
1. Register the function in the module's `BackendRegistry` at the bottom of the file.
1. Add the backend name to the corresponding `Literal` alias in `_backends/_registry.py` (`DistanceBackend` or `RMSDBackend`).
1. Add agreement tests in `tests/analysis/test_<kind>.py` guarded by the relevant `requires_*` marker and `@pytest.mark.gpu` (if GPU-only).
1. **Do not change the public function's default backend** -- keep it at `"mdtraj"`.

### New backend registry

To introduce a registry for a new multi-backend compute function:

1. Create `src/mdpp/analysis/_backends/_<kind>.py` with a `Protocol` class defining the shared call signature.
1. Declare the registry as `<kind>_backends: BackendRegistry[<Kind>BackendFn] = BackendRegistry(default="mdtraj")` -- always parameterise with the Protocol so callers get typed `compute_fn` from `registry.get()`.
1. Add a `Literal` alias to `_backends/_registry.py` (`type <Kind>Backend = Literal["mdtraj", "numba", ...]`) and re-export it from `_backends/__init__.py`.
1. The public wrapper in `src/mdpp/analysis/<kind>.py` should import the registry and delegate via `compute_fn = <kind>_backends.get(backend)`, letting mypy infer the Protocol type.

### New cheminformatics function

1. Create or extend a module in `src/mdpp/chem/`.
1. Functions take `Chem.rdchem.Mol` or SMILES strings as input.
1. Add re-export to `chem/__init__.py` and `__all__`.
1. Write tests in `tests/chem/`.

### New parser

1. Prefer wrapping an existing library (panedr, MDAnalysis) over writing custom parsing.
1. Add to `src/mdpp/core/parsers.py`.
1. Re-export in `core/__init__.py`.
