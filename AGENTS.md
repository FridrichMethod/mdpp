# mdpp agent guide

Instructions for every AI coding agent (Codex, Cursor, Copilot, Claude Code, and others) working in this repository. This file is the single source of project policy; Claude Code imports it through `CLAUDE.md`.

## What This Project Is

**mdpp** — a Python 3.12+ library for molecular dynamics simulation pre- and post-processing, plus small-molecule cheminformatics. Supports GROMACS, AMBER, OpenFE, and BrownDye workflows.

## Setup

```bash
conda create -n mdpp python=3.12 -y && conda activate mdpp
bash setup.sh
```

## Environment Usage

- Use the `mdpp` conda environment for any Python command that relies on project dependencies.
- When running non-interactively, prefer `conda run -n mdpp ...` instead of creating a separate virtualenv or using `uv run`.
- Treat the workspace `.venv/` and `uv.lock` as agent-created artifacts to avoid for this repository unless the user explicitly asks for `uv`.

## Verification

```bash
pytest                             # run all tests (auto-parallel)
pytest tests/analysis/test_metrics.py   # run a specific test file
ruff check src/ tests/ --fix       # lint
ruff format src/ tests/            # format
mypy src/mdpp/                     # type checking
pre-commit run --all-files         # full check suite
pip install -e ".[docs]" && mkdocs serve   # docs preview
```

### Pytest markers

Three custom markers gate optional test subsets (`--strict-markers` enforced):

| Marker | Purpose |
|---|---|
| `benchmark` | Performance timing tests with printed reports |
| `slow` | Resource-intensive tests (>10s runtime) |
| `gpu` | Tests that exercise GPU backends cupy/torch/jax |

Combine markers with boolean expressions:

```bash
pytest -m benchmark                     # run only benchmarks
pytest -m "not benchmark"               # skip all benchmarks
pytest -m "benchmark and not slow"      # fast benchmarks only
pytest -m "not slow"                    # skip resource-intensive tests
pytest -m "not gpu"                     # CPU-only run
pytest -m "benchmark and gpu"           # all GPU-exercising benchmarks
pytest -m "not (slow or benchmark)"     # minimum fast CI subset
pytest -m "gpu and not slow" -n 0       # GPU agreement (serial -- see note)
pytest -m "benchmark and gpu and not slow" -n 0   # fast GPU benchmarks
```

> **Always pass `-n 0` to GPU-marked runs.** The default 24-worker
> pytest-xdist parallelism opens 24 simultaneous CUDA contexts on the
> single visible GPU and triggers spurious `CUBLAS_STATUS_ALLOC_FAILED`
> / `OutOfMemoryError` failures even on 96 GB cards. The same suite
> passes cleanly under `-n 0` (or `-n 1`).

Register any new markers in `pyproject.toml` `[tool.pytest.ini_options].markers`.

### Post-Edit Validation

After modifying any production code under `src/mdpp/`, you MUST complete the following loop before considering the task done. Repeat until no CRITICAL issues remain:

1. **Write tests** -- add or update tests for every changed function. Tests live under `tests/` mirroring the `src/mdpp/` layout.
1. **Run tests** -- `conda run -n mdpp pytest <relevant scope>` (use full `pytest` when multiple areas are affected).
1. **Run pre-commit** -- `conda run -n mdpp pre-commit run --all-files` (covers ruff lint, ruff format, mypy type checking, shellcheck).
1. **Run AI review** -- request an independent AI code review of the changes.
1. **Fix issues** -- address any CRITICAL or HIGH issues from tests, pre-commit, or review.
1. **Repeat from step 2** until all checks pass and no CRITICAL issues are found.

Prefer `conda run -n mdpp ...` for all non-interactive checks.

## Important: Do Not

- Do not remove dependencies from `pyproject.toml` `[project.dependencies]`.
- Do not use relative imports.
- Do not put test `__init__.py` files in test directories.
- Do not modify files in `results/` (untracked temporary directory).
- Do not write custom parsers when a library exists (use panedr, MDAnalysis, etc.).

## Mandatory Conventions

1. **Absolute imports only** — `from mdpp.core.trajectory import load_trajectory`.
1. **Google docstrings** — enforced by ruff pydocstyle; all public functions must have Args/Returns/Raises sections.
1. **Frozen dataclasses** — analysis results use `@dataclass(frozen=True, slots=True)`.
1. **Keyword-only args** — after the first positional arg in compute/plot functions.
1. **No builtin shadowing** — do not create modules named `io`, `pdb`, `types`, etc.; the package uses `core/` (not `io/`), `protein.py` (not `pdb.py`), `_types.py` (not `types.py`).
1. **Type aliases** — shared aliases live in `mdpp._types` (`StrPath`, `PathLike`).
1. **Exports** — every `__init__.py` has an `__all__` list. New public functions must be added.
1. **Units** — internal arrays use nm/ps (MDTraj convention); display properties convert to Å/ns.
1. **Chem functions** — take `Chem.rdchem.Mol` or SMILES strings; fingerprint generators are in `FP_GENERATORS` dict.
1. **3D visualization** — `plots/three_d.py` uses py3Dmol and nglview for notebook-based interactive views.

## Python Style

- **Python >= 3.12** required. Use `type` statement for type aliases (not `TypeAlias`).
- **Line length**: 100 characters.
- **Type hints required** — every function (public and private) must have complete type annotations for all parameters and the return type. Use modern union syntax (`X | None` not `Optional[X]`).
- **No special characters** — production code and comments must use only standard ASCII. No ligatures, emoji, Unicode arrows/symbols, or non-ASCII punctuation. Standard keyboard symbols (`!@#$%^&*()` etc.) are fine.

## Float Dtype System

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

## Analysis Modules

Every `compute_*` function:

1. Takes `traj: md.Trajectory` as the first argument (or a feature matrix).
1. Uses keyword-only arguments after the first positional arg.
1. Accepts `dtype: DtypeArg = None` as the last keyword argument.
1. Returns a frozen `@dataclass(frozen=True, slots=True)`.
1. Provides unit-conversion properties (`.time_ns`, `.rmsd_angstrom`, etc.).
1. Imports trajectory helpers from `mdpp.core.trajectory`.

- Use `NDArray[np.floating]` (not `np.float64`) in result fields so the dataclass reflects the actual stored dtype, which defaults to float32.
- Validate inputs early (raise `ValueError` with descriptive messages).

## Plot Modules

Every `plot_*` function:

1. Takes an analysis result dataclass as the first argument.
1. Accepts `ax: Axes | None = None` and returns `Axes`.
1. Uses `from mdpp.plots.utils import get_axis`.
1. Sets axis labels with display units (Å, ns).

- Import result types from `mdpp.analysis.*`, not from plots.
- When adding a new plot function, also add it to `plots/__init__.py` `__all__`.

## Compute Backend Conventions

**Default backend rule**: every public compute function that accepts a `backend=` argument MUST default to `"mdtraj"` **when mdtraj provides a native kernel for that computation**. Other backends (`numba`, `torch`, `jax`, `cupy`) are performance options that callers must opt into explicitly. The only current exception is `compute_dccm`, which defaults to `"numpy"` because mdtraj has no native covariance kernel -- `numpy`'s BLAS GEMM is multi-threaded and works without any optional dependency.

Reasons:

- Only `mdtraj` supports periodic boundary conditions -- defaulting to anything else would silently drop PBC for users who don't read the backend parameter.
- All PBC-relevant analysis functions sharing the same default keeps API behavior consistent across `compute_distances`, `compute_rmsd_matrix`, `featurize_ca_distances`, etc.
- The optional GPU backends (`[gpu]` extra) must never be required for the common path.

Users who want performance explicitly pass `backend="numba"` (or a GPU backend) and accept the PBC limitation.

When reviewing or writing code, never silently change a public function's default backend away from its current value (`"mdtraj"`, or `"numpy"` for DCCM). Add performance notes in the docstring pointing users to `backend="numba"` or a GPU backend when they need speed.

**Uniform signature rule**: every backend registered in a given `BackendRegistry` MUST accept the exact same call signature as the Protocol type parameter on that registry. If one backend needs an extra keyword argument (e.g. `periodic` on mdtraj), every other backend in the same registry MUST also accept that keyword, silently ignoring it if unused (mark `# noqa: ARG001` and document as "accepted for Protocol uniformity, ignored"). This keeps the dispatcher free of per-backend branching and preserves type inference for callers.

**Registry typing rule**: every `BackendRegistry[F]` instance MUST be parameterised with an explicit `Protocol` type `F`:

```python
from typing import Protocol

class RMSDMatrixBackendFn(Protocol):
    def __call__(
        self,
        traj: md.Trajectory,
        atom_indices: NDArray[np.int_],
    ) -> NDArray[np.floating]: ...

rmsd_matrix_backends: BackendRegistry[RMSDMatrixBackendFn] = BackendRegistry(default="mdtraj")
```

Never declare a bare `BackendRegistry` without a type parameter -- `registry.get(backend)` would return an unbound `F` and the dispatcher would lose the signature of `compute_fn` at the call site. The Protocol lives in the same `_backends/_<kind>.py` file as the backends it describes (not in the shared `_registry.py`) so the registry module stays decoupled from any particular backend signature.

**Backend dtype rule**: Protocols return `NDArray[np.floating]` (not `NDArray[np.float64]`) and every backend returns its **native** dtype -- float32 for mdtraj and the GPU backends (torch/jax/cupy), float64 for numba. Public `compute_*` wrappers then cast with `astype(resolved, copy=False)` / `np.asarray(result, dtype=resolved)` so when the backend's native dtype already matches the user's resolved dtype (the float32 default), **no redundant copy** is made. This is essential at large N: forcing `float64` on an N^2 matrix would cost 115 GB at n=120k purely for a type contract and would OOM any 128 GB host. Never add an unconditional `.astype(np.float64)` at a backend boundary.

**GPU cache cleanup rule**: torch and cupy GPU-backed compute kernels MUST be decorated with the matching framework-specific cleanup decorator from `_backends/_imports.py`:

| Backend | Decorator |
|---|---|
| `torch` | `@clean_torch_cache` |
| `cupy` | `@clean_cupy_cache` |
| `jax` | (none) |

The decorators call the framework's cache-clear API (`torch.cuda.empty_cache()`, `cp.get_default_memory_pool().free_all_blocks()`) in a `finally` block so pooled device memory is returned to the driver on both normal return and exceptions. Apply decorators to inner kernel functions (e.g. `rmsd_torch`, `distances_cupy`), **never** the outer CPU-side `compute_*` wrappers (the wrappers are CPU-only and delegate to the kernel via the registry). The decorators use PEP 695 generic syntax (`[**P, T]`) so mypy preserves the Protocol signature at registry call sites.

JAX kernels are deliberately **not** decorated. `jax.clear_caches()` clears JIT compilation caches, not device memory -- trashing the compilation cache after every call forces a multi-second recompile on the next invocation. JAX has no public API for returning pooled device memory to the driver anyway (XLA manages it directly).

## Tests

Tests live in `tests/analysis/`, `tests/plots/`, and `tests/chem/`, mirroring the source tree.

- Mirror `src/mdpp/` structure under `tests/`.
- Shared fixtures in `tests/conftest.py`.
- Test functions are named `test_<function>_<behavior>`.
- Import from public API paths (`from mdpp.analysis.metrics import compute_rmsd`).
- Use `pytest.approx` for floats.
- Plotting tests: `matplotlib.use("Agg")`, close figures after assertions.
- When adding a new benchmark, decorate with `@pytest.mark.benchmark` (and `@pytest.mark.slow` if >10s; add `@pytest.mark.gpu` if it exercises cupy/torch/jax backends).

## Shell Scripts

- **Shebang**: always use `#!/usr/bin/env bash` (never `#!/bin/bash`).
- **Executable bit**: all `.sh` and `.sbatch` files must have `chmod +x`.
- First lines after the shebang: `set -euo pipefail`.
- Indent with 4 spaces (enforced by shfmt via pre-commit).
- Pass shellcheck at error severity.
- ASCII only -- no Unicode arrows, ligatures, emoji, or non-ASCII punctuation.
- All shell scripts live in top-level `scripts/<engine>/<category>/` — not packaged, copy to MD working directories.
- SLURM batch scripts (`.sbatch`) live alongside their `.sh` counterparts in the same directory.
- **Argument parsing** (for scripts accepting flags/options):
  - Use manual `while [[ $# -gt 0 ]]; do case "$1" in ...` loops (not `getopts`) to support both short and long flags.
  - Define a `usage()` function that documents all arguments.
  - Always support `-h` / `--help`.
  - Provide both short and long forms for every flag (e.g. `-j` / `--jobs`, `-n` / `--dry-run`).
  - Validate required arguments and print clear error messages on invalid input.
  - Scripts that only accept simple positional arguments (e.g. `$1`) do not need this treatment.

## Package Layout

Source is under `src/mdpp/` using the src-layout convention:

| Subpackage | Purpose | Key patterns |
|---|---|---|
| `core/` | Trajectory I/O, file parsers | `load_trajectory`, `load_trajectories`, `read_xvg`, `read_edr` |
| `constants.py` | Physical constants | `GAS_CONSTANT_KJ_MOL_K`, `DEFAULT_TEMPERATURE_K` |
| `analysis/` | Compute functions | `compute_*(traj, *, ...) -> FrozenDataclass` |
| `analysis/_backends/` | Private backend subpackage | `BackendRegistry[F]`, `require_torch/jax/cupy`, `DistanceBackend`/`RMSDBackend`/`DCCMBackend` Literals, `clean_torch_cache`/`clean_cupy_cache` decorators |
| `chem/` | Small-molecule cheminformatics | `MolSupplier`, `calc_descs`, `gen_fp`, `calc_sim`, `is_pains` |
| `plots/` | Visualization (2D, 3D, molecules) | `plot_*(result, *, ax=None) -> Axes`, `draw_mol`, `view_mol_3d` |
| `prep/` | System preparation | `fix_pdb`, `strip_solvent`, `run_propka`, ligand tools |

Shell scripts (analysis wrappers, runtime helpers, build scripts, etc.) live in the top-level `scripts/` directory (not packaged).

Examples (notebooks and data) live in `examples/` (GROMACS, OpenFE RBFE, BrownDye).

For the full file tree run `git ls-files src/ scripts/ examples/` (summarised in README.md "Package Structure" and docs/guide/scripts.md); the dependency list lives in `pyproject.toml` `[project.dependencies]` and the README.md "Dependencies" table.

### OpenFE Scripts (`scripts/openfe/`)

SLURM submission scripts for running OpenFE RBFE transformations on Sherlock.
**Requires OpenFE >= 1.10.0** for `--resume` checkpoint support.

| Script | Purpose |
|---|---|
| `quickrun/quickrun.sh` | Submit all `transformations/*.json` as SLURM array jobs (`-r N` for repeats) |
| `quickrun/quickrun.sbatch` | Batch script: runs `openfe quickrun --resume` via Apptainer on a `--gpu_cmode=shared` GPU |
| `runtime/check_status.sh` | Check transformation replica status and optionally restart failed replicas |
| `runtime/monitor.sbatch` | Periodic monitor: runs check_status, emails report, self-resubmits via SLURM |

- `--gpu_cmode=shared` puts the GPU in Default compute mode, which already lets a single openfe process hold the multiple CUDA contexts openmmtools ContextCache needs. Do not start a per-job CUDA MPS control daemon in `quickrun.sbatch`: when two array tasks land on the same node, the competing per-job daemons collide and OpenMM reports "No compatible CUDA device is available".
- `--resume` enables checkpoint-based resumption after preemption on `owners` partition.
- Output goes to `results/<name>/replica_<id>/`.

## File Naming

- Analysis modules: `src/mdpp/analysis/<topic>.py`
- Plot modules: `src/mdpp/plots/<topic>.py`
- Helper utilities within a subpackage: `utils.py`
- MDP config templates: `scripts/gromacs/mdps/<ff>/<step>.mdp`
- Shell scripts (not packaged): `scripts/<engine>/<category>/<script>.sh`
- SLURM scripts: `scripts/<engine>/<category>/<script>.sbatch`

## Clustering API

`mdpp.analysis.clustering` exposes seven sklearn-style callable classes:

| Class | Input | Backend / notes |
|---|---|---|
| `Gromos` | RMSD matrix | Numba JIT (greedy largest-first, Daura 1999) |
| `Hierarchical` | RMSD matrix | scipy linkage + fcluster |
| `DBSCAN` | RMSD matrix | Numba JIT (default) or sklearn `metric="precomputed"` |
| `HDBSCAN` | RMSD matrix | sklearn `metric="precomputed"` |
| `KMeans` | Feature matrix | scikit-learn |
| `MiniBatchKMeans` | Feature matrix | scikit-learn |
| `RegularSpace` | Feature matrix | deeptime |

Each class is `@dataclass(frozen=True, slots=True)` with parameters at construction and a `__call__(data) -> ClusteringResult | FeatureClusteringResult` invocation.

```python
result = Gromos(cutoff_nm=0.15)(rmsd_matrix.rmsd_matrix_nm)
result = KMeans(n_clusters=10)(pca.projections)
```

Do **not** add the old function-form wrappers (`compute_gromos_clusters`, etc.) -- they were removed and there is no backward-compat shim.

## Adding a New Analysis

1. Create/extend a file in `src/mdpp/analysis/`.
1. Define result dataclass(es) with frozen=True, slots=True.
1. Write `compute_*` function following the existing signature pattern.
1. Add exports to `src/mdpp/analysis/__init__.py`.
1. If visual output makes sense, add `plot_*` in `src/mdpp/plots/` and export it.
1. Write tests in `tests/analysis/`.

## Adding a New Compute Backend

For existing multi-backend functions (e.g. `compute_rmsd_matrix`, pairwise distances):

1. Add the implementation in the matching `src/mdpp/analysis/_backends/_<kind>.py` file, matching the `Protocol` type defined at the top of that file exactly.
1. Use `require_torch()` / `require_jax()` / `require_cupy()` from `_backends/_imports.py` for optional GPU libraries -- never import them at module top-level.
1. Decorate torch/cupy GPU kernels with `@clean_torch_cache` / `@clean_cupy_cache` from `_backends/_imports.py` so pooled memory is released in a `finally` block after the kernel runs. Do **not** apply any cleanup decorator to JAX kernels -- `jax.clear_caches()` trashes JIT compilation caches and forces slow recompiles.
1. If you introduce a new keyword argument, also retrofit every existing backend in the same registry to accept it (silently ignoring when unused, marked `# noqa: ARG001`).
1. Register in the module's `BackendRegistry` at the bottom of the file.
1. Add the backend name to the corresponding `Literal` alias (`DistanceBackend` / `RMSDBackend` / `DCCMBackend`) in `_backends/_registry.py`.
1. Add agreement tests in `tests/analysis/test_<kind>.py` guarded by the relevant `requires_*` skip marker and `@pytest.mark.gpu` (if GPU-only).
1. **Do not change the public function's default backend** -- keep `compute_distances` / `compute_rmsd_matrix` / `featurize_ca_distances` defaulting to `"mdtraj"` and `compute_dccm` defaulting to `"numpy"`.

## Adding a New Backend Registry

To introduce a registry for a new multi-backend compute function:

1. Create `src/mdpp/analysis/_backends/_<kind>.py` with a `Protocol` class defining the shared call signature.
1. Declare the registry as `<kind>_backends: BackendRegistry[<Kind>BackendFn] = BackendRegistry(default="mdtraj")` (or another sensible no-optional-dep default if mdtraj has no kernel for that computation -- e.g. `_dccm.py` uses `default="numpy"`). Always parameterise with the Protocol so callers get typed `compute_fn` from `registry.get()`.
1. Add a `Literal` alias to `_backends/_registry.py` (`type <Kind>Backend = Literal["mdtraj", "numba", ...]`) and re-export it from `_backends/__init__.py`.
1. The public wrapper in `src/mdpp/analysis/<kind>.py` imports the registry and delegates via `compute_fn = <kind>_backends.get(backend)`, letting mypy infer the Protocol type at the call site.

## Adding a New Chem Function

1. Create/extend a file in `src/mdpp/chem/`.
1. Functions take `Chem.rdchem.Mol` or SMILES strings as input.
1. Add exports to `src/mdpp/chem/__init__.py`.
1. Write tests in `tests/chem/`.

## Adding a New Parser

1. Prefer wrapping an existing library (panedr, MDAnalysis) over writing custom parsing.
1. Add to `src/mdpp/core/parsers.py`.
1. Re-export in `core/__init__.py`.
