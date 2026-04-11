# AGENTS.md

Instructions for AI coding agents (OpenAI Codex, etc.) working in this repository.

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
pytest                             # run all tests
ruff check src/ tests/ --fix       # lint
ruff format src/ tests/            # format
pre-commit run --all-files         # full check suite
```

After modifying any production code under `src/mdpp/`, you MUST complete the following loop before considering the task done. Repeat until no CRITICAL issues remain:

1. **Write tests** -- add or update tests for every changed function. Tests live under `tests/` mirroring the `src/mdpp/` layout.
1. **Run tests** -- `conda run -n mdpp pytest <relevant scope>` (use full `pytest` when multiple areas are affected).
1. **Run pre-commit** -- `conda run -n mdpp pre-commit run --all-files` (covers ruff lint, ruff format, mypy type checking, shellcheck).
1. **Run AI review** -- request an independent AI code review of the changes (e.g. Codex review, Claude code-reviewer, or equivalent).
1. **Fix issues** -- address any CRITICAL or HIGH issues from tests, pre-commit, or review.
1. **Repeat from step 2** until all checks pass and no CRITICAL issues are found.

Prefer `conda run -n mdpp ...` for all non-interactive checks.

## Package Layout

Source is under `src/mdpp/` using the src-layout convention:

| Subpackage | Purpose | Key patterns |
|---|---|---|
| `core/` | Trajectory I/O, file parsers | `load_trajectory`, `load_trajectories`, `read_xvg`, `read_edr` |
| `constants.py` | Physical constants | `GAS_CONSTANT_KJ_MOL_K`, `DEFAULT_TEMPERATURE_K` |
| `analysis/` | Compute functions | `compute_*(traj, *, ...) -> FrozenDataclass` |
| `analysis/_backends/` | Private backend subpackage | `BackendRegistry[F]`, `require_torch/jax/cupy`, `DistanceBackend`/`RMSDBackend` Literals |
| `chem/` | Small-molecule cheminformatics | `MolSupplier`, `calc_descs`, `gen_fp`, `calc_sim`, `is_pains` |
| `plots/` | Visualization (2D, 3D, molecules) | `plot_*(result, *, ax=None) -> Axes`, `draw_mol`, `view_mol_3d` |
| `prep/` | System preparation | `fix_pdb`, `strip_solvent`, `run_propka`, ligand tools |

Shell scripts (analysis wrappers, runtime helpers, build scripts, etc.) live in the top-level `scripts/` directory (not packaged).

Examples (notebooks and data) live in `examples/` (GROMACS, OpenFE RBFE, BrownDye).

### OpenFE Scripts (`scripts/openfe/`)

SLURM submission scripts for running OpenFE RBFE transformations on Sherlock.
**Requires OpenFE >= 1.10.0** for `--resume` checkpoint support.

| Script | Purpose |
|---|---|
| `quickrun/quickrun.sh` | Submit all `transformations/*.json` as SLURM array jobs (`-r N` for repeats) |
| `quickrun/quickrun.sbatch` | Batch script: starts CUDA MPS, runs `openfe quickrun --resume` via Apptainer |
| `runtime/check_status.sh` | Check transformation replica status and optionally restart failed replicas |
| `runtime/monitor.sbatch` | Periodic monitor: runs check_status, emails report, self-resubmits via SLURM |

- CUDA MPS is required for Sherlock's `Exclusive_Process` GPU mode (openmmtools needs multiple CUDA contexts).
- `--resume` enables checkpoint-based resumption after preemption on `owners` partition.
- Output goes to `results/<name>/replica_<id>/`.

Tests live in `tests/analysis/`, `tests/plots/`, and `tests/chem/`, mirroring the source tree.

## Mandatory Conventions

1. **Absolute imports only** — `from mdpp.core.trajectory import load_trajectory`.
1. **Google docstrings** — all public functions must have Args/Returns/Raises sections.
1. **Frozen dataclasses** — analysis results use `@dataclass(frozen=True, slots=True)`.
1. **Keyword-only args** — after the first positional arg in compute/plot functions.
1. **No builtin shadowing** — do not create modules named `io`, `pdb`, `types`, etc.
1. **Type aliases** — shared aliases live in `mdpp._types` (`StrPath`, `PathLike`).
1. **Exports** — every `__init__.py` has an `__all__` list. New public functions must be added.
1. **Units** — internal arrays use nm/ps (MDTraj convention); display properties convert to Å/ns.
1. **Chem functions** — take `Chem.rdchem.Mol` or SMILES strings; fingerprint generators are in `FP_GENERATORS` dict.
1. **3D visualization** — `plots/three_d.py` uses py3Dmol and nglview for notebook-based interactive views.

## File Naming

- Analysis modules: `src/mdpp/analysis/<topic>.py`
- Plot modules: `src/mdpp/plots/<topic>.py`
- Helper utilities within a subpackage: `utils.py`
- MDP config templates: `scripts/gromacs/mdps/<ff>/<step>.mdp`
- Shell scripts (not packaged): `scripts/<engine>/<category>/<script>.sh`
- SLURM scripts: `scripts/<engine>/<category>/<script>.sbatch`

## Adding a New Analysis

1. Create/extend a file in `src/mdpp/analysis/`.
1. Define result dataclass(es) with frozen=True, slots=True.
1. Write `compute_*` function following the existing signature pattern.
1. Add exports to `src/mdpp/analysis/__init__.py`.
1. If visual output makes sense, add `plot_*` in `src/mdpp/plots/` and export it.
1. Write tests in `tests/analysis/`.

## Adding a New Compute Backend

For existing multi-backend functions (e.g. `compute_rmsd_matrix`, pairwise distances):

1. Add the implementation in the matching `src/mdpp/analysis/_backends/_<kind>.py` file, matching the existing signature.
1. Use `require_torch()` / `require_jax()` / `require_cupy()` from `_backends/_imports.py` for optional GPU libraries -- never import them at module top-level.
1. Register in the module's `BackendRegistry` at the bottom of the file.
1. Add the backend name to the corresponding `Literal` alias (`DistanceBackend` / `RMSDBackend`) in `_backends/_registry.py`.
1. Add agreement tests in `tests/analysis/test_<kind>.py` guarded by the relevant `requires_*` skip marker.

## Adding a New Chem Function

1. Create/extend a file in `src/mdpp/chem/`.
1. Functions take `Chem.rdchem.Mol` or SMILES strings as input.
1. Add exports to `src/mdpp/chem/__init__.py`.
1. Write tests in `tests/chem/`.

## Important: Do Not

- Do not remove dependencies from `pyproject.toml` `[project.dependencies]`.
- Do not use relative imports.
- Do not put test `__init__.py` files in test directories.
- Do not modify files in `results/` (untracked temporary directory).
- Do not write custom parsers when a library exists (use panedr, MDAnalysis, etc.).
