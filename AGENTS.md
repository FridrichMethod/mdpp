# AGENTS.md

Instructions for AI coding agents (OpenAI Codex, etc.) working in this repository.

## What This Project Is

**mdpp** — a Python 3.12+ library for molecular dynamics simulation pre- and post-processing. Supports GROMACS, AMBER, and OpenFE workflows.

## Setup

```bash
conda create -n mdpp python=3.12 -y && conda activate mdpp
bash setup.sh
```

## Verification

```bash
pytest                             # run all tests
ruff check src/ tests/ --fix       # lint
ruff format src/ tests/            # format
pre-commit run --all-files         # full check suite
```

## Package Layout

Source is under `src/mdpp/` using the src-layout convention:

| Subpackage | Purpose | Key patterns |
|---|---|---|
| `core/` | Trajectory I/O, file parsers | `load_trajectory`, `read_xvg`, `read_edr` |
| `analysis/` | Compute functions | `compute_*(traj, *, ...) -> FrozenDataclass` |
| `plots/` | Visualization | `plot_*(result, *, ax=None, ...) -> Axes` |
| `prep/` | System preparation | `fix_pdb`, `strip_solvent`, ligand tools |
| `data/` | MDP config templates | `list_mdp_templates`, `get_mdp_template`, `copy_mdp_files` |
| `scripts/` | Bundled utility scripts | `list_scripts`, `get_script_path`, `copy_scripts` |

Workflow scripts that users copy to MD working directories live in the top-level `scripts/` directory (not packaged).

Tests live in `tests/analysis/`, `tests/plots/`, and `tests/scripts/`, mirroring the source tree.

## Mandatory Conventions

1. **Absolute imports only** — `from mdpp.core.trajectory import load_trajectory`.
1. **Google docstrings** — all public functions must have Args/Returns/Raises sections.
1. **Frozen dataclasses** — analysis results use `@dataclass(frozen=True, slots=True)`.
1. **Keyword-only args** — after the first positional arg in compute/plot functions.
1. **No builtin shadowing** — do not create modules named `io`, `pdb`, `types`, etc.
1. **Type aliases** — shared aliases live in `mdpp._types` (`StrPath`, `PathLike`).
1. **Exports** — every `__init__.py` has an `__all__` list. New public functions must be added.
1. **Units** — internal arrays use nm/ps (MDTraj convention); display properties convert to Å/ns.

## File Naming

- Analysis modules: `src/mdpp/analysis/<topic>.py`
- Plot modules: `src/mdpp/plots/<topic>.py`
- Helper utilities within a subpackage: `utils.py`
- Packaged utility scripts: `src/mdpp/scripts/<engine>/<category>/<script>.sh`
- MDP config templates: `src/mdpp/data/mdps/<step>.mdp`
- Workflow scripts (not packaged): `scripts/<engine>/<category>/<script>.sh`
- SLURM scripts: `scripts/<engine>/<category>/sherlock/<script>.sbatch`

## CLI

The `mdpp` CLI is installed via `pip install mdpp`:

```bash
mdpp list [prefix]            # list bundled utility scripts
mdpp show <path>              # print script content to stdout
mdpp copy <category> <dest>   # copy scripts to working directory
mdpp mdps <dest>              # copy MDP templates to working directory
```

## Adding a New Analysis

1. Create/extend a file in `src/mdpp/analysis/`.
1. Define result dataclass(es) with frozen=True, slots=True.
1. Write `compute_*` function following the existing signature pattern.
1. Add exports to `src/mdpp/analysis/__init__.py`.
1. If visual output makes sense, add `plot_*` in `src/mdpp/plots/` and export it.
1. Write tests in `tests/analysis/`.

## Important: Do Not

- Do not remove dependencies from `pyproject.toml` `[project.dependencies]`.
- Do not use relative imports.
- Do not put test `__init__.py` files in test directories.
- Do not modify files in `results/` (untracked temporary directory).
- Do not write custom parsers when a library exists (use panedr, MDAnalysis, etc.).
