# Scripts & Data

mdpp bundles GROMACS utility scripts and MDP parameter templates as package data. These are installed alongside the Python code via `pip install mdpp` and can be accessed programmatically or via the `mdpp` CLI.

## CLI Usage

```bash
# List all available utility scripts
mdpp list

# List scripts in a specific category
mdpp list gromacs/analysis

# View a script's content
mdpp show gromacs/runtime/restart.sh

# Copy scripts to a working directory
mdpp copy gromacs/analysis ./my_simulation/

# Copy MDP template files
mdpp mdps ./my_simulation/
```

## Python API

### Scripts

```python
from mdpp.scripts import list_scripts, get_script_path, read_script, copy_scripts

# Discover available scripts
scripts = list_scripts()                          # all scripts
scripts = list_scripts("gromacs/analysis")        # filter by category

# Get filesystem path to a script
path = get_script_path("gromacs/runtime/restart.sh")

# Read script content as a string
content = read_script("gromacs/runtime/restart.sh")

# Copy an entire category to a directory
copy_scripts("gromacs/analysis", "./my_simulation/")
```

### MDP Templates

```python
from mdpp.data import list_mdp_templates, get_mdp_template, copy_mdp_files

# List available MDP files
templates = list_mdp_templates()

# Read MDP content (accepts short name, with/without extension)
content = get_mdp_template("step5_production")

# Copy all MDP files to a directory
copy_mdp_files("./my_simulation/")
```

## Available Scripts

### Analysis (`gromacs/analysis/`)

| Script | Description |
|---|---|
| `gmx_rmsd.sh` | RMSD calculation |
| `gmx_rmsf.sh` | RMSF calculation |
| `gmx_dssp.sh` | Secondary structure (DSSP) |
| `gmx_hbond.sh` | Hydrogen bond analysis |
| `gmx_energy.sh` | Energy analysis |
| `gmx_gyrate.sh` | Radius of gyration |
| `gmx_sasa.sh` | Solvent accessible surface area |
| `gmx_cluster.sh` | Conformational clustering |

### Runtime (`gromacs/runtime/`)

| Script | Description |
|---|---|
| `check_status.sh` | Monitor simulation job status |
| `restart.sh` | Restart from checkpoint |
| `mdextend.sh` | Extend simulation time |
| `mdexport.sh` | Export trajectory |

### Other Categories

- **`gromacs/compilation/`** -- GROMACS build scripts (generic + Sherlock HPC)
- **`gromacs/mdenv/`** -- Environment setup scripts (Sherlock)
- **`gromacs/postprocessing/`** -- Trajectory postprocessing
- **`gromacs/visualization/`** -- PyMOL movie generation

## MDP Templates

Five standard GROMACS simulation stages:

| File | Stage |
|---|---|
| `step4.0_minimization.mdp` | Energy minimization |
| `step4.1_equilibration.mdp` | NVT equilibration |
| `step4.2_equilibration.mdp` | NPT equilibration |
| `step4.3_equilibration.mdp` | Final equilibration |
| `step5_production.mdp` | Production MD |

## Workflow Scripts (Not Packaged)

Scripts that are meant to be copied to MD working directories and run directly from the terminal live in the top-level `scripts/` directory and are **not** included in the pip-installed package:

- `scripts/gromacs/mdrun/` -- `mdprep.sh`, `mdrun.sh`, REST2 scripts, SLURM sbatch files
- `scripts/openfe/` -- OpenFE quickrun scripts
