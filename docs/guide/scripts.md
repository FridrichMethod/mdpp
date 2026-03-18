# Scripts & Data

## MDP Templates

mdpp bundles GROMACS MDP parameter templates as package data, installed alongside the Python code via `pip install mdpp`.

### CLI Usage

```bash
# Copy MDP template files to a working directory
mdpp mdps ./my_simulation/
```

### Python API

```python
from mdpp.data import list_mdp_templates, get_mdp_template, copy_mdp_files

# List available MDP files
templates = list_mdp_templates()

# Read MDP content (accepts short name, with/without extension)
content = get_mdp_template("step5_production")

# Copy all MDP files to a directory
copy_mdp_files("./my_simulation/")
```

### Available Templates

Five standard GROMACS simulation stages:

| File | Stage |
|---|---|
| `step4.0_minimization.mdp` | Energy minimization |
| `step4.1_equilibration.mdp` | NVT equilibration |
| `step4.2_equilibration.mdp` | NPT equilibration |
| `step4.3_equilibration.mdp` | Final equilibration |
| `step5_production.mdp` | Production MD |

## Scripts

All shell scripts live in the top-level `scripts/` directory and are **not** included in the pip-installed package. Copy or symlink them into your MD working directories as needed.

### GROMACS

#### Analysis (`scripts/gromacs/analysis/`)

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

#### Runtime (`scripts/gromacs/runtime/`)

| Script | Description |
|---|---|
| `check_status.sh` | Monitor simulation job status |
| `restart.sh` | Restart from checkpoint |
| `mdextend.sh` | Extend simulation time |
| `mdexport.sh` | Export trajectory |

#### MD Run (`scripts/gromacs/mdrun/`)

| Script | Description |
|---|---|
| `mdprep.sh` | Minimization & equilibration pipeline |
| `mdrun.sh` | Production run (GPU-accelerated) |
| `rest2/rest2.sh` | REST2 topology setup |
| `rest2/mdrun_mpi_plumed.sh` | REST2 with PLUMED (multi-replica) |

#### Other Categories

- **`scripts/gromacs/compilation/`** -- GROMACS build scripts (generic + Sherlock HPC)
- **`scripts/gromacs/mdenv/`** -- Environment setup scripts (Sherlock)
- **`scripts/gromacs/postprocessing/`** -- Trajectory postprocessing
- **`scripts/gromacs/visualization/`** -- PyMOL movie generation

#### SLURM (Sherlock HPC)

SLURM batch scripts are in `sherlock/` subdirectories within each category:

- `scripts/gromacs/analysis/sherlock/`
- `scripts/gromacs/compilation/sherlock/`
- `scripts/gromacs/mdenv/sherlock/`
- `scripts/gromacs/mdrun/sherlock/`

### OpenFE

- `scripts/openfe/quickrun.sh` -- Batch submission wrapper for OpenFE transformations
- `scripts/openfe/quickrun.sbatch` -- Apptainer-based OpenFE execution on Sherlock
