# Scripts

## MDP Templates

The GROMACS MDP templates live in the repository under `scripts/gromacs/mdps/`.
They are organized by force field:

```bash
# CHARMM-style templates
cp scripts/gromacs/mdps/charmm/*.mdp ./my_simulation/

# AMBER-style templates
cp scripts/gromacs/mdps/amber/*.mdp ./my_simulation/
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

All shell scripts live in the top-level `scripts/` directory and are **not** included in
the pip-installed package. Copy or symlink them into your MD working directories as
needed.

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
| `extend.sh` | Extend simulation time |
| `export.sh` | Export trajectory |

#### MD Run (`scripts/gromacs/mdrun/`)

| Script | Description |
|---|---|
| `mdprep.sh` | Minimization & equilibration pipeline |
| `mdrun.sh` | Production run (GPU-accelerated) |
| `rest2/rest2.sh` | REST2 topology setup |
| `rest2/mdrun_mpi_plumed.sh` | REST2 with PLUMED (multi-replica) |

#### Other Categories

- **`scripts/gromacs/mdps/`** -- force-field-specific MDP templates
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
