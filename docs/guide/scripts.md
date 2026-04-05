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
- **`scripts/gromacs/data_transfer/`** -- DTN download scripts (Sherlock)
- **`scripts/gromacs/postprocessing/`** -- Trajectory postprocessing
- **`scripts/gromacs/visualization/`** -- PyMOL movie generation

#### SLURM (Sherlock HPC)

SLURM batch scripts (`.sbatch`) live alongside their `.sh` counterparts in each category directory.

### OpenFE

**Requires OpenFE >= 1.10.0** for checkpoint-based resumption (`--resume`).

| Script | Description |
|---|---|
| `quickrun/quickrun.sh` | Submit all `transformations/*.json` as SLURM array jobs |
| `quickrun/quickrun.sbatch` | SLURM batch script: starts CUDA MPS, runs `openfe quickrun --resume` via Apptainer |
| `runtime/check_status.sh` | Check transformation replica status and optionally restart failed replicas |
| `runtime/monitor.sbatch` | Periodic monitor: runs check_status, emails report, self-resubmits via SLURM |

#### Quick start

```bash
# Copy scripts to your working directory (alongside transformations/)
cp -r scripts/openfe/{quickrun,runtime} /path/to/workdir/
cd /path/to/workdir/

# Submit with 3 independent repeats per transformation
./quickrun/quickrun.sh -r 3

# Check status of all transformations
./runtime/check_status.sh

# Start periodic monitoring (emails reports, restarts failures, self-resubmits)
sbatch runtime/monitor.sbatch -d /path/to/workdir -e $USER@stanford.edu
```

#### Output structure

```
results/<transformation_name>/replica_0/
results/<transformation_name>/replica_1/
results/<transformation_name>/replica_2/
```

#### Preemption handling

Jobs on the `owners` partition are automatically requeued when preempted.
`quickrun.sbatch` uses `--resume` so requeued jobs continue from the last
checkpoint instead of starting over. CUDA MPS is started automatically to
work around Sherlock's `Exclusive_Process` GPU mode.
