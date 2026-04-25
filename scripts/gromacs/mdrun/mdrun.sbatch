#!/usr/bin/env bash
#SBATCH --job-name=mdrun
#SBATCH --output=mdrun_%j.out
#SBATCH --error=mdrun_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${USER}@stanford.edu
#SBATCH --partition=gpu,owners
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=12G
#SBATCH --constraint="GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:A40|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S|GPU_SKU:V100_PCIE|GPU_SKU:V100S_PCIE|GPU_SKU:V100_SXM2"
#SBATCH --time=2-00:00:00

set -euo pipefail

ml gcc/12.4.0
ml cmake/3.31.4
ml make/4.4
ml cuda/12.6.1
ml fftw/3.3.9
ml gsl/2.7
ml openblas/0.3.28
ml python/3.12.1

GMX_VERSION="2025.2"
GMX_PREFIX="${GROUP_HOME}/gromacs-${GMX_VERSION}"

source "${GMX_PREFIX}/bin/GMXRC"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# If the checkpoint file for production exists, will skip the pre-processing step
# and continue from the checkpoint file
attempt=0
max_restarts=3

while true; do
    attempt=$((attempt + 1))
    status=0
    bash mdprep.sh && bash mdrun.sh || status=$?

    if [[ $status -eq 0 ]]; then
        exit 0
    elif [[ $status -eq 134 || $status -eq 139 ]]; then
        echo "mdrun.sh failed with exit code ${status} on attempt ${attempt} at $(date). Restarting..." >&2

        if [[ $attempt -lt $max_restarts ]]; then
            sleep 60
            continue
        else
            echo "mdrun.sh reached maximum restart attempts (${max_restarts}). Exiting with status ${status}." >&2
            exit "${status}"
        fi
    else
        exit "${status}"
    fi
done
