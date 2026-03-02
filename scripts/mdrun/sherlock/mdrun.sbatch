#!/bin/bash
#SBATCH --job-name=mdrun
#SBATCH --output=mdrun_%j.out
#SBATCH --error=mdrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=gpu,owners
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --constraint="GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:A40|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S"
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00

ml gcc/12.4.0
ml cmake/3.31.4
ml make/4.4
ml cuda/12.6.1
ml fftw/3.3.9
ml gsl/2.7
ml openblas/0.3.28
ml python/3.12.1

GMXRC="/home/groups/ayting/gromacs-2025.2/bin/GMXRC"
source "${GMXRC}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# If the checkpoint file for production exists, will skip the pre-processing step
# and continue from the checkpoint file
bash mdprep.sh && bash mdrun.sh
