#!/bin/bash
#SBATCH --job-name=mdrun
#SBATCH --output=mdrun_%j.out
#SBATCH --error=mdrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=btrippe,stat,bioe,gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="GPU_GEN:AMP|GPU_GEN:HPR|GPU_GEN:LOV|GPU_GEN:VLT"
#SBATCH --prefer="GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S"
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

ml gcc/12.4.0
ml cmake/3.31.4
ml make/4.4
ml cuda/12.6.1
ml openmpi/5.0.5
ml python/3.12.1

GMXRC="/home/groups/ayting/gromacs-2025.2/bin/GMXRC"
source "${GMXRC}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# If the checkpoint file for production exists, will skip the pre-processing step
# and continue from the checkpoint file
bash mdprep.sh && bash mdrun.sh
