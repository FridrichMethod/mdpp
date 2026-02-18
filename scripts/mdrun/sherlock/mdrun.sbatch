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
#SBATCH --constraint="GPU_GEN:VLT|GPU_GEN:AMP|GPU_GEN:ADA|GPU_GEN:HOP"
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

ml cuda/12.6.1

GMXRC="/home/groups/ayting/gromacs-2025.2/bin/GMXRC"
source "${GMXRC}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# If the checkpoint file for production exists, will skip the pre-processing step
# and continue from the checkpoint file
bash mdprep.sh
bash mdrun.sh
