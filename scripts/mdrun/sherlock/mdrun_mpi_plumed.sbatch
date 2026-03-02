#!/bin/bash
#SBATCH --job-name=mdrun
#SBATCH --output=mdrun_%j.out
#SBATCH --error=mdrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=btrippe,stat,bioe
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --constraint="GPU_GEN:AMP|GPU_GEN:HPR|GPU_GEN:LOV"
#SBATCH --prefer="GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S"
#SBATCH --mem-per-gpu=16G
#SBATCH --time=7-00:00:00

ml gcc/12.4.0
ml cmake/3.31.4
ml make/4.4
ml cuda/12.6.1
ml openmpi/5.0.5
ml fftw/3.3.9
ml gsl/2.7
ml openblas/0.3.28
ml ucx/1.17.0
ml python/3.12.1

PLUMED_PREFIX="/home/groups/ayting/plumed-2.10.0"
export PLUMED_KERNEL="${PLUMED_PREFIX}/lib/libplumedKernel.so"
export PATH="$PLUMED_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PLUMED_PREFIX/lib:$LD_LIBRARY_PATH"

GMXRC="/home/groups/ayting/gromacs-2025.0/bin/GMXRC"
source "${GMXRC}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# If the checkpoint file for production exists, will skip the pre-processing step
# and continue from the checkpoint file
bash mdprep.sh && bash mdrun_mpi_plumed.sh
