#!/bin/bash
#SBATCH --job-name=mdrun
#SBATCH --output=mdrun_%j.out
#SBATCH --error=mdrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

ml gcc/12.4.0
ml cmake/3.31.4
ml make/4.4
ml cuda/12.6.1

source /home/groups/ayting/gromacs-2025.2/bin/GMXRC

# Minimization
# In the case that there is a problem during minimization using a single precision of GROMACS, please try to use
# a double precision of GROMACS only for the minimization step.
gmx grompp -f step4.0_minimization.mdp -o step4.0_minimization.tpr -c step3_input.gro -r step3_input.gro -p topol.top -n index.ndx
gmx mdrun -v -deffnm step4.0_minimization -ntmpi "$SLURM_NTASKS" -ntomp "$SLURM_CPUS_PER_TASK"

# Equilibration
gmx grompp -f step4.1_equilibration.mdp -o step4.1_equilibration.tpr -c step4.0_minimization.gro -r step3_input.gro -p topol.top -n index.ndx
gmx mdrun -v -deffnm step4.1_equilibration -ntmpi "$SLURM_NTASKS" -ntomp "$SLURM_CPUS_PER_TASK"

# Production
gmx grompp -f step5_production.mdp -o step5_1.tpr -c step4.1_equilibration.gro -p topol.top -n index.ndx
gmx mdrun -v -deffnm step5_1 -pin on -pmefft gpu -bonded gpu -pme gpu -nb gpu -update gpu

# For extending simulation by 200 ns
# gmx convert-tpr -s step5_1.tpr -extend 200000 -o step5_1.tpr

# If you want to restart the production run from a checkpoint file, uncomment the following line:
# gmx mdrun -v -deffnm step5_1 -pin on -pmefft gpu -bonded gpu -pme gpu -nb gpu -cpi step5_1.cpt
