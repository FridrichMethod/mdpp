#!/bin/bash

set -euo pipefail

ml gcc/12.4.0
ml cmake/3.31.4
ml make/4.4
ml cuda/12.6.1
ml openmpi/5.0.5
ml python/3.12.1

GMXRC="/home/groups/ayting/gromacs-2025.2/bin/GMXRC"
source "${GMXRC}"

bash gmx_rmsd.sh
bash gmx_rmsf.sh
bash gmx_gyrate.sh
bash gmx_sasa.sh
bash gmx_energy.sh
bash gmx_dssp.sh
bash gmx_hbond.sh
bash gmx_cluster.sh
