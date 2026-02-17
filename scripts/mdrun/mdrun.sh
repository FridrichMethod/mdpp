#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# -ntmpi 1 -ntomp ${OMP_NUM_THREADS} are set automatically by the scheduler
MDRUN_FLAGS=(
    -v
    -pin on # pin threads to cores
    -pmefft gpu
    -bonded gpu
    -pme gpu
    -nb gpu
    -update gpu # update will partially be done on CPUs by default
)

# If the checkpoint file exists, restart the production run from the checkpoint file
if [[ -s "${PRODUCTION}".cpt ]]; then
    gmx mdrun \
        -deffnm "${PRODUCTION}" \
        -cpi "${PRODUCTION}".cpt \
        "${MDRUN_FLAGS[@]}"
else
    gmx mdrun \
        -deffnm "${PRODUCTION}" \
        "${MDRUN_FLAGS[@]}"
fi
