#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Set the number of (thread) MPT explicitly to avoid conflicting demands
MDRUN_FLAGS=(
    -v
    -ntmpi 1
    -ntomp "${OMP_NUM_THREADS}"
    -pin on # pin threads to cores
    -pmefft gpu
    -bonded cpu # reduce GPU state copy time
    -pme gpu
    -nb gpu
    -update gpu  # update will partially be done on CPUs by default
    -nstlist 100 # adjust larger for faster pair search
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
