#!/bin/bash

set -euo pipefail

INPUT=step3_input
MINIMIZATION=step4.0_minimization
EQUILIBRATION_NVT=step4.1_equilibration
EQUILIBRATION_NPT=step4.2_equilibration
PRODUCTION=step5_production

MDRUN_FLAGS=(
    -v
    -pin on
    -pmefft gpu
    -bonded gpu
    -pme gpu
    -nb gpu
    -update gpu
)

if [[ -s "${PRODUCTION}".cpt ]]; then
    echo "Checkpoint file for production exists. Skipping pre-processing step."
    exit 0
fi

# Minimization
gmx grompp \
    -f "${MINIMIZATION}".mdp \
    -o "${MINIMIZATION}".tpr \
    -c "${INPUT}".gro \
    -r "${INPUT}".gro \
    -p topol.top \
    -n index.ndx
gmx mdrun \
    -deffnm "${MINIMIZATION}" \
    "${MDRUN_FLAGS[@]}"

# Equilibration NVT
gmx grompp \
    -f "${EQUILIBRATION_NVT}".mdp \
    -o "${EQUILIBRATION_NVT}".tpr \
    -c "${MINIMIZATION}".gro \
    -r "${MINIMIZATION}".gro \
    -p topol.top \
    -n index.ndx
gmx mdrun \
    -deffnm "${EQUILIBRATION_NVT}" \
    "${MDRUN_FLAGS[@]}"

# Equilibration NPT
# Start from the checkpoint file of the NVT equilibration
gmx grompp \
    -f "${EQUILIBRATION_NPT}".mdp \
    -o "${EQUILIBRATION_NPT}".tpr \
    -c "${EQUILIBRATION_NVT}".gro \
    -r "${EQUILIBRATION_NVT}".gro \
    -t "${EQUILIBRATION_NVT}".cpt \
    -p topol.top \
    -n index.ndx
gmx mdrun \
    -deffnm "${EQUILIBRATION_NPT}" \
    "${MDRUN_FLAGS[@]}"

# Production
# Start from the checkpoint file of the NPT equilibration
# No restraints are applied in production
gmx grompp \
    -f "${PRODUCTION}".mdp \
    -o "${PRODUCTION}".tpr \
    -c "${EQUILIBRATION_NPT}".gro \
    -t "${EQUILIBRATION_NPT}".cpt \
    -p topol.top \
    -n index.ndx
