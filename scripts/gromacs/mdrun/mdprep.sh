#!/usr/bin/env bash

set -euo pipefail

INPUT=step3_input
MINIMIZATION=step4.0_minimization
EQUILIBRATION_NVT=step4.1_equilibration
EQUILIBRATION_NPT=step4.2_equilibration
EQUILIBRATION_NPT_NO_RESTRAINTS=step4.3_equilibration
PRODUCTION=step5_production

# Set the number of (thread) MPT explicitly to avoid conflicting demands
MDPREP_FLAGS=(
    -v
    -ntmpi 1
    -ntomp "${OMP_NUM_THREADS}"
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
    "${MDPREP_FLAGS[@]}"

printf "Potential\n\n" | gmx energy \
    -f "${MINIMIZATION}".edr \
    -o "${MINIMIZATION}_potential.xvg"

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
    "${MDPREP_FLAGS[@]}"

printf "Temperature\n\n" | gmx energy \
    -f "${EQUILIBRATION_NVT}".edr \
    -o "${EQUILIBRATION_NVT}_temperature.xvg"

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
    "${MDPREP_FLAGS[@]}"

printf "Pressure\n\n" | gmx energy \
    -f "${EQUILIBRATION_NPT}".edr \
    -o "${EQUILIBRATION_NPT}_pressure.xvg"
printf "Volume\n\n" | gmx energy \
    -f "${EQUILIBRATION_NPT}".edr \
    -o "${EQUILIBRATION_NPT}_volume.xvg"
printf "Density\n\n" | gmx energy \
    -f "${EQUILIBRATION_NPT}".edr \
    -o "${EQUILIBRATION_NPT}_density.xvg"

# Equilibration NPT no restraints
# Start from the checkpoint file of the NPT equilibration
# No restraints are applied in production
gmx grompp \
    -f "${EQUILIBRATION_NPT_NO_RESTRAINTS}".mdp \
    -o "${EQUILIBRATION_NPT_NO_RESTRAINTS}".tpr \
    -c "${EQUILIBRATION_NPT}".gro \
    -t "${EQUILIBRATION_NPT}".cpt \
    -p topol.top \
    -n index.ndx
gmx mdrun \
    -deffnm "${EQUILIBRATION_NPT_NO_RESTRAINTS}" \
    "${MDPREP_FLAGS[@]}"

printf "Pressure\n\n" | gmx energy \
    -f "${EQUILIBRATION_NPT_NO_RESTRAINTS}".edr \
    -o "${EQUILIBRATION_NPT_NO_RESTRAINTS}_pressure.xvg"
printf "Volume\n\n" | gmx energy \
    -f "${EQUILIBRATION_NPT_NO_RESTRAINTS}".edr \
    -o "${EQUILIBRATION_NPT_NO_RESTRAINTS}_volume.xvg"
printf "Density\n\n" | gmx energy \
    -f "${EQUILIBRATION_NPT_NO_RESTRAINTS}".edr \
    -o "${EQUILIBRATION_NPT_NO_RESTRAINTS}_density.xvg"

# Production
# Start from the checkpoint file of the NPT equilibration no restraints
# Use 4 fs timestep in production with HMR
gmx grompp \
    -f "${PRODUCTION}".mdp \
    -o "${PRODUCTION}".tpr \
    -c "${EQUILIBRATION_NPT_NO_RESTRAINTS}".gro \
    -t "${EQUILIBRATION_NPT_NO_RESTRAINTS}".cpt \
    -p topol.top \
    -n index.ndx
