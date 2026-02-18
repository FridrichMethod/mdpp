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

if ! command -v gracebat >/dev/null 2>&1; then
    echo "gracebat is not installed."
    exit 1
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

echo -e "Potential\n\n" | gmx energy \
    -f "${MINIMIZATION}".edr \
    -o "${MINIMIZATION}_potential.xvg"
gracebat \
    -nxy "${MINIMIZATION}_potential.xvg" \
    -hdevice PNG \
    -printfile "${MINIMIZATION}_potential.png"

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

echo -e "Temperature\n\n" | gmx energy \
    -f "${EQUILIBRATION_NVT}".edr \
    -o "${EQUILIBRATION_NVT}_temperature.xvg"
gracebat \
    -nxy "${EQUILIBRATION_NVT}_temperature.xvg" \
    -hdevice PNG \
    -printfile "${EQUILIBRATION_NVT}_temperature.png"

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

echo -e "Pressure\n\n" | gmx energy \
    -f "${EQUILIBRATION_NPT}".edr \
    -o "${EQUILIBRATION_NPT}_pressure.xvg"
gracebat \
    -nxy "${EQUILIBRATION_NPT}_pressure.xvg" \
    -hdevice PNG \
    -printfile "${EQUILIBRATION_NPT}_pressure.png"
echo -e "Density\n\n" | gmx energy \
    -f "${EQUILIBRATION_NPT}".edr \
    -o "${EQUILIBRATION_NPT}_density.xvg"
gracebat \
    -nxy "${EQUILIBRATION_NPT}_density.xvg" \
    -hdevice PNG \
    -printfile "${EQUILIBRATION_NPT}_density.png"

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
