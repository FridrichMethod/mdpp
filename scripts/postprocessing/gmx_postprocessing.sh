#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Select the protein chain A and its backbone
# CHARMM-GUI predefines the protein chain A as molecule 1
gmx select \
    -s "${PRODUCTION}".tpr \
    -on index_protein_chain_A.ndx \
    -select '
        "System" group "System";
        "Chain_A" (group "Protein" and molecule 1);
        "Chain_A_BB" (group "Backbone" and molecule 1)
    '

# Center the trajectory
# The trajectory is centered on protein chain A to maintain the integrity of the multimeric complex
# This prevents the protein assembly from being artificially split by periodic boundary conditions
# See https://gromacs.bioexcel.eu/t/protein-come-out-of-the-box-but-trjconv-command-didnt-fix-the-problem/3851/8
printf "Chain_A\nSystem\n" | gmx trjconv \
    -s "${PRODUCTION}".tpr \
    -f "${PRODUCTION}".xtc \
    -o "${PRODUCTION}_center.xtc" \
    -n index_protein_chain_A.ndx \
    -center \
    -pbc mol \
    -ur compact

# Fit the trajectory
printf "Chain_A_BB\nSystem\n" | gmx trjconv \
    -s "${PRODUCTION}".tpr \
    -f "${PRODUCTION}_center.xtc" \
    -o "${PRODUCTION}_fit.xtc" \
    -n index_protein_chain_A.ndx \
    -fit rot+trans

# Extract the complex (solute)
printf "SOLU\n" | gmx trjconv \
    -s "${PRODUCTION}".tpr \
    -f "${PRODUCTION}_fit.xtc" \
    -o "${PRODUCTION}_complex_fit.xtc" \
    -n index.ndx

# Extract the first frame
printf "SOLU\n" | gmx trjconv \
    -s "${PRODUCTION}".tpr \
    -f "${PRODUCTION}_fit.xtc" \
    -o "${PRODUCTION}_complex_fit.pdb" \
    -n index.ndx \
    -dump 0

# Create a solute-only TPR for use with the extracted complex trajectory
printf "SOLU\n" | gmx convert-tpr \
    -s "${PRODUCTION}".tpr \
    -o "${PRODUCTION}_complex_fit.tpr" \
    -n index.ndx
