#!/usr/bin/env bash

set -euo pipefail

PRODUCTION=step5_production

mkdir -p tmp

cp index.ndx tmp/
cp "${PRODUCTION}".gro "${PRODUCTION}".edr "${PRODUCTION}".tpr tmp/

ln -s "$(realpath "${PRODUCTION}.xtc")" tmp/

cd tmp

# gmx select \
#     -s "${PRODUCTION}.tpr" \
#     -on index.ndx \
#     -select '
#         "System" (group "System");
#         "SOLU" (group 1 or group 13 or group 14);  # choose the group you want to extract
#     '

# Select the protein chain A and its backbone using default topology groups,
# then append SOLU/SOLV/System from the CHARMM-GUI index
# (gmx select without -n sees default groups like Protein and Backbone)
gmx select \
    -s "${PRODUCTION}.tpr" \
    -on index_protein_chain_A.ndx \
    -select '"Chain_A" (group "Protein" and molecule 1)'
echo >>index_protein_chain_A.ndx
cat index.ndx >>index_protein_chain_A.ndx

# Create a solute-only TPR for use with the extracted complex trajectory
printf "SOLU\n" | gmx convert-tpr \
    -s "${PRODUCTION}.tpr" \
    -o "${PRODUCTION}_complex_fit.tpr" \
    -n index_protein_chain_A.ndx

# Regenerate index groups for the SOLU-only TPR (atom indices are renumbered)
gmx select \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -on index_complex.ndx \
    -select '
        "Chain_A_BB" (group "Backbone" and molecule 1);
        "System" group "System"
    '

# Center the trajectory
# Center on chain A and make each molecule whole. This does not guarantee that
# separate chains form the intended assembly; inspect multimer imaging and use
# an explicit assembly/cluster group beforehand when needed.
printf "Chain_A\nSOLU\n" | gmx trjconv \
    -s "${PRODUCTION}.tpr" \
    -f "${PRODUCTION}.xtc" \
    -o "${PRODUCTION}_complex_center.xtc" \
    -n index_protein_chain_A.ndx \
    -center \
    -pbc mol \
    -ur compact

# Extract the first frame
# We have to extract the first frame at this stage to preserve proper chain topology and visual integrity in PyMOL.
# Performing this step after complex extraction would result in broken chains and misrepresentation of the structure.
printf "Chain_A\nSOLU\n" | gmx trjconv \
    -s "${PRODUCTION}.tpr" \
    -f "${PRODUCTION}.xtc" \
    -o "${PRODUCTION}_complex_fit.pdb" \
    -n index_protein_chain_A.ndx \
    -center \
    -pbc mol \
    -ur compact \
    -dump 0

rm "${PRODUCTION}.xtc"

# Fit the trajectory
printf "Chain_A_BB\nSystem\n" | gmx trjconv \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_center.xtc" \
    -o "${PRODUCTION}_complex_fit.xtc" \
    -n index_complex.ndx \
    -fit rot+trans

# Retain the unfitted trajectory for analyses that apply periodic geometry.
# The fitted coordinates no longer have the same orientation as the box.

# Smooth for visualization only; never use this output for fluctuation,
# population, contact, or kinetic analysis.
gmx filter \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_fit.xtc" \
    -ol "${PRODUCTION}_complex_fit_smoothed.xtc" \
    -nf 10
