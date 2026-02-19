#!/bin/bash

set -euo pipefail

# Center the trajectory
echo -e "Protein\nSystem\n" | gmx trjconv \
    -s step5_1.tpr \
    -f step5_1.xtc \
    -o step5_1_center.xtc \
    -center \
    -pbc mol \
    -ur compact

# Fit the trajectory
echo -e "Backbone\nSystem\n" | gmx trjconv \
    -s step5_1.tpr \
    -f step5_1_center.xtc \
    -o step5_1_fit.xtc \
    -fit rot+trans

# Generate index file for protein-ligand complex
gmx select \
    -s step5_1.tpr \
    -on index.ndx \
    -select '"Complex" group 1 or group 13'

# Extract the complex
echo "Complex" | gmx trjconv \
    -s step5_1.tpr \
    -f step5_1_fit.xtc \
    -o step5_1_complex_fit.xtc \
    -n index.ndx

# Extract the first frame
echo "Complex" | gmx trjconv \
    -s step5_1.tpr \
    -f step5_1_fit.xtc \
    -o step5_1_complex_fit.pdb \
    -n index.ndx \
    -dump 0
