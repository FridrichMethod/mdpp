#!/bin/bash

set -euo pipefail

# Center the trajectory
echo -e "Protein\nSystem\n" | gmx trjconv -s step5_1.tpr -f step5_1.xtc -o step5_1_center.xtc -center -pbc mol -ur compact
# Fit the trajectory
echo -e "Backbone\nSystem\n" | gmx trjconv -s step5_1.tpr -f step5_1_center.xtc -o step5_1_fit.xtc -fit rot+trans

# Extract the complex
echo "Protein" | gmx trjconv -s step5_1.tpr -f step5_1_fit.xtc -o step5_1_complex_fit.xtc
# Extract the first frame
echo "Protein" | gmx trjconv -s step5_1.tpr -f step5_1_fit.xtc -o step5_1_complex_fit.pdb -dump 0
