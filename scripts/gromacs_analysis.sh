#!/bin/bash


# Center the trajectory
echo -e "Protein\nSystem\n" | gmx trjconv -s step5_1.tpr -f step5_1.xtc -o step5_1_center.xtc -center -pbc mol -ur compact
# Fit the trajectory
echo -e "Backbone\nSystem\n" | gmx trjconv -s step5_1.tpr -f step5_1_center.xtc -o step5_1_fit.xtc -fit rot+trans

# Generate index file
gmx make_ndx -f step5_1.gro -o index.ndx <<EOF
1 | 13
q
EOF

# Extract the complex
echo 17 | gmx trjconv -s step5_1.tpr -f step5_1_fit.xtc -o step5_1_complex_fit.xtc -n index.ndx
# Extract the first frame
echo 17 | gmx trjconv -s step5_1.tpr -f step5_1_fit.xtc -o step5_1_complex_fit.pdb -n index.ndx -dump 0
