#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Cluster the trajectory using GROMOS clustering algorithm
printf "Backbone\nSystem\n" | gmx cluster \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_fit.xtc" \
    -cl "${PRODUCTION}_cluster.pdb" \
    -clid "${PRODUCTION}_clid.xvg" \
    -dist "${PRODUCTION}_dist.xvg" \
    -sz "${PRODUCTION}_size.xvg" \
    -method gromos \
    -b 0 \
    -cutoff 0.2
