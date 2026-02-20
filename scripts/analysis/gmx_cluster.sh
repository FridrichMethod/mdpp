#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Cluster the trajectory using GROMOS clustering algorithm
printf "Backbone\nSystem\n" | gmx cluster \
    -s "${PRODUCTION}_complex.tpr" \
    -f "${PRODUCTION}_complex.xtc" \
    -cl "${PRODUCTION}_cluster.pdb" \
    -clid "${PRODUCTION}_clid.xvg" \
    -dist "${PRODUCTION}_dist.xvg" \
    -sz "${PRODUCTION}_size.xvg" \
    -method gromos \
    -b 0 \
    -cutoff 0.2

gracebat \
    -nxy "${PRODUCTION}_clid.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_clid.png"

gracebat \
    -nxy "${PRODUCTION}_dist.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_dist.png"

gracebat \
    -nxy "${PRODUCTION}_size.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_size.png"
