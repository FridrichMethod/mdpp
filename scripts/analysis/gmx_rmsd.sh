#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Backbone RMSD (least-squares fit on backbone, RMSD of backbone)
printf "Backbone\nBackbone\n" | gmx rms \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_fit.xtc" \
    -o "${PRODUCTION}_rmsd.xvg" \
    -tu ns

gracebat \
    -nxy "${PRODUCTION}_rmsd.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_rmsd.png"
