#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Intra-protein hydrogen bonds (GROMACS 2024+ selection-based interface)
gmx hbond \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_fit.xtc" \
    -num "${PRODUCTION}_hbond.xvg" \
    -r Protein \
    -t Protein \
    -tu ns

gracebat \
    -nxy "${PRODUCTION}_hbond.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_hbond.png"
