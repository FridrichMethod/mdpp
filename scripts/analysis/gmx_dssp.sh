#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Secondary structure assignment (GROMACS 2024+ native DSSP)
gmx dssp \
    -s "${PRODUCTION}_complex.tpr" \
    -f "${PRODUCTION}_complex.xtc" \
    -o "${PRODUCTION}_dssp.dat" \
    -num "${PRODUCTION}_dssp.xvg" \
    -sel Protein \
    -tu ns

gracebat \
    -nxy "${PRODUCTION}_dssp.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_dssp.png"
