#!/usr/bin/env bash

set -euo pipefail

PRODUCTION=step5_production

# Periodic geometry requires unfitted coordinates matching the stored unit cell.
# Both postprocessing scripts retain this centered, whole-solute trajectory.

# Secondary structure assignment (GROMACS 2024+ native DSSP)
gmx dssp \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_center.xtc" \
    -o "${PRODUCTION}_dssp.dat" \
    -num "${PRODUCTION}_dssp.xvg" \
    -sel Protein \
    -tu ns
