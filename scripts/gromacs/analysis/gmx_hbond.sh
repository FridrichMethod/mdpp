#!/usr/bin/env bash

set -euo pipefail

PRODUCTION=step5_production

# Periodic geometry requires unfitted coordinates matching the stored unit cell.
# Both postprocessing scripts retain this centered, whole-solute trajectory.

# Intra-protein hydrogen bonds (GROMACS 2024+ selection-based interface)
gmx hbond \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_center.xtc" \
    -num "${PRODUCTION}_hbond.xvg" \
    -r Protein \
    -t Protein \
    -tu ns
