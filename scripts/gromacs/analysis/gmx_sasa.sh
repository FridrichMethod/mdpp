#!/usr/bin/env bash

set -euo pipefail

PRODUCTION=step5_production

# Periodic geometry requires unfitted coordinates matching the stored unit cell.
# Both postprocessing scripts retain this centered, whole-solute trajectory.

# Solvent accessible surface area (total and per-residue)
printf "Protein\n" | gmx sasa \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_center.xtc" \
    -o "${PRODUCTION}_sasa.xvg" \
    -or "${PRODUCTION}_sasa_residue.xvg" \
    -tu ns
