#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Solvent accessible surface area (total and per-residue)
printf "Protein\n" | gmx sasa \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_fit.xtc" \
    -o "${PRODUCTION}_sasa.xvg" \
    -or "${PRODUCTION}_sasa_residue.xvg" \
    -tu ns
