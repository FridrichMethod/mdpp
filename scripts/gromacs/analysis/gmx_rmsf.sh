#!/usr/bin/env bash

set -euo pipefail

PRODUCTION=step5_production

# Per-residue RMSF of backbone atoms
# -oq writes a PDB with RMSF as B-factors for visualization in PyMOL
printf "Backbone\n" | gmx rmsf \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_fit.xtc" \
    -o "${PRODUCTION}_rmsf.xvg" \
    -oq "${PRODUCTION}_rmsf.pdb" \
    -res
