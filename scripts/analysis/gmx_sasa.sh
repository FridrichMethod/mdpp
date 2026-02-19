#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Solvent accessible surface area (total and per-residue)
printf "Protein\n" | gmx sasa \
    -s "${PRODUCTION}_complex.tpr" \
    -f "${PRODUCTION}_complex.xtc" \
    -o "${PRODUCTION}_sasa.xvg" \
    -or "${PRODUCTION}_sasa_residue.xvg" \
    -tu ns

gracebat \
    -nxy "${PRODUCTION}_sasa.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_sasa.png"

gracebat \
    -nxy "${PRODUCTION}_sasa_residue.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_sasa_residue.png"
