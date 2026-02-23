#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Radius of gyration
printf "Protein\n" | gmx gyrate \
    -s "${PRODUCTION}_complex_fit.tpr" \
    -f "${PRODUCTION}_complex_fit.xtc" \
    -o "${PRODUCTION}_gyrate.xvg" \
    -tu ns

gracebat \
    -nxy "${PRODUCTION}_gyrate.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_gyrate.png"
