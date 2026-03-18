#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Potential, kinetic, and total energy
printf "Potential\nKinetic-En.\nTotal-Energy\n\n" | gmx energy \
    -f "${PRODUCTION}".edr \
    -o "${PRODUCTION}_energy.xvg"

# Temperature
printf "Temperature\n\n" | gmx energy \
    -f "${PRODUCTION}".edr \
    -o "${PRODUCTION}_temperature.xvg"

# Pressure
printf "Pressure\n\n" | gmx energy \
    -f "${PRODUCTION}".edr \
    -o "${PRODUCTION}_pressure.xvg"

# Volume
printf "Volume\n\n" | gmx energy \
    -f "${PRODUCTION}".edr \
    -o "${PRODUCTION}_volume.xvg"

# Density
printf "Density\n\n" | gmx energy \
    -f "${PRODUCTION}".edr \
    -o "${PRODUCTION}_density.xvg"
