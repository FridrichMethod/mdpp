#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Potential, kinetic, and total energy
printf "Potential\nKinetic-En.\nTotal-Energy\n\n" | gmx energy \
    -f "${PRODUCTION}".edr \
    -o "${PRODUCTION}_energy.xvg" \
    -tu ns

gracebat \
    -nxy "${PRODUCTION}_energy.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_energy.png"

# Temperature
printf "Temperature\n\n" | gmx energy \
    -f "${PRODUCTION}".edr \
    -o "${PRODUCTION}_temperature.xvg" \
    -tu ns

gracebat \
    -nxy "${PRODUCTION}_temperature.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_temperature.png"

# Pressure
printf "Pressure\n\n" | gmx energy \
    -f "${PRODUCTION}".edr \
    -o "${PRODUCTION}_pressure.xvg" \
    -tu ns

gracebat \
    -nxy "${PRODUCTION}_pressure.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_pressure.png"

# Volume
printf "Volume\n\n" | gmx energy \
    -f "${PRODUCTION}".edr \
    -o "${PRODUCTION}_volume.xvg" \
    -tu ns

gracebat \
    -nxy "${PRODUCTION}_volume.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_volume.png"

# Density
printf "Density\n\n" | gmx energy \
    -f "${PRODUCTION}".edr \
    -o "${PRODUCTION}_density.xvg" \
    -tu ns

gracebat \
    -nxy "${PRODUCTION}_density.xvg" \
    -hdevice PNG \
    -printfile "${PRODUCTION}_density.png"
