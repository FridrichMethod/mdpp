#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Extend the simulation time
EXTEND_TIME=${1:?Usage: mdextend.sh <time_in_ps>} # in ps
gmx convert-tpr \
    -s "${PRODUCTION}".tpr \
    -extend "${EXTEND_TIME}" \
    -o "${PRODUCTION}".tpr
