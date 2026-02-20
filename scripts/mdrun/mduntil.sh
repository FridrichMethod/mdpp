#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

# Truncate the simulation time
UNTIL_TIME=${1:?Usage: mduntil.sh <time_in_ps>} # in ps
gmx convert-tpr \
    -s "${PRODUCTION}".tpr \
    -until "${UNTIL_TIME}" \
    -o "${PRODUCTION}".tpr
