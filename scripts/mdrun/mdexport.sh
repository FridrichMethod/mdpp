#!/bin/bash

set -euo pipefail

PRODUCTION=step5_production

SKIP=${1:?Usage: mdexport.sh <skip>}

mkdir -p tmp

cp "./*.xvg" tmp/
cp "${PRODUCTION}".gro "${PRODUCTION}".edr "${PRODUCTION}".tpr tmp/

printf "0\n" | gmx trjconv \
    -s "${PRODUCTION}".tpr \
    -f "${PRODUCTION}".xtc \
    -o "tmp/${PRODUCTION}.xtc" \
    -skip "${SKIP}"
