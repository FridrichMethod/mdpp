#!/usr/bin/env bash

set -euo pipefail

PRODUCTION=step5_production

if [[ "$#" -eq 0 ]]; then
    echo "Usage: $(basename "$0") [gmx convert-tpr options, e.g. -extend 100000]"
    exit 1
fi

gmx convert-tpr \
    -s "${PRODUCTION}.tpr" \
    -o "${PRODUCTION}.tpr" \
    "$@"
