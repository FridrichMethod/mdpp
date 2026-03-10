#!/bin/bash

set -euo pipefail

PROJECT_ROOT_DIR=$(git rev-parse --show-toplevel)
WORKING_DIR="${PROJECT_ROOT_DIR}/results/1x2h"
SCRIPTS_DIR="${PROJECT_ROOT_DIR}/scripts"
BATCH_SIZE=2

TRANSFORMATION_DIR="${WORKING_DIR}/transformations"
RESULTS_DIR="${WORKING_DIR}/results"

if [[ ! -d "${TRANSFORMATION_DIR}" ]]; then
    echo "Missing transformations dir: ${TRANSFORMATION_DIR}" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

shopt -s nullglob

FILES=()
for file in "${TRANSFORMATION_DIR}"/*.json; do
    if ((BATCH_SIZE <= 0)); then
        FILES+=("${file}")
        continue
    fi

    FILES+=("${file}")
    if ((${#FILES[@]} >= BATCH_SIZE)); then
        sbatch "${SCRIPTS_DIR}/quickrun.sbatch" "${FILES[@]}" -o "${RESULTS_DIR}"
        FILES=()
    fi
done

if ((${#FILES[@]} > 0)); then
    sbatch "${SCRIPTS_DIR}/quickrun.sbatch" "${FILES[@]}" -o "${RESULTS_DIR}"
fi
