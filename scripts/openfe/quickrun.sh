#!/bin/bash

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKING_DIR="."
BATCH_SIZE=1

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
