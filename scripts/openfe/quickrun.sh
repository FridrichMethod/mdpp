#!/bin/bash

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKING_DIR="."
N_REPLICAS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n | --n-replicas)
            N_REPLICAS=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

TRANSFORMATION_DIR="${WORKING_DIR}/transformations"
RESULTS_DIR="${WORKING_DIR}/results"

if [[ ! -d "${TRANSFORMATION_DIR}" ]]; then
    echo "Missing transformations dir: ${TRANSFORMATION_DIR}" >&2
    exit 1
fi

mkdir -p logs "${RESULTS_DIR}"

shopt -s nullglob

ARRAY_FLAG=(--array="0-$((N_REPLICAS - 1))")

submitted=0
for file in "${TRANSFORMATION_DIR}"/*.json; do
    sbatch "${ARRAY_FLAG[@]}" "${SCRIPTS_DIR}/quickrun.sbatch" "${file}" -o "${RESULTS_DIR}" -n "${N_REPLICAS}"
    submitted=$((submitted + 1))
done

if ((submitted == 0)); then
    echo "No .json files found in ${TRANSFORMATION_DIR}" >&2
    exit 1
fi
