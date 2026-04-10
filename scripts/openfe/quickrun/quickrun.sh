#!/usr/bin/env bash

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKING_DIR="."
REPEATS=1

usage() {
    cat >&2 <<'EOF'
Usage: quickrun.sh [-r N] [-h]

Submit OpenFE quickrun jobs for all transformations in ./transformations/.

Options:
    -r N    Number of repeats per transformation (default: 1)
    -h      Show this help
EOF
    exit 1
}

while getopts ":r:h" opt; do
    case "$opt" in
        r)
            REPEATS="$OPTARG"
            [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || {
                echo "Error: -r requires a positive integer, got: ${REPEATS}" >&2
                exit 1
            }
            ;;
        h) usage ;;
        \?)
            echo "Error: invalid option -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Error: option -$OPTARG requires an argument" >&2
            usage
            ;;
    esac
done
shift $((OPTIND - 1))

TRANSFORMATION_DIR="${WORKING_DIR}/transformations"
RESULTS_DIR="${WORKING_DIR}/results"

if [[ ! -d "${TRANSFORMATION_DIR}" ]]; then
    echo "Missing transformations dir: ${TRANSFORMATION_DIR}" >&2
    exit 1
fi

mkdir -p logs "${RESULTS_DIR}"

shopt -s nullglob

ARRAY_FLAG=(--array="0-$((REPEATS - 1))")

submitted=0
for file in "${TRANSFORMATION_DIR}"/*.json; do
    sbatch --chdir "$(pwd -P)" "${ARRAY_FLAG[@]}" "${SCRIPTS_DIR}/quickrun.sbatch" "${file}" -o "${RESULTS_DIR}"
    submitted=$((submitted + 1))
done

if ((submitted == 0)); then
    echo "No .json files found in ${TRANSFORMATION_DIR}" >&2
    exit 1
fi
