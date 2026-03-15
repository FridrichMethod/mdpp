#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <slurm_job_id> [slurm_job_id ...]" >&2
    exit 1
}

[[ $# -ge 1 ]] || usage

for JOBID in "$@"; do
    if [[ ! "$JOBID" =~ ^[0-9]+$ ]]; then
        echo "Error: job id must be an integer, got: $JOBID" >&2
        exit 1
    fi
done

START_DIR="$(pwd)"

for JOBID in "$@"; do
    mapfile -d '' MATCHES < <(
        find . -type f -name "mdrun_${JOBID}.*" -print0
    )

    if [[ ${#MATCHES[@]} -eq 0 ]]; then
        echo "Error: no files matching mdrun_${JOBID}.* found under $START_DIR" >&2
        exit 1
    fi

    declare -A DIRS=()
    for f in "${MATCHES[@]}"; do
        d="$(dirname "$f")"
        DIRS["$d"]=1
    done

    if [[ ${#DIRS[@]} -ne 1 ]]; then
        echo "Error: found mdrun_${JOBID}.* in multiple directories:" >&2
        for d in "${!DIRS[@]}"; do
            echo "  $d" >&2
        done
        exit 1
    fi

    TARGET_DIR="${!DIRS[@]}"

    ERR_FILE="${TARGET_DIR}/mdrun_${JOBID}.err"
    OUT_FILE="${TARGET_DIR}/mdrun_${JOBID}.out"
    SBATCH_FILE="${TARGET_DIR}/mdrun.sbatch"

    if [[ ! -f "$ERR_FILE" || ! -f "$OUT_FILE" ]]; then
        echo "Error: expected both files in $TARGET_DIR:" >&2
        echo "  $ERR_FILE" >&2
        echo "  $OUT_FILE" >&2
        exit 1
    fi

    if [[ ! -f "$SBATCH_FILE" ]]; then
        echo "Error: $SBATCH_FILE not found" >&2
        exit 1
    fi

    echo "[$JOBID] Found job files in: $TARGET_DIR"
    echo "[$JOBID] Submitting: $SBATCH_FILE"

    cd "$TARGET_DIR" && sbatch mdrun.sbatch && cd "$START_DIR"

    unset DIRS
done

echo "Returned to: $START_DIR"
