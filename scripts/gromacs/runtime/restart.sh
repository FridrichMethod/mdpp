#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

err() {
    printf '%bError:%b %s\n' "${RED}" "${NC}" "$*" >&2
    exit 1
}
info() { printf '%b[%s]%b %s\n' "${GREEN}" "$1" "${NC}" "$2"; }
warn() { printf '%bWarning:%b %s\n' "${YELLOW}" "${NC}" "$*" >&2; }

usage() {
    echo "Usage: $0 <slurm_job_id> [slurm_job_id ...]" >&2
    exit 1
}

[[ $# -ge 1 ]] || usage

for JOBID in "$@"; do
    [[ "$JOBID" =~ ^[0-9]+$ ]] || err "job id must be an integer, got: ${BOLD}$JOBID${NC}"
done

START_DIR="$(pwd)"

for JOBID in "$@"; do
    mapfile -d '' MATCHES < <(
        find . -type f -name "mdrun_${JOBID}.*" -print0
    )

    if [[ ${#MATCHES[@]} -eq 0 ]]; then
        err "no files matching ${BOLD}mdrun_${JOBID}.*${NC} found under ${BLUE}$START_DIR${NC}"
    fi

    declare -A DIRS=()
    for f in "${MATCHES[@]}"; do
        d="$(dirname "$f")"
        DIRS["$d"]=1
    done

    if [[ ${#DIRS[@]} -ne 1 ]]; then
        printf '%bError:%b found %bmdrun_%s.*%b in multiple directories:\n' "${RED}" "${NC}" "${BOLD}" "${JOBID}" "${NC}" >&2
        for d in "${!DIRS[@]}"; do
            echo "  $d" >&2
        done
        exit 1
    fi

    TARGET_DIR="${!DIRS[*]}"

    ERR_FILE="${TARGET_DIR}/mdrun_${JOBID}.err"
    OUT_FILE="${TARGET_DIR}/mdrun_${JOBID}.out"
    SBATCH_FILE="${TARGET_DIR}/mdrun.sbatch"

    if [[ ! -f "$ERR_FILE" || ! -f "$OUT_FILE" ]]; then
        err "expected both .err and .out in ${BLUE}$TARGET_DIR${NC}"
    fi

    if [[ ! -f "$SBATCH_FILE" ]]; then
        err "${BLUE}$SBATCH_FILE${NC} not found"
    fi

    info "$JOBID" "Found job files in: ${BLUE}$TARGET_DIR${NC}"
    info "$JOBID" "Submitting: ${BLUE}$SBATCH_FILE${NC}"

    sbatch --chdir "${TARGET_DIR}" "${SBATCH_FILE}"

    unset DIRS
done
