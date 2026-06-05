#!/usr/bin/env bash

set -euo pipefail

GRAY='\033[90m'
BLUE='\033[34m'
GREEN='\033[32m'
RED='\033[31m'
BOLD='\033[1m'
RESET='\033[0m'

PRODUCTION=step5_production

usage() {
    printf '%bUsage: %s [-j N] <directory>%b\n' "${RED}" "$0" "${RESET}" >&2
    printf '%b  -j, --jobs N  Limit to N parallel jobs (default: unlimited)%b\n' "${RED}" "${RESET}" >&2
    printf '%b  -h, --help    Show this help%b\n' "${RED}" "${RESET}" >&2
    printf '%b\nRecursively finds simulation directories containing %s.tpr under <directory>.%b\n' "${RED}" "${PRODUCTION}" "${RESET}" >&2
    printf '%bExample: %s -j 4 results/md/replica2%b\n' "${RED}" "$0" "${RESET}" >&2
    exit 1
}

# Max parallel jobs. 0 = unlimited (all subdirectories launch simultaneously).
MAX_JOBS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j | --jobs)
            [[ $# -lt 2 ]] && {
                printf '%bError: %s requires an argument%b\n' "${RED}" "$1" "${RESET}" >&2
                usage
            }
            [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
                printf '%bError: -j requires a positive integer%b\n' "${RED}" "${RESET}" >&2
                exit 1
            }
            MAX_JOBS="$2"
            shift 2
            ;;
        -h | --help) usage ;;
        -*)
            printf '%bError: unknown option: %s%b\n' "${RED}" "$1" "${RESET}" >&2
            usage
            ;;
        *) break ;;
    esac
done

[[ $# -eq 1 ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$(realpath "$1")"

if [[ ! -d "${TARGET_DIR}" ]]; then
    printf '%bError: %s is not a directory%b\n' "${RED}" "${TARGET_DIR}" "${RESET}" >&2
    exit 1
fi

PIDS=()
SUBDIRS=()
FAILED=()
RUNNING=0
NEXT_WAIT=0
declare -A WAITED
declare -A PID_NAME

# Reap the oldest still-running launched job (FIFO) and record its status.
# Uses `wait <pid>` rather than `wait -n -p VAR`, which is bash >= 5.1 only and
# silently breaks the throttle on RHEL8-class bash 4.4 found on many HPC nodes.
# FIFO reaping is sufficient to enforce the -j concurrency cap.
reap_one() {
    local pid="${PIDS[${NEXT_WAIT}]}"
    if ! wait "${pid}"; then
        FAILED+=("${PID_NAME[${pid}]}")
    fi
    WAITED[${pid}]=1
    NEXT_WAIT=$((NEXT_WAIT + 1))
    RUNNING=$((RUNNING - 1))
}

# Discover simulation directories by locating ${PRODUCTION}.tpr recursively.
mapfile -t SIM_DIRS < <(find "${TARGET_DIR}" -name "${PRODUCTION}.tpr" -type f -printf '%h\n' | sort -u)

for subdir in "${SIM_DIRS[@]}"; do
    [[ -d "${subdir}" ]] || continue

    while [[ ${MAX_JOBS} -gt 0 && ${RUNNING} -ge ${MAX_JOBS} ]]; do
        reap_one
    done

    label="${subdir#"${TARGET_DIR}"/}"

    (
        cd "${subdir}"
        printf '%b[%s]%b %bStarting:%b %b%s%b\n' "${GRAY}" "$(date '+%H:%M:%S')" "${RESET}" "${BLUE}" "${RESET}" "${BOLD}" "${label}" "${RESET}"
        bash "${SCRIPT_DIR}/gmx_postprocessing_fast.sh"
        printf '%b[%s]%b %bFinished:%b %b%s%b\n' "${GRAY}" "$(date '+%H:%M:%S')" "${RESET}" "${GREEN}" "${RESET}" "${BOLD}" "${label}" "${RESET}"
    ) &

    PIDS+=($!)
    SUBDIRS+=("${label}")
    PID_NAME[$!]="${label}"

    RUNNING=$((RUNNING + 1))
done

for i in "${!PIDS[@]}"; do
    [[ -n "${WAITED[${PIDS[$i]}]+x}" ]] && continue
    if ! wait "${PIDS[$i]}"; then
        FAILED+=("${SUBDIRS[$i]}")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf '%bFailed jobs:%b\n' "${RED}" "${RESET}"
    for name in "${FAILED[@]}"; do
        printf '  %b- %s%b\n' "${RED}" "${name}" "${RESET}"
    done
    exit 1
fi

printf '%bAll %d jobs completed successfully%b\n' "${GREEN}" "${#PIDS[@]}" "${RESET}"
