#!/bin/bash

set -euo pipefail

GRAY='\033[90m'
BLUE='\033[34m'
GREEN='\033[32m'
RED='\033[31m'
BOLD='\033[1m'
RESET='\033[0m'

MAX_JOBS=0

while getopts "j:" opt; do
    case "${opt}" in
        j) MAX_JOBS="${OPTARG}" ;;
        *)
            echo -e "${RED}Usage: $0 [-j max_parallel] <directory>${RESET}"
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -ne 1 ]]; then
    echo -e "${RED}Usage: $0 [-j max_parallel] <directory>${RESET}"
    echo -e "${RED}Example: $0 -j 4 results/md/replica2${RESET}"
    exit 1
fi

TARGET_DIR="$(realpath "$1")"

if [[ ! -d "${TARGET_DIR}" ]]; then
    echo -e "${RED}Error: ${TARGET_DIR} is not a directory${RESET}"
    exit 1
fi

PIDS=()
SUBDIRS=()
FAILED=()
RUNNING=0

for subdir in "${TARGET_DIR}"/*/; do
    [[ -d "${subdir}" ]] || continue
    compgen -G "${subdir}"/*.xtc >/dev/null || continue

    if [[ ${MAX_JOBS} -gt 0 && ${RUNNING} -ge ${MAX_JOBS} ]]; then
        wait -n || true
        RUNNING=$((RUNNING - 1))
    fi

    (
        cd "${subdir}"
        echo -e "${GRAY}[$(date '+%H:%M:%S')]${RESET} ${BLUE}Starting:${RESET} ${BOLD}$(basename "${subdir}")${RESET}"
        bash gmx_postprocessing.sh
        echo -e "${GRAY}[$(date '+%H:%M:%S')]${RESET} ${GREEN}Finished:${RESET} ${BOLD}$(basename "${subdir}")${RESET}"
    ) &

    PIDS+=($!)
    SUBDIRS+=("$(basename "${subdir}")")
    RUNNING=$((RUNNING + 1))
done

if [[ ${MAX_JOBS} -gt 0 ]]; then
    echo -e "${BLUE}Launched ${#PIDS[@]} jobs (max ${MAX_JOBS} parallel)${RESET}"
else
    echo -e "${BLUE}Launched ${#PIDS[@]} jobs in parallel${RESET}"
fi

for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        FAILED+=("${SUBDIRS[$i]}")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo -e "${RED}Failed jobs:${RESET}"
    for name in "${FAILED[@]}"; do
        echo -e "  ${RED}- ${name}${RESET}"
    done
    exit 1
fi

echo -e "${GREEN}All ${#PIDS[@]} jobs completed successfully${RESET}"
