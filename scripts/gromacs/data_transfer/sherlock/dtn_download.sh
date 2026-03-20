#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: ${0##*/} [-j <n>] REMOTE_DIR LOCAL_DIR" >&2
    exit 1
}

LOGIN_HOST="sherlock-plain"
DTN_HOST="sherlock-dtn"
JOBS=8

while getopts "j:" opt; do
    case $opt in
        j) JOBS="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[[ $# -eq 2 ]] || usage

REMOTE_DIR="$1"
LOCAL_DIR="$2"

mkdir -p "$LOCAL_DIR"

mapfile -t SUBDIRS < <(
    ssh -n "$LOGIN_HOST" "find \"$REMOTE_DIR\" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'" | sort
)

parallel -j "$JOBS" \
    rsync -ahP --append-verify \
    "$DTN_HOST:$REMOTE_DIR/{}/" \
    "$LOCAL_DIR/{}/" \
    ::: "${SUBDIRS[@]}"
