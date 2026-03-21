#!/bin/bash

set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: ${0##*/} [-j <n>] [-n] REMOTE_DIR LOCAL_DIR

Options:
    -j <n>   Number of parallel rsync jobs (default: 4)
    -n       Dry run (show what would be transferred)

Example:
    ${0##*/} -j 4 /scratch/users/\$USER/md_runs /data/local/md_runs
EOF
    exit 1
}

LOGIN_HOST="sherlock-plain"
DTN_HOST="sherlock-dtn"
JOBS=4
DRY_RUN=""

SSH_OPTS="ssh -T -c aes128-gcm@openssh.com -o Compression=no -o ServerAliveInterval=60"

while getopts "j:n" opt; do
    case $opt in
        j) JOBS="$OPTARG" ;;
        n) DRY_RUN="--dry-run" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[[ $# -eq 2 ]] || usage

REMOTE_DIR="$1"
LOCAL_DIR="$2"

mkdir -p "$LOCAL_DIR"

echo "=== Discovering subdirectories in $REMOTE_DIR ==="
mapfile -t SUBDIRS < <(
    ssh -n "$LOGIN_HOST" "find \"$REMOTE_DIR\" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'" | sort
)
[[ ${#SUBDIRS[@]} -gt 0 ]] || {
    echo "No subdirectories found in $REMOTE_DIR" >&2
    exit 1
}

echo "=== Found ${#SUBDIRS[@]} subdirectories, transferring with $JOBS parallel jobs ==="

LOGDIR="$LOCAL_DIR/.transfer_logs"
mkdir -p "$LOGDIR"

parallel -j "$JOBS" --bar --joblog "$LOGDIR/joblog.tsv" \
    rsync -ahP --append-verify $DRY_RUN \
    -e "'$SSH_OPTS'" \
    --log-file="'$LOGDIR/{}.log'" \
    "$DTN_HOST:$REMOTE_DIR/{}/" \
    "$LOCAL_DIR/{}/" \
    ::: "${SUBDIRS[@]}"

echo "=== Transfer complete. Logs in $LOGDIR ==="
