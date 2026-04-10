#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: ${0##*/} [-j N] [-n] [-h] REMOTE_DIR LOCAL_DIR

Options:
    -j, --jobs N    Number of parallel rsync jobs (default: number of subdirs)
    -n, --dry-run   Dry run (show what would be transferred)
    -h, --help      Show this help

Example:
    ${0##*/} -j 4 /scratch/users/\$USER/md_runs /data/local/md_runs
EOF
    exit 1
}

LOGIN_HOST="sherlock-plain"
DTN_HOST="sherlock-dtn"
JOBS=0
DRY_RUN=""

SSH_OPTS="ssh -T -c aes128-gcm@openssh.com -o Compression=no -o ServerAliveInterval=60"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j | --jobs)
            [[ $# -lt 2 ]] && {
                echo "Error: $1 requires an argument" >&2
                usage
            }
            [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
                echo "Error: -j requires a positive integer" >&2
                usage
            }
            JOBS="$2"
            shift 2
            ;;
        -n | --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        -h | --help) usage ;;
        -*)
            echo "Error: unknown option: $1" >&2
            usage
            ;;
        *) break ;;
    esac
done
[[ $# -eq 2 ]] || usage

REMOTE_DIR="$1"
LOCAL_DIR="$2"

if [[ ! "$REMOTE_DIR" =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
    echo "Error: REMOTE_DIR contains unsafe characters: $REMOTE_DIR" >&2
    exit 1
fi

mkdir -p "$LOCAL_DIR"

echo "=== Discovering subdirectories in $REMOTE_DIR ==="
mapfile -t SUBDIRS < <(
    ssh -n "$LOGIN_HOST" "find $(printf '%q' "$REMOTE_DIR") -mindepth 1 -maxdepth 1 -type d -printf '%f\n'" | sort
)
[[ ${#SUBDIRS[@]} -gt 0 ]] || {
    echo "No subdirectories found in $REMOTE_DIR" >&2
    exit 1
}

# Default to one job per subdirectory if -j was not specified.
[[ "$JOBS" -eq 0 ]] && JOBS=${#SUBDIRS[@]}

echo "=== Found ${#SUBDIRS[@]} subdirectories, transferring with $JOBS parallel jobs ==="

LOGDIR="$LOCAL_DIR/.transfer_logs"
mkdir -p "$LOGDIR"

RSYNC_OPTS=(-ahP --append-verify)
[[ -n "$DRY_RUN" ]] && RSYNC_OPTS+=("$DRY_RUN")

parallel -j "$JOBS" --bar --joblog "$LOGDIR/joblog.tsv" \
    rsync "${RSYNC_OPTS[@]}" \
    -e "'$SSH_OPTS'" \
    --log-file="'$LOGDIR/{}.log'" \
    "$DTN_HOST:$REMOTE_DIR/{}/" \
    "$LOCAL_DIR/{}/" \
    ::: "${SUBDIRS[@]}"

echo "=== Transfer complete. Logs in $LOGDIR ==="
