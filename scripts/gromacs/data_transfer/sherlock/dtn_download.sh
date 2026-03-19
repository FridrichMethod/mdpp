#!/bin/bash

set -euo pipefail

LOGIN_HOST="sherlock-plain"
DTN_HOST="sherlock-dtn"

REMOTE_DIR="/scratch/groups/ayting/zyli2002/turboid/md/replica2"
LOCAL_DIR="/data/data2/zyli2002/md/replica2"
JOBS=8

mkdir -p "$LOCAL_DIR"

mapfile -t SUBDIRS < <(
    ssh -n "$LOGIN_HOST" "find \"$REMOTE_DIR\" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'" | sort
)

parallel -j "$JOBS" \
    rsync -ahP --append-verify \
    "$DTN_HOST:$REMOTE_DIR/{}/" \
    "$LOCAL_DIR/{}/" \
    ::: "${SUBDIRS[@]}"
