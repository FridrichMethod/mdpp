#!/usr/bin/env bash
# Launch both matched receptor conformations for the paired-state example.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/../run_fep_plus.sh" \
    --workflow paired \
    -c both \
    --jobname-base fepp_open_closed \
    "$@" \
    --workflow paired \
    -c both
