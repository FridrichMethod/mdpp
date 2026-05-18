#!/usr/bin/env bash
# Generate APBS maps for the complex body.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_ROOT="${TMP_ROOT:-$EXAMPLE_DIR/tmp}"

export INPUT_DIR="${INPUT_DIR:-$TMP_ROOT/complex/ambertools}"
export APBS_DIR="${APBS_DIR:-$TMP_ROOT/complex/apbs}"
export PQR_STEMS="${PQR_STEMS:-complex}"

bash "$EXAMPLE_DIR/_run_apbs_common.sh"
