#!/usr/bin/env bash
# Generate APBS maps for the protein-only substrate body.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_ROOT="${TMP_ROOT:-$EXAMPLE_DIR/tmp}"

export INPUT_DIR="${INPUT_DIR:-$TMP_ROOT/substrate/pdb2pqr}"
export APBS_DIR="${APBS_DIR:-$TMP_ROOT/substrate/apbs}"
export PQR_STEMS="${PQR_STEMS:-substrate}"

bash "$EXAMPLE_DIR/_run_apbs_common.sh"
