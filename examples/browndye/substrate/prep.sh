#!/usr/bin/env bash
# Prepare the protein-only substrate body with PDB2PQR.
#
# PDB2PQR 3.7.1 does not expose an ff19SB force-field option. The available
# protein-only APBS route is the PDB2PQR AMBER force field with PropKa pH
# assignment. The complex body still uses AmberTools/tleap ff19SB.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_ROOT="${TMP_ROOT:-$EXAMPLE_DIR/tmp}"
INPUT_PDB="${INPUT_PDB:-$EXAMPLE_DIR/substrate.pdb}"
STEP_DIR="${SUBSTRATE_PDB2PQR_DIR:-$TMP_ROOT/substrate/pdb2pqr}"
INTERMEDIATE_DIR="${INTERMEDIATE_DIR:-$STEP_DIR/intermediate}"
PH="${PH:-7.4}"
PDB2PQR_FF="${PDB2PQR_FF:-AMBER}"

require_file() {
    local path="$1"
    if [[ ! -s "$path" ]]; then
        printf 'Missing required file: %s\n' "$path" >&2
        exit 1
    fi
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$cmd" >&2
        exit 1
    fi
}

require_file "$INPUT_PDB"
require_cmd pdb2pqr

mkdir -p "$STEP_DIR" "$INTERMEDIATE_DIR"
cp "$INPUT_PDB" "$INTERMEDIATE_DIR/substrate.pdb"

echo "=== PDB2PQR substrate ==="
(
    cd "$INTERMEDIATE_DIR"
    pdb2pqr \
        --ff "$PDB2PQR_FF" \
        --ffout "$PDB2PQR_FF" \
        --keep-chain \
        --drop-water \
        --titration-state-method propka \
        --with-ph "$PH" \
        substrate.pdb \
        substrate.pqr 2>&1 | tee substrate.pdb2pqr.log
)

require_file "$INTERMEDIATE_DIR/substrate.pqr"
cp "$INTERMEDIATE_DIR/substrate.pqr" "$STEP_DIR/substrate.pqr"
cp "$INTERMEDIATE_DIR/substrate.pdb2pqr.log" "$STEP_DIR/substrate.pdb2pqr.log"

echo "=== Done ==="
ls -lh "$STEP_DIR/substrate.pqr"
