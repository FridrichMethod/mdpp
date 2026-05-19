#!/usr/bin/env bash
# bdrun.sh - run BrownDye and compute association-rate estimates.
#
# Prerequisite: run all cells of complex_pqr.ipynb (which writes
# tmp/bdprep/intermediate/${CORE0}_${CORE1}_simulation.xml).
#
# Layout:
#   tmp/bdrun/              main BrownDye simulation outputs
#   tmp/bdrun/intermediate/ transient BrownDye simulation working files
#
# Usage:
#   bash bdrun.sh
#   MODE=we bash bdrun.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT="${TMP_ROOT:-$SCRIPT_DIR/tmp}"
BDPREP_DIR="${BDPREP_DIR:-$TMP_ROOT/bdprep}"
BDPREP_INTERMEDIATE_DIR="${BDPREP_INTERMEDIATE_DIR:-$BDPREP_DIR/intermediate}"
STEP_DIR="${BDRUN_DIR:-$TMP_ROOT/bdrun}"
INTERMEDIATE_DIR="${INTERMEDIATE_DIR:-$STEP_DIR/intermediate}"
BD_HOME="${BD_HOME:-/apps/browndye2}"
BD_BIN="${BD_BIN:-$BD_HOME/bin}"
BD_AUX="${BD_AUX:-$BD_HOME/aux}"
export PATH="$BD_BIN:$BD_AUX:$PATH"

CORE0="${CORE0:-complex}"
CORE1="${CORE1:-substrate}"
SIMULATION_XML="${SIMULATION_XML:-${CORE0}_${CORE1}_simulation.xml}"
RESULTS_FILE="${RESULTS_FILE:-results.xml}"
MODE="${MODE:-nam}"

require_file() {
    local path="$1"
    if [[ ! -s "$path" ]]; then
        printf 'Missing required file: %s\n' "$path" >&2
        exit 1
    fi
}

resolve_tool() {
    local tool="$1"
    if [[ -x "$BD_BIN/$tool" ]]; then
        printf '%s\n' "$BD_BIN/$tool"
        return
    fi
    if [[ -x "$BD_AUX/$tool" ]]; then
        printf '%s\n' "$BD_AUX/$tool"
        return
    fi
    if command -v "$tool" >/dev/null 2>&1; then
        command -v "$tool"
        return
    fi

    printf 'Missing BrownDye tool: %s\n' "$tool" >&2
    printf 'Checked BD_BIN=%s, BD_AUX=%s, and PATH.\n' "$BD_BIN" "$BD_AUX" >&2
    exit 1
}

require_file "$BDPREP_INTERMEDIATE_DIR/$SIMULATION_XML"
mkdir -p "$STEP_DIR" "$INTERMEDIATE_DIR"
cp -a "$BDPREP_INTERMEDIATE_DIR"/. "$INTERMEDIATE_DIR"/
cd "$INTERMEDIATE_DIR"

case "$MODE" in
    nam)
        NAM_SIMULATION="$(resolve_tool nam_simulation)"
        COMPUTE_RATE_CONSTANT="$(resolve_tool compute_rate_constant)"
        echo "=== BrownDye NAM simulation ==="
        "$NAM_SIMULATION" "$SIMULATION_XML"
        require_file "$RESULTS_FILE"
        "$COMPUTE_RATE_CONSTANT" <"$RESULTS_FILE" | tee rate_constant.txt
        cp "$RESULTS_FILE" "$STEP_DIR/$RESULTS_FILE"
        cp rate_constant.txt "$STEP_DIR/rate_constant.txt"
        ;;
    we)
        BUILD_BINS="$(resolve_tool build_bins)"
        WE_SIMULATION="$(resolve_tool we_simulation)"
        COMPUTE_RATE_CONSTANT_WE="$(resolve_tool compute_rate_constant_we)"
        echo "=== BrownDye weighted-ensemble simulation ==="
        "$BUILD_BINS" "$SIMULATION_XML"
        "$WE_SIMULATION" "$SIMULATION_XML"
        require_file "$RESULTS_FILE"
        "$COMPUTE_RATE_CONSTANT_WE" <"$RESULTS_FILE" | tee rate_constant_we.txt
        cp "$RESULTS_FILE" "$STEP_DIR/$RESULTS_FILE"
        cp rate_constant_we.txt "$STEP_DIR/rate_constant_we.txt"
        ;;
    *)
        printf 'Unknown MODE: %s (expected nam or we)\n' "$MODE" >&2
        exit 1
        ;;
esac

echo "=== Done ==="
