#!/usr/bin/env bash
# bdrun.sh - run BrownDye and compute association-rate estimates.
#
# Prerequisite:
#   cd examples/browndye
#   bash bdprep.sh
#
# Usage:
#   bash bdrun.sh
#   MODE=we bash bdrun.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR/tmp}"
BD_HOME="${BD_HOME:-/apps/browndye2}"
BD_BIN="${BD_BIN:-$BD_HOME/bin}"
BD_AUX="${BD_AUX:-$BD_HOME/aux}"

CORE0="${CORE0:-protein}"
CORE1="${CORE1:-ligand}"
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

cd "$WORKDIR"
require_file "$SIMULATION_XML"

case "$MODE" in
    nam)
        NAM_SIMULATION="$(resolve_tool nam_simulation)"
        COMPUTE_RATE_CONSTANT="$(resolve_tool compute_rate_constant)"
        echo "=== BrownDye NAM simulation ==="
        "$NAM_SIMULATION" "$SIMULATION_XML"
        require_file "$RESULTS_FILE"
        "$COMPUTE_RATE_CONSTANT" <"$RESULTS_FILE" | tee rate_constant.txt
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
        ;;
    *)
        printf 'Unknown MODE: %s (expected nam or we)\n' "$MODE" >&2
        exit 1
        ;;
esac

echo "=== Done ==="
