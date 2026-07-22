#!/usr/bin/env bash
#
# Launch a native Schrodinger FEP+ relative-binding job from a FEP map built by
# build_fepp_inputs.sh. One job per protein conformation (open / closed).
#
# Each map has ~12 nodes and a handful of edges; every edge runs a complex and
# a solvent leg as a lambda-replica-exchange (FEP/REST) simulation on the GPU,
# so a full run is hours-to-days of GPU time. Use --prepare first to validate
# that the map yields a runnable multisim system without launching production.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHRODINGER="${SCHRODINGER:-/apps/schrodinger2025-4}"
WORK_DIR="${SCRIPT_DIR}/tmp"

CONFORMATION="open"
HOST="localhost"
SUBHOST="localhost"
JOBNAME=""
PREPARE=0

usage() {
    cat <<EOF
Usage: ${0##*/} [options]

Launch (or prepare) a native FEP+ relative-binding job for one conformation.

Options:
  -c, --conformation NAME   Conformation to run (open | closed). Default: open.
  -H, --host HOST           Job-control main host. Default: ${HOST}.
  -S, --subhost SUBHOST     GPU subhost(s) for the FEP subjobs. Default: ${SUBHOST}.
  -j, --jobname NAME        Job name. Default: fepp_<conformation>.
  -p, --prepare             Build the multisim inputs only; do NOT run (dry run).
  -h, --help                Show this help and exit.

Environment:
  SCHRODINGER               Suite root. Default: ${SCHRODINGER}.

Examples:
  ${0##*/} -c open --prepare
  ${0##*/} -c closed -H localhost -S localhost
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c | --conformation)
            CONFORMATION="$2"
            shift 2
            ;;
        -H | --host)
            HOST="$2"
            shift 2
            ;;
        -S | --subhost)
            SUBHOST="$2"
            shift 2
            ;;
        -j | --jobname)
            JOBNAME="$2"
            shift 2
            ;;
        -p | --prepare)
            PREPARE=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "${CONFORMATION}" != "open" && "${CONFORMATION}" != "closed" ]]; then
    echo "ERROR: conformation must be 'open' or 'closed', got '${CONFORMATION}'" >&2
    exit 1
fi

map_file="${WORK_DIR}/${CONFORMATION}_map.fmp"
if [[ ! -f "${map_file}" ]]; then
    echo "ERROR: FEP map not found: ${map_file}" >&2
    echo "       Run ./build_fepp_inputs.sh -c ${CONFORMATION} first." >&2
    exit 1
fi
if [[ -z "${JOBNAME}" ]]; then
    JOBNAME="fepp_${CONFORMATION}"
fi

cd "${WORK_DIR}"

cmd=("${SCHRODINGER}/fep_plus"
    -HOST "${HOST}"
    -SUBHOST "${SUBHOST}"
    -JOBNAME "${JOBNAME}"
    "${map_file}")
if [[ "${PREPARE}" -eq 1 ]]; then
    cmd+=(-prepare)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
