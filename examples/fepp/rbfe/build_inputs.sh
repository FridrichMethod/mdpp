#!/usr/bin/env bash
# Build one receptor conformation for the standalone multi-ligand RBFE example.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFORMATION="${FEPP_CONFORMATION:-open}"

case "${CONFORMATION}" in
    open | closed) ;;
    *)
        echo "ERROR: FEPP_CONFORMATION must be open or closed; got '${CONFORMATION}'" >&2
        exit 1
        ;;
esac

exec "${SCRIPT_DIR}/../build_fepp_inputs.sh" "$@" -c "${CONFORMATION}"
