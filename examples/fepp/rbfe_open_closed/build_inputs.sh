#!/usr/bin/env bash
# Build the matched open/closed input cohort for the paired-state example.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/../build_fepp_inputs.sh" "$@" -c both
