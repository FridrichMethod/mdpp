#!/usr/bin/env bash
# run_apbs.sh - generate APBS potential maps for BrownDye rigid bodies.
#
# Prerequisite:
#   bash run_ambertools.sh
#
# Usage:
#   conda activate ambertools
#   cd examples/browndye && bash run_apbs.sh
#
# By default this writes protein.in/protein.dx and ligand.in/ligand.dx. BrownDye
# expects one electrostatic map per independently moving rigid body, not a single
# map for the bound complex.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR/tmp}"
IONIC_STRENGTH="${IONIC_STRENGTH:-0.150}"
PQR_STEMS="${PQR_STEMS:-protein ligand}"
SOLUTE_DIELECTRIC="${SOLUTE_DIELECTRIC:-2.0}"
SOLVENT_DIELECTRIC="${SOLVENT_DIELECTRIC:-78.54}"
SOLVENT_RADIUS="${SOLVENT_RADIUS:-1.4}"
TEMPERATURE="${TEMPERATURE:-298.15}"
FINE_SPACING="${FINE_SPACING:-0.5}"
FINE_PADDING="${FINE_PADDING:-20.0}"
COARSE_PADDING="${COARSE_PADDING:-40.0}"

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

write_apbs_input() {
    local stem="$1"
    python3 - \
        "$stem" \
        "$IONIC_STRENGTH" \
        "$SOLUTE_DIELECTRIC" \
        "$SOLVENT_DIELECTRIC" \
        "$SOLVENT_RADIUS" \
        "$TEMPERATURE" \
        "$FINE_SPACING" \
        "$FINE_PADDING" \
        "$COARSE_PADDING" <<'PY'
from __future__ import annotations

from math import ceil
import sys

stem = sys.argv[1]
ionic_strength = float(sys.argv[2])
solute_dielectric = float(sys.argv[3])
solvent_dielectric = float(sys.argv[4])
solvent_radius = float(sys.argv[5])
temperature = float(sys.argv[6])
fine_spacing = float(sys.argv[7])
fine_padding = float(sys.argv[8])
coarse_padding = float(sys.argv[9])
pqr = f"{stem}.pqr"
apbs_input = f"{stem}.in"


def parse_pqr(path: str) -> tuple[list[float], list[float]]:
    coords: list[tuple[float, float, float]] = []
    radii: list[float] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            x, y, z = (float(value) for value in fields[-5:-2])
            radius = float(fields[-1])
            coords.append((x, y, z))
            radii.append(radius)
    if not coords:
        raise ValueError(f"No atoms found in {path}")

    lower = [min(coord[i] - radii[index] for index, coord in enumerate(coords)) for i in range(3)]
    upper = [max(coord[i] + radii[index] for index, coord in enumerate(coords)) for i in range(3)]
    return lower, upper


def apbs_friendly_dime(length: float) -> int:
    target = int(ceil(length / fine_spacing)) + 1
    candidates = sorted({c * 2**n + 1 for c in range(1, 7) for n in range(1, 12)})
    for candidate in candidates:
        if candidate >= target:
            return candidate
    return candidates[-1]


lower, upper = parse_pqr(pqr)
center = [(low + high) / 2.0 for low, high in zip(lower, upper, strict=True)]
span = [high - low for low, high in zip(lower, upper, strict=True)]
fglen = [value + fine_padding for value in span]
cglen = [max(value + coarse_padding, fine) for value, fine in zip(span, fglen, strict=True)]
dime = [apbs_friendly_dime(length) for length in fglen]

with open(apbs_input, "w", encoding="utf-8") as handle:
    handle.write(
        f"""read
    mol pqr {pqr}
end
elec
    mg-auto
    dime {dime[0]} {dime[1]} {dime[2]}
    cglen {cglen[0]:.4f} {cglen[1]:.4f} {cglen[2]:.4f}
    fglen {fglen[0]:.4f} {fglen[1]:.4f} {fglen[2]:.4f}
    cgcent {center[0]:.4f} {center[1]:.4f} {center[2]:.4f}
    fgcent {center[0]:.4f} {center[1]:.4f} {center[2]:.4f}
    mol 1
    lpbe
    bcfl sdh
    ion charge -1.00 conc {ionic_strength:.4f} radius 1.8150
    ion charge 1.00 conc {ionic_strength:.4f} radius 1.8750
    pdie {solute_dielectric:.4f}
    sdie {solvent_dielectric:.4f}
    srfm smol
    chgm spl2
    sdens 10.00
    srad {solvent_radius:.4f}
    swin 0.30
    temp {temperature:.2f}
    calcenergy total
    calcforce no
    write pot dx {stem}
end
print elecEnergy 1 end
quit
"""
    )
PY
}

run_apbs() {
    local stem="$1"
    echo "=== APBS: $stem ==="
    write_apbs_input "$stem"
    apbs "$stem.in" 2>&1 | tee "$stem.apbs.log"

    if [[ -s "$stem-PE0.dx" ]]; then
        mv "$stem-PE0.dx" "$stem.dx"
    elif [[ -s "$stem.pqr.dx" ]]; then
        mv "$stem.pqr.dx" "$stem.dx"
    fi

    require_file "$stem.dx"
}

require_cmd python3
require_cmd apbs

cd "$WORKDIR"
read -r -a stems <<<"$PQR_STEMS"
for stem in "${stems[@]}"; do
    require_file "$stem.pqr"
    run_apbs "$stem"
done

echo "=== Done ==="
ls -lh ./*.dx
