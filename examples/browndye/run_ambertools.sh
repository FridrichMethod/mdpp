#!/usr/bin/env bash
# run_ambertools.sh - AMBER/GAFF2 parameterization for BrownDye/APBS inputs.
#
# Prerequisite: run complex_pqr.ipynb first to produce protein_fixed.pdb and ligand.sdf
# in WORKDIR.
#
# Pipeline:
#   1. pdb4amber   - dry/fix protein PDB for tleap while preserving hydrogens
#   2. antechamber - assign GAFF2 atom types and AM1-BCC charges to ligand
#   3. tleap       - write separate protein/ligand and optional complex topologies
#   4. ParmEd      - convert prmtop/rst7 to PQR (charges + mbondi3 radii)
#
# Usage:
#   conda activate ambertools
#   cd examples/browndye && bash run_ambertools.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR/tmp}"
PROTEIN_FF="${PROTEIN_FF:-leaprc.protein.ff19SB}"
LIGAND_FF="${LIGAND_FF:-leaprc.gaff2}"
PB_RADII="${PB_RADII:-mbondi3}"
STRIP_PROTEIN_H="${STRIP_PROTEIN_H:-1}"

read_optional_file() {
    local path="$1"
    if [[ -s "$path" ]]; then
        read -r value <"$path" || true
        printf '%s\n' "$value"
    fi
}

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

LIG_RESNAME="${LIG_RESNAME:-$(read_optional_file "$WORKDIR/ligand_resname.txt")}"
LIG_RESNAME="${LIG_RESNAME:-LIG}"
NET_CHARGE="${NET_CHARGE:-$(read_optional_file "$WORKDIR/ligand_charge.txt")}"
NET_CHARGE="${NET_CHARGE:--1}"

require_file "$WORKDIR/protein_fixed.pdb"
require_file "$WORKDIR/ligand.sdf"
require_cmd pdb4amber
require_cmd obabel
require_cmd antechamber
require_cmd parmchk2
require_cmd tleap
require_cmd python3

cd "$WORKDIR"

echo "=== 1. pdb4amber ==="
pdb4amber_args=(-i protein_fixed.pdb -o protein_amber.pdb -d --no-conect)
if [[ "$STRIP_PROTEIN_H" == "1" ]]; then
    pdb4amber_args+=(-y)
fi
pdb4amber "${pdb4amber_args[@]}"

echo "=== 2. antechamber + parmchk2 ==="
obabel ligand.sdf -O ligand_seed.mol2
python3 - "$LIG_RESNAME" <<'PY'
from pathlib import Path
import sys

resname = sys.argv[1]
path = Path("ligand_seed.mol2")
text = path.read_text()
for old in ("UNL1", "UNL", "UNK"):
    text = text.replace(old, resname)
path.write_text(text)
PY

antechamber \
    -i ligand_seed.mol2 -fi mol2 \
    -o ligand_amber.mol2 -fo mol2 \
    -c bcc -s 2 -at gaff2 \
    -nc "$NET_CHARGE" -rn "$LIG_RESNAME"

parmchk2 -i ligand_amber.mol2 -f mol2 -o ligand.frcmod

echo "=== 3. tleap ==="
cat >tleap.in <<EOF
source $PROTEIN_FF
source $LIGAND_FF

$LIG_RESNAME = loadmol2 ligand_amber.mol2
loadamberparams ligand.frcmod
protein = loadpdb protein_amber.pdb
complex = combine {protein $LIG_RESNAME}

set default PBRadii $PB_RADII
saveamberparm protein protein.prmtop protein.rst7
saveamberparm $LIG_RESNAME ligand.prmtop ligand.rst7
saveamberparm complex complex.prmtop complex.rst7
quit
EOF
tleap -f tleap.in

echo "=== 4. ParmEd -> PQR ==="
python3 -c "
import parmed as pmd
for stem in ('protein', 'ligand', 'complex'):
    parm = pmd.load_file(f'{stem}.prmtop', xyz=f'{stem}.rst7')
    parm.save(f'{stem}.pqr', overwrite=True)
"

echo "=== Done ==="
ls -lh protein.pqr ligand.pqr complex.pqr
