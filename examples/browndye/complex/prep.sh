#!/usr/bin/env bash
# Prepare the complex body with AmberTools/GAFF2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_ROOT="${TMP_ROOT:-$EXAMPLE_DIR/tmp}"
INPUT_DIR="${INPUT_DIR:-$TMP_ROOT/complex/prep}"
STEP_DIR="${AMBERTOOLS_DIR:-$TMP_ROOT/complex/ambertools}"
INTERMEDIATE_DIR="${INTERMEDIATE_DIR:-$STEP_DIR/intermediate}"
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

LIG_RESNAME="${LIG_RESNAME:-$(read_optional_file "$INPUT_DIR/ligand_resname.txt")}"
LIG_RESNAME="${LIG_RESNAME:-LIG}"
NET_CHARGE="${NET_CHARGE:-$(read_optional_file "$INPUT_DIR/ligand_charge.txt")}"
NET_CHARGE="${NET_CHARGE:--1}"

require_file "$INPUT_DIR/protein_fixed.pdb"
require_file "$INPUT_DIR/ligand.sdf"
require_cmd pdb4amber
require_cmd obabel
require_cmd antechamber
require_cmd parmchk2
require_cmd tleap
require_cmd python3
python3 -c "import parmed" 2>/dev/null || {
    printf 'Missing Python package: parmed\n' >&2
    exit 1
}

mkdir -p "$STEP_DIR" "$INTERMEDIATE_DIR"
cp "$INPUT_DIR/protein_fixed.pdb" "$INTERMEDIATE_DIR/protein_fixed.pdb"
cp "$INPUT_DIR/ligand.sdf" "$INTERMEDIATE_DIR/ligand.sdf"
cd "$INTERMEDIATE_DIR"

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
if resname not in text:
    raise SystemExit(f"Residue name {resname!r} not found in {path} after replacement")
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

for stem in protein ligand complex; do
    cp "$stem.prmtop" "$STEP_DIR/$stem.prmtop"
    cp "$stem.rst7" "$STEP_DIR/$stem.rst7"
    cp "$stem.pqr" "$STEP_DIR/$stem.pqr"
done

echo "=== Done ==="
ls -lh "$STEP_DIR"/complex.pqr "$STEP_DIR"/protein.pqr "$STEP_DIR"/ligand.pqr
