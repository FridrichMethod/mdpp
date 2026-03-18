#!/usr/bin/env bash
# run_amber_apbs.sh - Steps 3-6 of BrownDye2 complex PQR preparation
#
# Expects in WORKDIR: complex.pdb, ligand.sdf, ligand.pdb
# Produces: complex.prmtop, complex.rst7, complex.pqr, complex.in, complex.dx

set -euo pipefail

# Configurations
WORKDIR="."
LIG_RESNAME="LIG"
NET_CHARGE="-1"
IONIC_STRENGTH="0.150"
PROTEIN_FF="leaprc.protein.ff19SB"
LIGAND_FF="leaprc.gaff2"

# Check required commands
for cmd in obabel antechamber parmchk2 tleap python3 inputgen apbs; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: $cmd not found" >&2
        exit 1
    fi
done

cd "$WORKDIR"

for f in complex.pdb ligand.sdf ligand.pdb; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file not found: $WORKDIR/$f" >&2
        exit 1
    fi
done

# Step 3: AmberTools parameterization
echo "=== Step 3: AmberTools parameterization ==="

echo "--- antechamber ---"
obabel ligand.sdf -O ligand_seed.mol2

antechamber \
    -i ligand_seed.mol2 \
    -fi mol2 \
    -o ligand_amber.mol2 \
    -fo mol2 \
    -c bcc \
    -s 2 \
    -at gaff2 \
    -nc "$NET_CHARGE" \
    -rn "$LIG_RESNAME" \
    -an n \
    -a ligand.pdb \
    -fa pdb \
    -ao name

echo "--- parmchk2 ---"
parmchk2 -i ligand_amber.mol2 -f mol2 -o ligand.frcmod

echo "--- tleap ---"
cat >tleap.in <<EOF
source $PROTEIN_FF
source $LIGAND_FF

$LIG_RESNAME = loadmol2 ligand_amber.mol2
loadamberparams ligand.frcmod

complex = loadpdb complex.pdb
check complex
saveamberparm complex complex.prmtop complex.rst7
savepdb complex complex_from_tleap.pdb
quit
EOF

tleap -f tleap.in

# Step 4: ParmEd prmtop/rst7 -> PQR
echo "=== Step 4: ParmEd -> PQR ==="

python3 -c "
import parmed as pmd
parm = pmd.load_file('complex.prmtop', xyz='complex.rst7')
parm.save('complex.pqr', overwrite=True)
"

# Step 5: APBS input generation via pdb2pqr inputgen
echo "=== Step 5: APBS input generation ==="

inputgen "--istrng=${IONIC_STRENGTH}" --potdx complex.pqr
echo "Generated APBS input from complex.pqr"

# inputgen writes <stem>.in next to the PQR
APBS_IN="complex.in"
if [[ ! -f "$APBS_IN" ]]; then
    # Fall back to newest .in file in case of different naming
    APBS_IN="$(ls -1t ./*.in 2>/dev/null | head -n 1 || true)"
    if [[ -z "$APBS_IN" || ! -f "$APBS_IN" ]]; then
        echo "ERROR: inputgen did not produce an APBS input file" >&2
        exit 1
    fi
fi
echo "Using APBS input: $APBS_IN"

# Step 6: Run APBS
echo "=== Step 6: Run APBS ==="

apbs "$APBS_IN"

echo "=== Done ==="
ls -lh complex.prmtop complex.rst7 complex.pqr "$APBS_IN" complex*.dx 2>/dev/null || true
