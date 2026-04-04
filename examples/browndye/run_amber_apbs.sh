#!/usr/bin/env bash
# run_amber_apbs.sh - AMBER parameterization + APBS electrostatics
#
# Prerequisite: run complex_pqr.ipynb first to produce protein_fixed.pdb and ligand.sdf
# in WORKDIR.
#
# Pipeline:
#   1. pdb4amber   - strip H and water, fix residue names for tleap
#   2. antechamber - assign GAFF2 atom types and AM1-BCC charges to ligand
#   3. tleap       - combine protein + ligand, write AMBER topology
#   4. ParmEd      - convert prmtop/rst7 to PQR (charges + mbondi3 radii)
#   5. inputgen    - generate APBS input from PQR dimensions
#   6. APBS        - solve linearized Poisson-Boltzmann equation
#
# Usage:
#   conda activate ambertools
#   cd examples/browndye && bash run_amber_apbs.sh

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
WORKDIR="tmp"
LIG_RESNAME="LIG"
NET_CHARGE="-1"
IONIC_STRENGTH="0.150"
PROTEIN_FF="leaprc.protein.ff19SB"
LIGAND_FF="leaprc.gaff2"
PB_RADII="mbondi3"

cd "$WORKDIR"

# ── 1. pdb4amber ────────────────────────────────────────────────────────────
echo "=== 1. pdb4amber ==="
pdb4amber -i protein_fixed.pdb -o protein_amber.pdb -y -d --no-conect

# ── 2. Ligand parameterization ──────────────────────────────────────────────
echo "=== 2. antechamber + parmchk2 ==="
obabel ligand.sdf -O ligand_seed.mol2
sed -i "s/UNL1/${LIG_RESNAME}/g" ligand_seed.mol2

antechamber \
    -i ligand_seed.mol2 -fi mol2 \
    -o ligand_amber.mol2 -fo mol2 \
    -c bcc -s 2 -at gaff2 \
    -nc "$NET_CHARGE" -rn "$LIG_RESNAME"

parmchk2 -i ligand_amber.mol2 -f mol2 -o ligand.frcmod

# ── 3. tleap ────────────────────────────────────────────────────────────────
echo "=== 3. tleap ==="
cat >tleap.in <<EOF
source $PROTEIN_FF
source $LIGAND_FF

$LIG_RESNAME = loadmol2 ligand_amber.mol2
loadamberparams ligand.frcmod
protein = loadpdb protein_amber.pdb
complex = combine {protein $LIG_RESNAME}

set default PBRadii $PB_RADII
saveamberparm complex complex.prmtop complex.rst7
quit
EOF
tleap -f tleap.in

# ── 4. ParmEd -> PQR ───────────────────────────────────────────────────────
echo "=== 4. ParmEd -> PQR ==="
python3 -c "
import parmed as pmd
parm = pmd.load_file('complex.prmtop', xyz='complex.rst7')
parm.save('complex.pqr', overwrite=True)
"

# ── 5. APBS input generation ───────────────────────────────────────────────
echo "=== 5. inputgen ==="
# inputgen CLI has a bug in pdb2pqr<=3.7.1: --istrng is parsed as str, not float.
# Use the Python API directly.
python3 -c "
from pdb2pqr.inputgen import Input
from pdb2pqr.psize import Psize

size = Psize()
size.run_psize('complex.pqr')
inp = Input('complex.pqr', size, method='mg-auto', asyncflag=False,
            istrng=${IONIC_STRENGTH}, potdx=True)
inp.print_input_files('complex.in')
"
# Fix DX output stem: inputgen writes 'write pot dx complex.pqr' -> APBS would
# produce complex.pqr.dx; change to 'complex' so output is complex-PE0.dx.
sed -i 's|write pot dx complex\.pqr|write pot dx complex|' complex.in

# ── 6. APBS ────────────────────────────────────────────────────────────────
echo "=== 6. APBS ==="
apbs complex.in 2>&1 | tee apbs.log
mv complex-PE0.dx complex.dx

echo "=== Done ==="
ls -lh complex.prmtop complex.rst7 complex.pqr complex.in complex.dx
