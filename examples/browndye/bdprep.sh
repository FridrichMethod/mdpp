#!/usr/bin/env bash
# bdprep.sh - build BrownDye XML inputs from prepared PQR/APBS files.
#
# Prerequisites:
#   conda activate ambertools
#   cd examples/browndye
#   bash run_ambertools.sh
#   bash run_apbs.sh
#
# This script creates:
#   protein_atoms.xml, ligand_atoms.xml
#   contact_types.xml, reaction_pairs.xml, reactions.xml
#   input.xml
#   protein_ligand_simulation.xml (via bd_top)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR/tmp}"
BD_HOME="${BD_HOME:-/apps/browndye2}"
BD_BIN="${BD_BIN:-$BD_HOME/bin}"
BD_AUX="${BD_AUX:-$BD_HOME/aux}"

CORE0="${CORE0:-protein}"
CORE1="${CORE1:-ligand}"
CORE0_IS_PROTEIN="${CORE0_IS_PROTEIN:-true}"
CORE1_IS_PROTEIN="${CORE1_IS_PROTEIN:-false}"
CORE0_DIELECTRIC="${CORE0_DIELECTRIC:-4.0}"
CORE1_DIELECTRIC="${CORE1_DIELECTRIC:-4.0}"

RXN_SEARCH_DISTANCE="${RXN_SEARCH_DISTANCE:-5.5}"
RXN_DISTANCE="${RXN_DISTANCE:-5.5}"
RXN_NEEDED="${RXN_NEEDED:-3}"

N_THREADS="${N_THREADS:-1}"
SEED="${SEED:-11111111}"
N_TRAJECTORIES="${N_TRAJECTORIES:-1000}"
N_TRAJECTORIES_PER_OUTPUT="${N_TRAJECTORIES_PER_OUTPUT:-10}"
MAX_N_STEPS="${MAX_N_STEPS:-1000000}"
RESULTS_FILE="${RESULTS_FILE:-results.xml}"

SOLVENT_DIELECTRIC="${SOLVENT_DIELECTRIC:-78.0}"
RELATIVE_VISCOSITY="${RELATIVE_VISCOSITY:-1.0}"
KT="${KT:-1.0}"
DESOLVATION_PARAMETER="${DESOLVATION_PARAMETER:-1.0}"
SOLVENT_RADIUS="${SOLVENT_RADIUS:-1.4}"
DEBYE_LENGTH="${DEBYE_LENGTH:-}"

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
    printf 'If /apps/browndye2 was built from source, compile/install the aux tools first.\n' >&2
    exit 1
}

infer_debye_length() {
    python3 - "$CORE0.apbs.log" "$CORE1.apbs.log" <<'PY'
import re
import sys
from pathlib import Path

for name in sys.argv[1:]:
    path = Path(name)
    if not path.is_file():
        continue
    text = path.read_text(errors="ignore")
    match = re.search(r"[Dd]ebye[- ]length[:\s]+([0-9.]+)", text)
    match = match or re.search(r"[Gg]ot debye length\s+([0-9.]+)", text)
    if match:
        print(match.group(1))
        raise SystemExit
raise SystemExit(1)
PY
}

write_contact_types() {
    python3 - "$CORE0.pqr" "$CORE1.pqr" >contact_types.xml <<'PY'
import sys
from pathlib import Path


def is_heavy_atom(atom_name: str) -> bool:
    stripped = atom_name.strip()
    return bool(stripped) and not stripped.upper().startswith("H")


def contacts(path: Path) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        atom = parts[2]
        residue = parts[3]
        key = (atom, residue)
        if is_heavy_atom(atom) and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


mol0 = contacts(Path(sys.argv[1]))
mol1 = contacts(Path(sys.argv[2]))

print("<contacts>")
print("  <combinations>")
for label, entries in (("molecule0", mol0), ("molecule1", mol1)):
    print(f"    <{label}>")
    for atom, residue in entries:
        print(f"      <contact> <atom> {atom} </atom> <residue> {residue} </residue> </contact>")
    print(f"    </{label}>")
print("  </combinations>")
print("</contacts>")
PY
}

write_input_xml() {
    cat >input.xml <<EOF
<top>
  <n_threads> $N_THREADS </n_threads>
  <seed> $SEED </seed>
  <output> $RESULTS_FILE </output>

  <n_trajectories> $N_TRAJECTORIES </n_trajectories>
  <n_trajectories_per_output> $N_TRAJECTORIES_PER_OUTPUT </n_trajectories_per_output>
  <max_n_steps> $MAX_N_STEPS </max_n_steps>

  <system>
    <reaction_file> reactions.xml </reaction_file>

    <solvent>
      <debye_length> $DEBYE_LENGTH </debye_length>
      <dielectric> $SOLVENT_DIELECTRIC </dielectric>
      <relative_viscosity> $RELATIVE_VISCOSITY </relative_viscosity>
      <kT> $KT </kT>
      <desolvation_parameter> $DESOLVATION_PARAMETER </desolvation_parameter>
      <solvent_radius> $SOLVENT_RADIUS </solvent_radius>
    </solvent>

    <time_step_tolerances>
      <minimum_core_dt> 0.0 </minimum_core_dt>
      <minimum_core_reaction_dt> 0.0 </minimum_core_reaction_dt>
    </time_step_tolerances>

    <group>
      <name> $CORE0 </name>
      <core>
        <name> $CORE0 </name>
        <all_in_surface> false </all_in_surface>
        <is_protein> $CORE0_IS_PROTEIN </is_protein>
        <atoms> ${CORE0}_atoms.xml </atoms>
        <electric_field>
          <grid> $CORE0.dx </grid>
        </electric_field>
        <dielectric> $CORE0_DIELECTRIC </dielectric>
      </core>
    </group>

    <group>
      <name> $CORE1 </name>
      <core>
        <name> $CORE1 </name>
        <all_in_surface> false </all_in_surface>
        <is_protein> $CORE1_IS_PROTEIN </is_protein>
        <atoms> ${CORE1}_atoms.xml </atoms>
        <electric_field>
          <grid> $CORE1.dx </grid>
        </electric_field>
        <dielectric> $CORE1_DIELECTRIC </dielectric>
      </core>
    </group>
  </system>
</top>
EOF
}

require_file "$WORKDIR/$CORE0.pqr"
require_file "$WORKDIR/$CORE1.pqr"
require_file "$WORKDIR/$CORE0.dx"
require_file "$WORKDIR/$CORE1.dx"
PQR2XML="$(resolve_tool pqr2xml)"
MAKE_RXN_PAIRS="$(resolve_tool make_rxn_pairs)"
MAKE_RXN_FILE="$(resolve_tool make_rxn_file)"
BD_TOP="$(resolve_tool bd_top)"

cd "$WORKDIR"

if [[ -z "$DEBYE_LENGTH" ]]; then
    if ! DEBYE_LENGTH="$(infer_debye_length)"; then
        printf 'Could not infer DEBYE_LENGTH from APBS logs. Set DEBYE_LENGTH explicitly.\n' >&2
        exit 1
    fi
fi

echo "=== 1. PQR -> BrownDye XML ==="
"$PQR2XML" <"$CORE0.pqr" >"${CORE0}_atoms.xml"
"$PQR2XML" <"$CORE1.pqr" >"${CORE1}_atoms.xml"

echo "=== 2. Reaction criteria ==="
write_contact_types
"$MAKE_RXN_PAIRS" -nonred \
    -mol0 "${CORE0}_atoms.xml" \
    -mol1 "${CORE1}_atoms.xml" \
    -ctypes contact_types.xml \
    -dist "$RXN_SEARCH_DISTANCE" >reaction_pairs.xml
"$MAKE_RXN_FILE" \
    -pairs reaction_pairs.xml \
    -state_from before \
    -state_to after \
    -rxn association \
    -mol0 "$CORE0" "$CORE0" \
    -mol1 "$CORE1" "$CORE1" \
    -distance "$RXN_DISTANCE" \
    -nneeded "$RXN_NEEDED" >reactions.xml

echo "=== 3. BrownDye top-level input ==="
write_input_xml
"$BD_TOP" input.xml

echo "=== Done ==="
echo "Simulation XML should be ${CORE0}_${CORE1}_simulation.xml"
