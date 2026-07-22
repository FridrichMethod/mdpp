#!/usr/bin/env bash
#
# Build native Schrodinger FEP+ relative-binding inputs for the LplA
# open/closed conformation comparison, reusing the ligand selection, the two
# protein conformations, and the SMILES chemistry from the OpenFE reference
# notebook (../openfe/rbfe_open_closed.ipynb).
#
# Native FEP+ pipeline (everything except the inputs is FEP+ native):
#   1. PrepWizard each apo conformation -> prepared receptor .mae (pH 7).
#   2. Concatenate the 12 OpenFE-aligned ligand poses -> ligands.maegz.
#   3. Assemble a pose-viewer file per conformation: receptor first, then the
#      shared ligand poses (all already in one NTD-aligned frame).
#   4. fep_mapper.py -> native optimized-topology FEP map (<conf>_map.fmp).
#
# The ligand poses are taken verbatim from OpenFE (one canonical pose per
# ligand, superposed into both pockets), so the FEP map and the eventual
# fep_plus run are apples-to-apples with the OpenFE network. FEP+ assigns its
# own OPLS4 charges, so the AM1-BCC charges from OpenFE are intentionally NOT
# reused here -- only the 3D poses are.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHRODINGER="${SCHRODINGER:-/apps/schrodinger2025-4}"

INPUT_DIR="${SCRIPT_DIR}/inputs"
LIGAND_DIR="${INPUT_DIR}/ligands"
WORK_DIR="${SCRIPT_DIR}/tmp"

# pH used for protein protonation -- matches PH = 7.0 in the OpenFE notebook.
PREP_PH="7.0"
# Restrained-minimization heavy-atom RMSD cutoff (A). The default 0.3 keeps the
# receptor in the OpenFE-aligned frame so the shared ligand poses stay in place.
PREP_RMSD="0.3"
# fep_mapper graph topology: full | normal | star | windmill. "normal" is the
# FEP+ default (minimal-redundant-like optimized graph).
TOPOLOGY="normal"

CONFORMATIONS=("open" "closed")
FORCE=0

usage() {
    cat <<EOF
Usage: ${0##*/} [options]

Build native FEP+ relative-binding inputs (receptor .mae + ligand .maegz +
pose-viewer + FEP map .fmp) for the LplA open/closed comparison.

Options:
  -c, --conformation NAME   Build only one conformation (open | closed).
                            Default: both.
  -t, --topology TYPE       fep_mapper topology: full|normal|star|windmill.
                            Default: ${TOPOLOGY}.
      --ph VALUE            PrepWizard protonation pH. Default: ${PREP_PH}.
      --rmsd VALUE          PrepWizard min RMSD cutoff (A). Default: ${PREP_RMSD}.
  -f, --force               Rebuild outputs even if they already exist.
  -h, --help                Show this help and exit.

Environment:
  SCHRODINGER               Suite root. Default: ${SCHRODINGER}.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c | --conformation)
            CONFORMATIONS=("$2")
            shift 2
            ;;
        -t | --topology)
            TOPOLOGY="$2"
            shift 2
            ;;
        --ph)
            PREP_PH="$2"
            shift 2
            ;;
        --rmsd)
            PREP_RMSD="$2"
            shift 2
            ;;
        -f | --force)
            FORCE=1
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

if [[ ! -x "${SCHRODINGER}/utilities/prepwizard" ]]; then
    echo "ERROR: Schrodinger not found at SCHRODINGER=${SCHRODINGER}" >&2
    exit 1
fi
for conf in "${CONFORMATIONS[@]}"; do
    if [[ "${conf}" != "open" && "${conf}" != "closed" ]]; then
        echo "ERROR: conformation must be 'open' or 'closed', got '${conf}'" >&2
        exit 1
    fi
done

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# ---------------------------------------------------------------------------
# Step 1: PrepWizard each apo conformation -> prepared receptor .mae
# ---------------------------------------------------------------------------
for conf in "${CONFORMATIONS[@]}"; do
    receptor="${WORK_DIR}/receptor_${conf}.mae"
    protein="${INPUT_DIR}/protein_${conf}.pdb"
    if [[ ! -f "${protein}" ]]; then
        echo "ERROR: missing protein input: ${protein}" >&2
        exit 1
    fi
    if [[ -f "${receptor}" && "${FORCE}" -eq 0 ]]; then
        echo "[${conf}] receptor exists, skipping PrepWizard: ${receptor}"
        continue
    fi
    echo "[${conf}] PrepWizard: ${protein} -> ${receptor}"
    "${SCHRODINGER}/utilities/prepwizard" \
        -NOJOBID \
        -fillsidechains \
        -disulfides \
        -epik_pH "${PREP_PH}" \
        -propka_pH "${PREP_PH}" \
        -rmsd "${PREP_RMSD}" \
        "${protein}" "${receptor}"
    # Remove the PrepWizard scratch job directory (regeneratable).
    rm -rf "${WORK_DIR}/protein_${conf}-"[0-9]*
done

# ---------------------------------------------------------------------------
# Step 2: concatenate the OpenFE-aligned ligand poses -> ligands.maegz
# ---------------------------------------------------------------------------
ligands_mae="${WORK_DIR}/ligands.maegz"
if [[ -f "${ligands_mae}" && "${FORCE}" -eq 0 ]]; then
    echo "ligand bundle exists, skipping: ${ligands_mae}"
else
    mapfile -t ligand_sdfs < <(find "${LIGAND_DIR}" -maxdepth 1 -name "*.sdf" | sort)
    if [[ "${#ligand_sdfs[@]}" -eq 0 ]]; then
        echo "ERROR: no ligand SDFs in ${LIGAND_DIR}" >&2
        exit 1
    fi
    echo "bundling ${#ligand_sdfs[@]} ligand poses -> ${ligands_mae}"
    cat_args=()
    for sdf in "${ligand_sdfs[@]}"; do
        cat_args+=(-isd "${sdf}")
    done
    "${SCHRODINGER}/utilities/structcat" "${cat_args[@]}" -omae "${ligands_mae}"
fi

# ---------------------------------------------------------------------------
# Step 3 + 4: pose-viewer assembly + native FEP map per conformation
# ---------------------------------------------------------------------------
for conf in "${CONFORMATIONS[@]}"; do
    receptor="${WORK_DIR}/receptor_${conf}.mae"
    pv="${WORK_DIR}/${conf}_pv.mae"
    map_base="${WORK_DIR}/${conf}_map"

    if [[ -f "${pv}" && "${FORCE}" -eq 0 ]]; then
        echo "[${conf}] pose-viewer exists, skipping: ${pv}"
    else
        echo "[${conf}] assembling pose-viewer (receptor + ligands) -> ${pv}"
        "${SCHRODINGER}/utilities/structcat" \
            -imae "${receptor}" \
            -imae "${ligands_mae}" \
            -omae "${pv}"
    fi

    if [[ -f "${map_base}.fmp" && "${FORCE}" -eq 0 ]]; then
        echo "[${conf}] FEP map exists, skipping fep_mapper: ${map_base}.fmp"
        continue
    fi
    echo "[${conf}] fep_mapper (topology=${TOPOLOGY}) -> ${map_base}.fmp"
    "${SCHRODINGER}/run" -FROM scisol fep_mapper.py \
        "${pv}" \
        -o "${map_base}" \
        -t "${TOPOLOGY}"
done

echo
echo "Done. Inputs written under: ${WORK_DIR}"
for conf in "${CONFORMATIONS[@]}"; do
    echo "  [${conf}] receptor_${conf}.mae  ${conf}_pv.mae  ${conf}_map.fmp  ${conf}_map.edge"
done
echo
echo "Inspect a map with:  \$SCHRODINGER/run \$SCHRODINGER/mmshare-v7.2/python/scripts/fmp_info.py tmp/open_map.fmp"
echo "Then launch FEP+ with: ./run_fep_plus.sh -c open"
