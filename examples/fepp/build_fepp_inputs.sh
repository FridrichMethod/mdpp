#!/usr/bin/env bash
# Build controlled, provenance-tracked native FEP+ inputs for LplA open/closed.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHRODINGER="$(realpath -m "${SCHRODINGER:-/apps/schrodinger2025-4}")"
INPUT_DIR="$(realpath -m "${FEPP_INPUT_DIR:-${SCRIPT_DIR}/inputs}")"
LIGAND_DIR="${INPUT_DIR}/ligands"
WORK_DIR="$(realpath -m "${FEPP_WORK_DIR:-${SCRIPT_DIR}/tmp}")"

PREP_PH="7.0"
PREP_RMSD="0.3"
HIS267_STATE="HIE"
TOPOLOGY="normal"
CONFORMATIONS=("open" "closed")
FORCE=0
CHECK_ONLY=0
ALLOW_MICROSTATE_MISMATCH=0
BUILD_SCHEMA="fepp-input-build-v5"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_value() {
    local option="$1"
    local value="${2-}"
    [[ -n "${value}" && "${value}" != -* ]] || die "${option} requires a value"
}

usage() {
    cat <<EOF
Usage: ${0##*/} [options]

Build native FEP+ receptor, ligand, pose-viewer, and paired .fmp/.edge inputs.
Outputs are reused only when dependency fingerprints and output hashes match.

Options:
  -c, --conformation NAME   open | closed | both (default: both).
  -t, --topology TYPE       full | normal | star | windmill (default: ${TOPOLOGY}).
      --ph VALUE            PrepWizard pH in [0,14] (default: ${PREP_PH}).
      --rmsd VALUE          Restrained-minimization RMSD in A, >0 (default: ${PREP_RMSD}).
      --his267-state STATE  HID | HIE | HIP | auto (default: ${HIS267_STATE}).
      --allow-microstate-mismatch
                            Record but allow receptor protonation mismatches.
      --check               Verify all requested artifacts; build nothing.
  -f, --force               Rebuild requested artifacts.
  -h, --help                Show this help.

Environment:
  SCHRODINGER               Suite root (default: ${SCHRODINGER}).
  FEPP_INPUT_DIR            Input root override, useful for isolated validation.
  FEPP_WORK_DIR             Generated-artifact root override.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c | --conformation)
            require_value "$1" "${2-}"
            case "$2" in
                open | closed) CONFORMATIONS=("$2") ;;
                both) CONFORMATIONS=("open" "closed") ;;
                *) die "conformation must be open, closed, or both; got '$2'" ;;
            esac
            shift 2
            ;;
        -t | --topology)
            require_value "$1" "${2-}"
            TOPOLOGY="$2"
            shift 2
            ;;
        --ph)
            require_value "$1" "${2-}"
            PREP_PH="$2"
            shift 2
            ;;
        --rmsd)
            require_value "$1" "${2-}"
            PREP_RMSD="$2"
            shift 2
            ;;
        --his267-state)
            require_value "$1" "${2-}"
            HIS267_STATE="$2"
            shift 2
            ;;
        --allow-microstate-mismatch)
            ALLOW_MICROSTATE_MISMATCH=1
            shift
            ;;
        --check)
            CHECK_ONLY=1
            shift
            ;;
        -f | --force)
            FORCE=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *) die "unknown argument: $1" ;;
    esac
done

case "${TOPOLOGY}" in
    full | normal | star | windmill) ;;
    *) die "topology must be full, normal, star, or windmill; got '${TOPOLOGY}'" ;;
esac
case "${HIS267_STATE}" in
    HID | HIE | HIP | auto) ;;
    *) die "His267 state must be HID, HIE, HIP, or auto; got '${HIS267_STATE}'" ;;
esac
[[ "${PREP_PH}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "pH must be numeric"
awk -v value="${PREP_PH}" 'BEGIN { exit !(value >= 0 && value <= 14) }' ||
    die "pH must be in [0,14]"
[[ "${PREP_RMSD}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "RMSD must be numeric"
awk -v value="${PREP_RMSD}" 'BEGIN { exit !(value > 0) }' || die "RMSD must be >0"
[[ "${CHECK_ONLY}" -eq 0 || "${FORCE}" -eq 0 ]] || die "--check and --force are incompatible"

for executable in \
    "${SCHRODINGER}/run" \
    "${SCHRODINGER}/utilities/prepwizard" \
    "${SCHRODINGER}/utilities/structcat"; do
    [[ -x "${executable}" ]] || die "required executable not found: ${executable}"
done
[[ -s "${SCHRODINGER}/version.txt" ]] ||
    die "Suite release metadata not found: ${SCHRODINGER}/version.txt"
for helper in \
    "${SCRIPT_DIR}/validate_ligand_inputs.py" \
    "${SCRIPT_DIR}/compare_receptor_microstates.py" \
    "${SCRIPT_DIR}/extract_edge_mappings.py"; do
    [[ -f "${helper}" ]] || die "required helper not found: ${helper}"
done
[[ -f "${INPUT_DIR}/ligands_amp_fep.smi" ]] ||
    die "required input not found: ${INPUT_DIR}/ligands_amp_fep.smi"

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    [[ -d "${WORK_DIR}" ]] || die "work directory does not exist: ${WORK_DIR}"
else
    mkdir -p "${WORK_DIR}"
fi
if [[ "${FEPP_BUILD_LOCK_HELD:-0}" != "1" ]]; then
    command -v flock >/dev/null || die "required executable not found: flock"
    exec {build_lock_fd}<"${WORK_DIR}"
    flock -x "${build_lock_fd}"
fi

file_hash() {
    sha256sum "$1" | awk '{print $1}'
}

text_hash() {
    sha256sum | awk '{print $1}'
}

artifact_current() {
    local fingerprint="$1"
    local sidecar="$2"
    shift 2
    [[ "${FORCE}" -eq 0 && -f "${sidecar}" ]] || return 1
    grep -Fqx "fingerprint=${fingerprint}" "${sidecar}" || return 1
    local output hash
    for output in "$@"; do
        [[ -s "${output}" ]] || return 1
        hash="$(file_hash "${output}")"
        grep -Fqx "output=${hash} ${output}" "${sidecar}" || return 1
    done
}

write_sidecar() {
    local sidecar="$1"
    local fingerprint="$2"
    local description="$3"
    local dependencies="$4"
    shift 4
    local temporary="${sidecar}.tmp.$$"
    {
        printf 'schema=%s\n' "${BUILD_SCHEMA}"
        printf 'fingerprint=%s\n' "${fingerprint}"
        printf 'created_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        printf 'description=%s\n' "${description}"
        printf 'dependencies=%s\n' "${dependencies}"
        printf 'suite=%s\n' "${SUITE_ID}"
        local output
        for output in "$@"; do
            printf 'output=%s %s\n' "$(file_hash "${output}")" "${output}"
        done
    } >"${temporary}"
    mv "${temporary}" "${sidecar}"
}

require_current_or_build() {
    local label="$1"
    local fingerprint="$2"
    local sidecar="$3"
    shift 3
    if artifact_current "${fingerprint}" "${sidecar}" "$@"; then
        echo "[cache] ${label}"
        return 0
    fi
    [[ "${CHECK_ONLY}" -eq 0 ]] || die "stale or missing artifact: ${label}"
    return 1
}

SUITE_ID="$(tr '\t\r\n' '   ' <"${SCHRODINGER}/version.txt")"
[[ "${SUITE_ID}" =~ [^[:space:]] ]] || die "Suite release metadata is empty"

mapfile -d '' -t ligand_sdfs < <(find "${LIGAND_DIR}" -maxdepth 1 -type f -name '*.sdf' -print0 | sort -z)
[[ "${#ligand_sdfs[@]}" -gt 0 ]] || die "no ligand SDF files in ${LIGAND_DIR}"
ligand_inputs_payload="$(
    printf '%s\n' \
        "${BUILD_SCHEMA}" \
        "suite=${SUITE_ID}" \
        "smiles=$(file_hash "${INPUT_DIR}/ligands_amp_fep.smi")" \
        "validator=$(file_hash "${SCRIPT_DIR}/validate_ligand_inputs.py")"
    for sdf in "${ligand_sdfs[@]}"; do
        printf 'sdf=%s %s\n' "$(file_hash "${sdf}")" "${sdf##*/}"
    done
)"
ligand_fingerprint="$(printf '%s' "${ligand_inputs_payload}" | text_hash)"

validation_report="${WORK_DIR}/ligand_validation.json"
validation_sidecar="${validation_report}.provenance"
if ! require_current_or_build \
    "ligand validation report" "${ligand_fingerprint}" "${validation_sidecar}" \
    "${validation_report}"; then
    stage="$(mktemp -d "${WORK_DIR}/.ligand-validation.XXXXXX")"
    trap 'rm -rf -- "${stage}"' EXIT
    staged_validation="${stage}/ligand_validation.json"
    "${SCHRODINGER}/run" python3 "${SCRIPT_DIR}/validate_ligand_inputs.py" \
        --smiles "${INPUT_DIR}/ligands_amp_fep.smi" \
        --ligand-dir "${LIGAND_DIR}" \
        --expected-charge -1 \
        -o "${staged_validation}"
    [[ -s "${staged_validation}" ]] || die "ligand validator produced no report"
    mv "${staged_validation}" "${validation_report}"
    write_sidecar \
        "${validation_sidecar}" "${ligand_fingerprint}" \
        "ligand titles, chemistry, stereochemistry, 3D conformers, and charge=-1" \
        "ligand_set_fingerprint=${ligand_fingerprint}" \
        "${validation_report}"
    rm -rf -- "${stage}"
    trap - EXIT
fi

ligands_mae="${WORK_DIR}/ligands.maegz"
ligands_sidecar="${ligands_mae}.provenance"
if ! require_current_or_build \
    "ligand bundle" "${ligand_fingerprint}" "${ligands_sidecar}" "${ligands_mae}"; then
    stage="$(mktemp -d "${WORK_DIR}/.ligands.XXXXXX")"
    trap 'rm -rf -- "${stage}"' EXIT
    staged_ligands="${stage}/ligands.maegz"
    structcat_args=()
    for sdf in "${ligand_sdfs[@]}"; do
        structcat_args+=(-isd "${sdf}")
    done
    echo "[build] ligand bundle (${#ligand_sdfs[@]} molecules)"
    "${SCHRODINGER}/utilities/structcat" "${structcat_args[@]}" -omae "${staged_ligands}"
    [[ -s "${staged_ligands}" ]] || die "structcat produced an empty ligand bundle"
    mv "${staged_ligands}" "${ligands_mae}"
    write_sidecar \
        "${ligands_sidecar}" "${ligand_fingerprint}" \
        "ordered SDF bundle; validated against ligands_amp_fep.smi; charge=-1" \
        "smiles_sha256=$(file_hash "${INPUT_DIR}/ligands_amp_fep.smi");ligand_set_fingerprint=${ligand_fingerprint}" \
        "${ligands_mae}"
    rm -rf -- "${stage}"
    trap - EXIT
fi

for conf in "${CONFORMATIONS[@]}"; do
    protein="${INPUT_DIR}/protein_${conf}.pdb"
    receptor="${WORK_DIR}/receptor_${conf}.mae"
    receptor_sidecar="${receptor}.provenance"
    [[ -f "${protein}" ]] || die "missing protein input: ${protein}"
    receptor_fingerprint="$(
        printf '%s\n' \
            "${BUILD_SCHEMA}" \
            "suite=${SUITE_ID}" \
            "protein=$(file_hash "${protein}")" \
            "ph=${PREP_PH}" \
            "rmsd=${PREP_RMSD}" \
            "his267=${HIS267_STATE}" | text_hash
    )"
    if ! require_current_or_build \
        "${conf} receptor" "${receptor_fingerprint}" "${receptor_sidecar}" "${receptor}"; then
        stage="$(mktemp -d "${WORK_DIR}/.${conf}-receptor.XXXXXX")"
        trap 'rm -rf -- "${stage}"' EXIT
        staged_receptor="${stage}/receptor_${conf}.mae"
        prep_args=(
            -NOJOBID
            -fillsidechains
            -disulfides
            -epik_pH "${PREP_PH}"
            -propka_pH "${PREP_PH}"
            -rmsd "${PREP_RMSD}"
        )
        if [[ "${HIS267_STATE}" != "auto" ]]; then
            prep_args+=(-force A:267 "${HIS267_STATE}")
        fi
        echo "[build] ${conf} receptor (His267=${HIS267_STATE})"
        (
            cd "${stage}"
            "${SCHRODINGER}/utilities/prepwizard" \
                "${prep_args[@]}" "${protein}" "${staged_receptor}"
        )
        [[ -s "${staged_receptor}" ]] || die "PrepWizard produced an empty receptor"
        mv "${staged_receptor}" "${receptor}"
        write_sidecar \
            "${receptor_sidecar}" "${receptor_fingerprint}" \
            "PrepWizard ph=${PREP_PH} rmsd_A=${PREP_RMSD} His267=${HIS267_STATE}" \
            "protein_sha256=$(file_hash "${protein}")" \
            "${receptor}"
        rm -rf -- "${stage}"
        trap - EXIT
    fi
done

if [[ " ${CONFORMATIONS[*]} " == *" open "* && " ${CONFORMATIONS[*]} " == *" closed "* ]]; then
    microstate_report="${WORK_DIR}/receptor_microstates.json"
    microstate_sidecar="${microstate_report}.provenance"
    microstate_fingerprint="$(
        printf '%s\n' \
            "${BUILD_SCHEMA}" \
            "suite=${SUITE_ID}" \
            "open=$(file_hash "${WORK_DIR}/receptor_open.mae")" \
            "closed=$(file_hash "${WORK_DIR}/receptor_closed.mae")" \
            "comparator=$(file_hash "${SCRIPT_DIR}/compare_receptor_microstates.py")" \
            "allow_mismatch=${ALLOW_MICROSTATE_MISMATCH}" | text_hash
    )"
    comparison_args=(
        "${SCRIPT_DIR}/compare_receptor_microstates.py"
        "${WORK_DIR}/receptor_open.mae"
        "${WORK_DIR}/receptor_closed.mae"
    )
    if [[ "${ALLOW_MICROSTATE_MISMATCH}" -eq 1 ]]; then
        comparison_args+=(--allow-microstate-mismatch)
    fi
    if ! require_current_or_build \
        "paired receptor microstate report" "${microstate_fingerprint}" \
        "${microstate_sidecar}" "${microstate_report}"; then
        stage="$(mktemp -d "${WORK_DIR}/.microstates.XXXXXX")"
        trap 'rm -rf -- "${stage}"' EXIT
        staged_microstates="${stage}/receptor_microstates.json"
        "${SCHRODINGER}/run" python3 "${comparison_args[@]}" -o "${staged_microstates}"
        [[ -s "${staged_microstates}" ]] || die "microstate comparator produced no report"
        mv "${staged_microstates}" "${microstate_report}"
        write_sidecar \
            "${microstate_sidecar}" "${microstate_fingerprint}" \
            "paired receptor formal charges and hydrogen-attachment microstates" \
            "open_sha256=$(file_hash "${WORK_DIR}/receptor_open.mae");closed_sha256=$(file_hash "${WORK_DIR}/receptor_closed.mae");allow_mismatch=${ALLOW_MICROSTATE_MISMATCH}" \
            "${microstate_report}"
        rm -rf -- "${stage}"
        trap - EXIT
    fi
else
    echo "WARNING: only one conformation requested; paired receptor microstates were not compared" >&2
fi

for conf in "${CONFORMATIONS[@]}"; do
    receptor="${WORK_DIR}/receptor_${conf}.mae"
    pv="${WORK_DIR}/${conf}_pv.mae"
    pv_sidecar="${pv}.provenance"
    pv_fingerprint="$(
        printf '%s\n' \
            "${BUILD_SCHEMA}" \
            "suite=${SUITE_ID}" \
            "receptor=$(file_hash "${receptor}")" \
            "ligands=$(file_hash "${ligands_mae}")" | text_hash
    )"
    if ! require_current_or_build "${conf} pose-viewer" "${pv_fingerprint}" "${pv_sidecar}" "${pv}"; then
        stage="$(mktemp -d "${WORK_DIR}/.${conf}-pv.XXXXXX")"
        trap 'rm -rf -- "${stage}"' EXIT
        staged_pv="${stage}/${conf}_pv.mae"
        echo "[build] ${conf} pose-viewer"
        "${SCHRODINGER}/utilities/structcat" \
            -imae "${receptor}" -imae "${ligands_mae}" -omae "${staged_pv}"
        [[ -s "${staged_pv}" ]] || die "structcat produced an empty pose-viewer"
        mv "${staged_pv}" "${pv}"
        write_sidecar \
            "${pv_sidecar}" "${pv_fingerprint}" \
            "receptor first, then validated shared ligand poses" \
            "receptor_sha256=$(file_hash "${receptor}");ligands_sha256=$(file_hash "${ligands_mae}")" \
            "${pv}"
        rm -rf -- "${stage}"
        trap - EXIT
    fi

    map_fmp="${WORK_DIR}/${conf}_map.fmp"
    map_edge="${WORK_DIR}/${conf}_map.edge"
    map_mappings="${WORK_DIR}/${conf}_map_mappings.json"
    map_sidecar="${WORK_DIR}/${conf}_map.provenance"
    map_fingerprint="$(
        printf '%s\n' \
            "${BUILD_SCHEMA}" \
            "suite=${SUITE_ID}" \
            "pose_viewer=$(file_hash "${pv}")" \
            "mapping_extractor=$(file_hash "${SCRIPT_DIR}/extract_edge_mappings.py")" \
            "environment_structures=1" \
            "topology=${TOPOLOGY}" | text_hash
    )"
    if ! require_current_or_build \
        "${conf} FEP map pair" "${map_fingerprint}" "${map_sidecar}" \
        "${map_fmp}" "${map_edge}" "${map_mappings}"; then
        stage="$(mktemp -d "${WORK_DIR}/.${conf}-map.XXXXXX")"
        trap 'rm -rf -- "${stage}"' EXIT
        staged_base="${stage}/${conf}_map"
        echo "[build] ${conf} FEP map (topology=${TOPOLOGY})"
        "${SCHRODINGER}/run" -FROM scisol fep_mapper.py \
            "${pv}" -o "${staged_base}" -e 1 -t "${TOPOLOGY}"
        [[ -s "${staged_base}.fmp" && -s "${staged_base}.edge" ]] ||
            die "fep_mapper did not produce a complete .fmp/.edge pair"
        staged_mappings="${stage}/${conf}_map_mappings.json"
        "${SCHRODINGER}/run" python3 "${SCRIPT_DIR}/extract_edge_mappings.py" \
            -f "${staged_base}.fmp" -o "${staged_mappings}"
        [[ -s "${staged_mappings}" ]] || die "atom-mapping extraction produced no JSON"
        mv "${staged_base}.fmp" "${map_fmp}"
        mv "${staged_base}.edge" "${map_edge}"
        mv "${staged_mappings}" "${map_mappings}"
        write_sidecar \
            "${map_sidecar}" "${map_fingerprint}" \
            "fep_mapper environment_structures=1 topology=${TOPOLOGY}; FMP, edge, and canonical mapping are one artifact" \
            "pose_viewer_sha256=$(file_hash "${pv}");mapping_extractor_sha256=$(file_hash "${SCRIPT_DIR}/extract_edge_mappings.py");environment_structures=1;topology=${TOPOLOGY}" \
            "${map_fmp}" "${map_edge}" "${map_mappings}"
        rm -rf -- "${stage}"
        trap - EXIT
    fi

    state_mapping_fingerprint="$(
        "${SCHRODINGER}/run" python3 -c \
            'import json, sys; print(json.load(open(sys.argv[1]))["mapping_fingerprint"])' \
            "${map_mappings}"
    )"
    [[ "${state_mapping_fingerprint}" =~ ^[0-9a-f]{64}$ ]] ||
        die "${conf} atom-mapping fingerprint is malformed"
    state_inputs="${WORK_DIR}/state_inputs_${conf}.tsv"
    state_sidecar="${state_inputs}.provenance"
    state_payload="$(
        printf 'schema_version\t1\n'
        printf 'build_schema\t%s\n' "${BUILD_SCHEMA}"
        printf 'suite_release\t%s\n' "${SUITE_ID}"
        printf 'state\t%s\n' "${conf}"
        printf 'receptor_sha256\t%s\n' "$(file_hash "${receptor}")"
        printf 'receptor_provenance_sha256\t%s\n' \
            "$(file_hash "${receptor}.provenance")"
        printf 'pose_viewer_sha256\t%s\n' "$(file_hash "${pv}")"
        printf 'pose_viewer_provenance_sha256\t%s\n' \
            "$(file_hash "${pv}.provenance")"
        printf 'ligand_bundle_sha256\t%s\n' "$(file_hash "${ligands_mae}")"
        printf 'ligand_validation_sha256\t%s\n' "$(file_hash "${validation_report}")"
        printf 'map_sha256\t%s\n' "$(file_hash "${map_fmp}")"
        printf 'edge_sha256\t%s\n' "$(file_hash "${map_edge}")"
        printf 'map_provenance_sha256\t%s\n' "$(file_hash "${map_sidecar}")"
        printf 'atom_mapping_fingerprint\t%s\n' "${state_mapping_fingerprint}"
        printf 'atom_mapping_json_sha256\t%s\n' "$(file_hash "${map_mappings}")"
    )"
    state_fingerprint="$(printf '%s' "${state_payload}" | text_hash)"
    if ! require_current_or_build \
        "${conf} state input manifest" "${state_fingerprint}" \
        "${state_sidecar}" "${state_inputs}"; then
        staged_state="$(mktemp "${WORK_DIR}/.${conf}-state-inputs.XXXXXX")"
        trap 'rm -f -- "${staged_state}"' EXIT
        printf '%s\n' "${state_payload}" >"${staged_state}"
        mv "${staged_state}" "${state_inputs}"
        write_sidecar \
            "${state_sidecar}" "${state_fingerprint}" \
            "single-state FEP+ input cohort for ${conf}" \
            "receptor, pose-viewer, ligand, map, edge, and atom-mapping artifacts" \
            "${state_inputs}"
        trap - EXIT
    fi
done

if [[ " ${CONFORMATIONS[*]} " == *" open "* && " ${CONFORMATIONS[*]} " == *" closed "* ]]; then
    cmp -s "${WORK_DIR}/open_map.edge" "${WORK_DIR}/closed_map.edge" ||
        die "open/closed mapper edge files differ; the ligand-network comparison is uncontrolled"
    open_mapping_fingerprint="$(
        "${SCHRODINGER}/run" python3 -c \
            'import json, sys; print(json.load(open(sys.argv[1]))["mapping_fingerprint"])' \
            "${WORK_DIR}/open_map_mappings.json"
    )"
    closed_mapping_fingerprint="$(
        "${SCHRODINGER}/run" python3 -c \
            'import json, sys; print(json.load(open(sys.argv[1]))["mapping_fingerprint"])' \
            "${WORK_DIR}/closed_map_mappings.json"
    )"
    [[ "${open_mapping_fingerprint}" =~ ^[0-9a-f]{64}$ ]] ||
        die "open atom-mapping fingerprint is malformed"
    [[ "${open_mapping_fingerprint}" == "${closed_mapping_fingerprint}" ]] ||
        die "open/closed atom mappings differ; the paired comparison is uncontrolled"

    paired_inputs="${WORK_DIR}/paired_inputs.tsv"
    paired_sidecar="${paired_inputs}.provenance"
    paired_payload="$(
        printf 'schema_version\t1\n'
        printf 'build_schema\t%s\n' "${BUILD_SCHEMA}"
        printf 'suite_release\t%s\n' "${SUITE_ID}"
        printf 'allow_microstate_mismatch\t%s\n' "${ALLOW_MICROSTATE_MISMATCH}"
        printf 'open_receptor_sha256\t%s\n' "$(file_hash "${WORK_DIR}/receptor_open.mae")"
        printf 'closed_receptor_sha256\t%s\n' "$(file_hash "${WORK_DIR}/receptor_closed.mae")"
        printf 'ligand_bundle_sha256\t%s\n' "$(file_hash "${WORK_DIR}/ligands.maegz")"
        printf 'ligand_validation_sha256\t%s\n' \
            "$(file_hash "${WORK_DIR}/ligand_validation.json")"
        printf 'receptor_microstates_sha256\t%s\n' \
            "$(file_hash "${WORK_DIR}/receptor_microstates.json")"
        printf 'open_map_sha256\t%s\n' "$(file_hash "${WORK_DIR}/open_map.fmp")"
        printf 'closed_map_sha256\t%s\n' "$(file_hash "${WORK_DIR}/closed_map.fmp")"
        printf 'open_edge_sha256\t%s\n' "$(file_hash "${WORK_DIR}/open_map.edge")"
        printf 'closed_edge_sha256\t%s\n' "$(file_hash "${WORK_DIR}/closed_map.edge")"
        printf 'open_map_provenance_sha256\t%s\n' \
            "$(file_hash "${WORK_DIR}/open_map.provenance")"
        printf 'closed_map_provenance_sha256\t%s\n' \
            "$(file_hash "${WORK_DIR}/closed_map.provenance")"
        printf 'open_atom_mapping_fingerprint\t%s\n' "${open_mapping_fingerprint}"
        printf 'closed_atom_mapping_fingerprint\t%s\n' "${closed_mapping_fingerprint}"
    )"
    paired_fingerprint="$(printf '%s' "${paired_payload}" | text_hash)"
    if ! require_current_or_build \
        "paired input manifest" "${paired_fingerprint}" \
        "${paired_sidecar}" "${paired_inputs}"; then
        staged_paired="$(mktemp "${WORK_DIR}/.paired-inputs.XXXXXX")"
        trap 'rm -f -- "${staged_paired}"' EXIT
        printf '%s\n' "${paired_payload}" >"${staged_paired}"
        mv "${staged_paired}" "${paired_inputs}"
        write_sidecar \
            "${paired_sidecar}" "${paired_fingerprint}" \
            "shared open/closed FEP+ input cohort" \
            "both receptor, ligand, map, edge, mapping, and microstate artifacts" \
            "${paired_inputs}"
        trap - EXIT
    fi
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo "All requested FEP+ inputs and provenance hashes are current."
else
    echo "Done. Provenance-tracked inputs are under: ${WORK_DIR}"
fi
for conf in "${CONFORMATIONS[@]}"; do
    echo "  [${conf}] receptor_${conf}.mae  ${conf}_pv.mae  ${conf}_map.{fmp,edge}  state_inputs_${conf}.tsv"
done
echo "Inspect: \$SCHRODINGER/run -FROM scisol fmp_info.py -f ${WORK_DIR}/${CONFORMATIONS[0]}_map.fmp"
