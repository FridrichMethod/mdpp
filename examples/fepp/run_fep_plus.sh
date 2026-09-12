#!/usr/bin/env bash
# Launch explicit, repeat-aware native FEP+ jobs from provenance-checked maps.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHRODINGER="$(realpath -m "${SCHRODINGER:-/apps/schrodinger2025-4}")"
WORK_DIR="$(realpath -m "${FEPP_WORK_DIR:-${SCRIPT_DIR}/tmp}")"

CONFORMATIONS=("open")
WORKFLOW="paired"
HOST="localhost"
SUBHOST="localhost"
JOBNAME_BASE=""
PREPARE=0
REPEATS=1
SEED_BASE=2014
FORCEFIELD="OPLS4"
WATER="SPC"
ENSEMBLE="muVT"
TIME_PS="5000"
EQUILIBRATION_TIME_PS="auto"
LAMBDA_WINDOWS=12
SALT_MOLAR="0.0"
MAXJOB=0
HIS267_STATE="HIE"
PREP_PH="7.0"
PREP_RMSD="0.3"
TOPOLOGY="normal"
ALLOW_MICROSTATE_MISMATCH=0
MAX_SEED=2147483647

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
    local conformation_default="${CONFORMATIONS[0]}"
    local jobname_default="${JOBNAME_BASE:-fepp}"
    if [[ "${#CONFORMATIONS[@]}" -eq 2 ]]; then
        conformation_default="both"
    fi
    cat <<EOF
Usage: ${0##*/} [options]

Launch native FEP+ with the documented launcher-controlled protocol choices
explicit. Each job runs in tmp/runs/<jobname>/ and records a read-only manifest
before launch. Production launch returns after submission, not job completion.

Options:
      --workflow MODE       paired | single (default: ${WORKFLOW}).
                            paired retains the matched open/closed cohort;
                            single requires exactly one conformation.
  -c, --conformation NAME   open | closed | both (default: ${conformation_default}).
  -H, --host HOST           Job-control main host (default: ${HOST}).
  -S, --subhost SUBHOST     GPU subhost(s) (default: ${SUBHOST}).
  -j, --jobname-base NAME   Job-name prefix (default: ${jobname_default}).
  -p, --prepare             Prepare only; requires --repeats 1.
      --repeats N           Independent repeats per state (default: ${REPEATS}).
      --seed-base N         First open seed; closed starts at N+1
                            (default: ${SEED_BASE}).
      --forcefield NAME     OPLS4 | OPLS5 (default: ${FORCEFIELD}).
      --water NAME          FEP+ water model (default: ${WATER}).
      --ensemble NAME       muVT | NPT | NVT (default: ${ENSEMBLE}).
      --time-ps VALUE       Production time, >=500 ps (default: ${TIME_PS}).
      --equilibration-time-ps VALUE
                            Complex unrestrained equilibration, >=0 ps
                            (default: auto; muVT=20, NPT/NVT=240).
      --lambda-windows N    Default-protocol windows, >=2 (default: ${LAMBDA_WINDOWS}).
      --salt-molar VALUE    Added salt concentration, >=0 M (default: ${SALT_MOLAR}).
      --maxjob N            Maximum simultaneous subjobs; 0=unlimited (default: ${MAXJOB}).

Input-build controls (must match build_fepp_inputs.sh provenance):
      --his267-state STATE  HID | HIE | HIP | auto (default: ${HIS267_STATE}).
      --ph VALUE            PrepWizard pH (default: ${PREP_PH}).
      --rmsd VALUE          PrepWizard RMSD in A (default: ${PREP_RMSD}).
      --topology TYPE       full | normal | star | windmill (default: ${TOPOLOGY}).
      --allow-microstate-mismatch
                            Launch an input build whose receptor comparison
                            explicitly allowed a microstate mismatch.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workflow)
            require_value "$1" "${2-}"
            case "$2" in
                paired | single) WORKFLOW="$2" ;;
                *) die "workflow must be paired or single; got '$2'" ;;
            esac
            shift 2
            ;;
        -c | --conformation)
            require_value "$1" "${2-}"
            case "$2" in
                open | closed) CONFORMATIONS=("$2") ;;
                both) CONFORMATIONS=("open" "closed") ;;
                *) die "conformation must be open, closed, or both; got '$2'" ;;
            esac
            shift 2
            ;;
        -H | --host)
            require_value "$1" "${2-}"
            HOST="$2"
            shift 2
            ;;
        -S | --subhost)
            require_value "$1" "${2-}"
            SUBHOST="$2"
            shift 2
            ;;
        -j | --jobname-base | --jobname)
            require_value "$1" "${2-}"
            JOBNAME_BASE="$2"
            shift 2
            ;;
        -p | --prepare)
            PREPARE=1
            shift
            ;;
        --repeats)
            require_value "$1" "${2-}"
            REPEATS="$2"
            shift 2
            ;;
        --seed-base)
            require_value "$1" "${2-}"
            SEED_BASE="$2"
            shift 2
            ;;
        --forcefield)
            require_value "$1" "${2-}"
            FORCEFIELD="$2"
            shift 2
            ;;
        --water)
            require_value "$1" "${2-}"
            WATER="$2"
            shift 2
            ;;
        --ensemble)
            require_value "$1" "${2-}"
            ENSEMBLE="$2"
            shift 2
            ;;
        --time-ps)
            require_value "$1" "${2-}"
            TIME_PS="$2"
            shift 2
            ;;
        --equilibration-time-ps)
            require_value "$1" "${2-}"
            EQUILIBRATION_TIME_PS="$2"
            shift 2
            ;;
        --lambda-windows)
            require_value "$1" "${2-}"
            LAMBDA_WINDOWS="$2"
            shift 2
            ;;
        --salt-molar)
            require_value "$1" "${2-}"
            SALT_MOLAR="$2"
            shift 2
            ;;
        --maxjob)
            require_value "$1" "${2-}"
            MAXJOB="$2"
            shift 2
            ;;
        --his267-state)
            require_value "$1" "${2-}"
            HIS267_STATE="$2"
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
        --topology)
            require_value "$1" "${2-}"
            TOPOLOGY="$2"
            shift 2
            ;;
        --allow-microstate-mismatch)
            ALLOW_MICROSTATE_MISMATCH=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -x "${SCHRODINGER}/fep_plus" ]] || die "FEP+ executable not found: ${SCHRODINGER}/fep_plus"
[[ -s "${SCHRODINGER}/version.txt" ]] ||
    die "Suite release metadata not found: ${SCHRODINGER}/version.txt"
[[ -x "${SCRIPT_DIR}/build_fepp_inputs.sh" ]] ||
    die "input builder is not executable: ${SCRIPT_DIR}/build_fepp_inputs.sh"
command -v flock >/dev/null || die "required executable not found: flock"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "repeats must be a positive integer"
[[ "${SEED_BASE}" =~ ^(0|[1-9][0-9]*)$ ]] || die "seed-base must be a nonnegative integer"
[[ "${#SEED_BASE}" -le 10 && "${SEED_BASE}" -lt "${MAX_SEED}" ]] ||
    die "seed-base must be <= $((MAX_SEED - 1))"
[[ "${#REPEATS}" -le 10 ]] || die "repeats is too large"
max_repeats=$(((MAX_SEED - SEED_BASE + 1) / 2))
[[ "${REPEATS}" -le "${max_repeats}" ]] ||
    die "derived repeat seeds must be <= ${MAX_SEED}; maximum repeats=${max_repeats}"
[[ "${LAMBDA_WINDOWS}" =~ ^[1-9][0-9]*$ && "${LAMBDA_WINDOWS}" -ge 2 ]] ||
    die "lambda-windows must be an integer >=2"
[[ "${MAXJOB}" =~ ^(0|[1-9][0-9]*)$ ]] || die "maxjob must be a nonnegative integer"
[[ "${FORCEFIELD}" == "OPLS4" || "${FORCEFIELD}" == "OPLS5" ]] ||
    die "forcefield must be OPLS4 or OPLS5"
case "${WATER}" in
    SPC | SPCE | TIP3P | TIP3P_CHARMM | TIP4P | TIP4PEW | TIP4P2005 | TIP5P | TIP4PD) ;;
    *) die "unsupported FEP+ water model: ${WATER}" ;;
esac
case "${ENSEMBLE}" in
    muVT | NPT | NVT) ;;
    *) die "ensemble must be muVT, NPT, or NVT" ;;
esac
if [[ "${EQUILIBRATION_TIME_PS}" == "auto" ]]; then
    if [[ "${ENSEMBLE}" == "muVT" ]]; then
        EQUILIBRATION_TIME_PS="20"
    else
        EQUILIBRATION_TIME_PS="240"
    fi
fi
for numeric_name in TIME_PS EQUILIBRATION_TIME_PS SALT_MOLAR PREP_PH PREP_RMSD; do
    numeric_value="${!numeric_name}"
    [[ "${numeric_value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
        die "${numeric_name} must be numeric"
done
awk -v value="${TIME_PS}" 'BEGIN { exit !(value >= 500) }' || die "time-ps must be >=500"
awk -v value="${EQUILIBRATION_TIME_PS}" 'BEGIN { exit !(value >= 0) }' ||
    die "equilibration-time-ps must be >=0"
awk -v value="${SALT_MOLAR}" 'BEGIN { exit !(value >= 0) }' || die "salt-molar must be >=0"
[[ "${PREPARE}" -eq 0 || "${REPEATS}" -eq 1 ]] || die "--prepare requires --repeats 1"
if [[ -n "${JOBNAME_BASE}" && ! "${JOBNAME_BASE}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    die "jobname-base may contain only letters, digits, dot, underscore, and hyphen"
fi
if [[ "${WORKFLOW}" == "single" ]]; then
    [[ "${#CONFORMATIONS[@]}" -eq 1 ]] ||
        die "single workflow requires exactly one conformation (open or closed)"
    [[ "${ALLOW_MICROSTATE_MISMATCH}" -eq 0 ]] ||
        die "--allow-microstate-mismatch applies only to the paired workflow"
fi

build_check=(
    "${SCRIPT_DIR}/build_fepp_inputs.sh"
    --check
    -c
)
if [[ "${WORKFLOW}" == "paired" ]]; then
    build_check+=(
        both
        --his267-state "${HIS267_STATE}"
        --ph "${PREP_PH}"
        --rmsd "${PREP_RMSD}"
        --topology "${TOPOLOGY}"
    )
else
    build_check+=(
        "${CONFORMATIONS[0]}"
        --his267-state "${HIS267_STATE}"
        --ph "${PREP_PH}"
        --rmsd "${PREP_RMSD}"
        --topology "${TOPOLOGY}"
    )
fi
if [[ "${ALLOW_MICROSTATE_MISMATCH}" -eq 1 ]]; then
    build_check+=(--allow-microstate-mismatch)
fi
mkdir -p "${WORK_DIR}/runs"
cohort_inputs_sha256=""
for conf in "${CONFORMATIONS[@]}"; do
    if [[ "${conf}" == "open" ]]; then
        state_seed_offset=0
    else
        state_seed_offset=1
    fi
    source_map_file="${WORK_DIR}/${conf}_map.fmp"
    source_map_edge="${WORK_DIR}/${conf}_map.edge"
    source_map_provenance="${WORK_DIR}/${conf}_map.provenance"
    source_map_mappings="${WORK_DIR}/${conf}_map_mappings.json"
    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
        seed=$((SEED_BASE + 2 * (repeat - 1) + state_seed_offset))
        if [[ -n "${JOBNAME_BASE}" ]]; then
            if [[ "${#CONFORMATIONS[@]}" -eq 1 && "${REPEATS}" -eq 1 ]]; then
                jobname="${JOBNAME_BASE}"
            else
                printf -v jobname '%s_%s_r%02d' "${JOBNAME_BASE}" "${conf}" "${repeat}"
            fi
        elif [[ "${REPEATS}" -eq 1 ]]; then
            jobname="fepp_${conf}"
        else
            printf -v jobname 'fepp_%s_r%02d' "${conf}" "${repeat}"
        fi
        if [[ "${PREPARE}" -eq 1 ]]; then
            jobname="${jobname}_prepare"
        fi
        run_dir="${WORK_DIR}/runs/${jobname}"

        # Hold the same lock as the builder while re-checking and snapshotting.
        # The launched input is then immutable with respect to later rebuilds.
        exec {build_lock_fd}<"${WORK_DIR}"
        flock -x "${build_lock_fd}"
        echo "Verifying ${WORKFLOW} input provenance before snapshotting ${jobname}..."
        FEPP_BUILD_LOCK_HELD=1 "${build_check[@]}"
        [[ ! -e "${run_dir}" ]] || die "run directory already exists: ${run_dir}"
        staged_run_dir="$(mktemp -d "${WORK_DIR}/runs/.${jobname}.XXXXXX")"
        trap 'rm -rf -- "${staged_run_dir}"' EXIT
        # Job Server stages relative inputs into scratch. The native workflow
        # also returns its input map, so keep the immutable snapshot separate.
        launch_map_file="working_map.fmp"
        map_file="${staged_run_dir}/input_map.fmp"
        map_edge="${staged_run_dir}/input_map.edge"
        map_provenance="${staged_run_dir}/input_map.provenance"
        map_mappings="${staged_run_dir}/input_map_mappings.json"
        ligand_bundle="${staged_run_dir}/input_ligands.maegz"
        ligand_validation="${staged_run_dir}/input_ligand_validation.json"
        snapshot_sources=(
            "${source_map_file}"
            "${source_map_edge}"
            "${source_map_provenance}"
            "${source_map_mappings}"
            "${WORK_DIR}/ligands.maegz"
            "${WORK_DIR}/ligand_validation.json"
        )
        snapshot_targets=(
            "${map_file}"
            "${map_edge}"
            "${map_provenance}"
            "${map_mappings}"
            "${ligand_bundle}"
            "${ligand_validation}"
        )
        if [[ "${WORKFLOW}" == "paired" ]]; then
            receptor_microstates="${staged_run_dir}/input_receptor_microstates.json"
            paired_inputs="${staged_run_dir}/input_paired_inputs.tsv"
            snapshot_sources+=(
                "${WORK_DIR}/receptor_microstates.json"
                "${WORK_DIR}/paired_inputs.tsv"
            )
            snapshot_targets+=("${receptor_microstates}" "${paired_inputs}")
        else
            state_inputs="${staged_run_dir}/input_state_inputs.tsv"
            snapshot_sources+=("${WORK_DIR}/state_inputs_${conf}.tsv")
            snapshot_targets+=("${state_inputs}")
        fi
        for index in "${!snapshot_sources[@]}"; do
            [[ -s "${snapshot_sources[index]}" ]] ||
                die "missing provenance input: ${snapshot_sources[index]}"
            cp -- "${snapshot_sources[index]}" "${snapshot_targets[index]}"
        done
        chmod 0444 "${snapshot_targets[@]}"
        cp -- "${map_file}" "${staged_run_dir}/${launch_map_file}"
        chmod 0644 "${staged_run_dir}/${launch_map_file}"
        if [[ "${WORKFLOW}" == "paired" ]]; then
            inputs_sha256="$(sha256sum "${paired_inputs}" | awk '{print $1}')"
        else
            inputs_sha256="$(sha256sum "${state_inputs}" | awk '{print $1}')"
        fi
        if [[ -z "${cohort_inputs_sha256}" ]]; then
            cohort_inputs_sha256="${inputs_sha256}"
        elif [[ "${inputs_sha256}" != "${cohort_inputs_sha256}" ]]; then
            die "input cohort changed while snapshotting this launch cohort"
        fi

        cmd=(
            "${SCHRODINGER}/fep_plus"
            -HOST "${HOST}"
            -SUBHOST "${SUBHOST}"
            -JOBNAME "${jobname}"
            -ff "${FORCEFIELD}"
            -custom-charge-mode assign
            -seed "${seed}"
            -water "${WATER}"
            -ensemble "${ENSEMBLE}"
            -time "${TIME_PS}"
            -equilibration-time "${EQUILIBRATION_TIME_PS}"
            -lambda-windows "${LAMBDA_WINDOWS}"
            -salt "${SALT_MOLAR}"
            -maxjob "${MAXJOB}"
            "${launch_map_file}"
        )
        if [[ "${PREPARE}" -eq 1 ]]; then
            cmd+=(-prepare)
        fi

        command_text=""
        printf -v command_text '%q ' "${cmd[@]}"
        suite_release="$(tr '\t\r\n' '   ' <"${SCHRODINGER}/version.txt")"
        [[ "${suite_release}" =~ [^[:space:]] ]] || die "Suite release metadata is empty"
        atom_mapping_fingerprint="$(
            "${SCHRODINGER}/run" python3 -c \
                'import json, sys; print(json.load(open(sys.argv[1]))["mapping_fingerprint"])' \
                "${map_mappings}"
        )"
        [[ "${atom_mapping_fingerprint}" =~ ^[0-9a-f]{64}$ ]] ||
            die "snapshot atom-mapping fingerprint is malformed"
        {
            if [[ "${WORKFLOW}" == "paired" ]]; then
                printf 'schema_version\t1\n'
                printf 'workflow_type\tpaired_open_closed\n'
            else
                printf 'schema_version\t2\n'
                printf 'workflow_type\tsingle_rbfe\n'
            fi
            printf 'created_utc\t%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
            printf 'schrodinger_root\t%s\n' "${SCHRODINGER}"
            printf 'suite_release\t%s\n' "${suite_release}"
            printf 'jobname\t%s\n' "${jobname}"
            printf 'state\t%s\n' "${conf}"
            printf 'repeat\t%s\n' "${repeat}"
            printf 'seed\t%s\n' "${seed}"
            printf 'host\t%s\n' "${HOST}"
            printf 'subhost\t%s\n' "${SUBHOST}"
            printf 'prepare_only\t%s\n' "${PREPARE}"
            printf 'forcefield\t%s\n' "${FORCEFIELD}"
            printf 'custom_charge_mode\tassign\n'
            printf 'water\t%s\n' "${WATER}"
            printf 'ensemble\t%s\n' "${ENSEMBLE}"
            printf 'time_ps\t%s\n' "${TIME_PS}"
            printf 'equilibration_time_ps\t%s\n' "${EQUILIBRATION_TIME_PS}"
            printf 'lambda_windows\t%s\n' "${LAMBDA_WINDOWS}"
            printf 'salt_molar\t%s\n' "${SALT_MOLAR}"
            printf 'maxjob\t%s\n' "${MAXJOB}"
            printf 'input_his267_state\t%s\n' "${HIS267_STATE}"
            printf 'input_prep_ph\t%s\n' "${PREP_PH}"
            printf 'input_prep_rmsd_A\t%s\n' "${PREP_RMSD}"
            printf 'input_map_topology\t%s\n' "${TOPOLOGY}"
            if [[ "${WORKFLOW}" == "paired" ]]; then
                printf 'input_allow_microstate_mismatch\t%s\n' \
                    "${ALLOW_MICROSTATE_MISMATCH}"
            fi
            printf 'source_map_path\t%s\n' "${source_map_file}"
            printf 'launch_map_file\t%s\n' "${launch_map_file}"
            printf 'launch_map_initial_sha256\t%s\n' \
                "$(sha256sum "${staged_run_dir}/${launch_map_file}" | awk '{print $1}')"
            printf 'map_sha256\t%s\n' "$(sha256sum "${map_file}" | awk '{print $1}')"
            printf 'edge_sha256\t%s\n' "$(sha256sum "${map_edge}" | awk '{print $1}')"
            printf 'map_provenance_sha256\t%s\n' \
                "$(sha256sum "${map_provenance}" | awk '{print $1}')"
            printf 'atom_mapping_fingerprint\t%s\n' "${atom_mapping_fingerprint}"
            printf 'atom_mapping_json_sha256\t%s\n' \
                "$(sha256sum "${map_mappings}" | awk '{print $1}')"
            printf 'ligand_bundle_sha256\t%s\n' \
                "$(sha256sum "${ligand_bundle}" | awk '{print $1}')"
            printf 'ligand_validation_sha256\t%s\n' \
                "$(sha256sum "${ligand_validation}" | awk '{print $1}')"
            if [[ "${WORKFLOW}" == "paired" ]]; then
                printf 'receptor_microstates_sha256\t%s\n' \
                    "$(sha256sum "${receptor_microstates}" | awk '{print $1}')"
                printf 'paired_inputs_sha256\t%s\n' "${inputs_sha256}"
            else
                printf 'state_inputs_sha256\t%s\n' "${inputs_sha256}"
            fi
            printf 'command\t%s\n' "${command_text% }"
        } >"${staged_run_dir}/manifest.tsv"
        chmod 0444 "${staged_run_dir}/manifest.tsv"
        mv "${staged_run_dir}" "${run_dir}"
        trap - EXIT
        flock -u "${build_lock_fd}"
        exec {build_lock_fd}>&-

        printf 'Running in %q:\n  ' "${run_dir}"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        (
            cd "${run_dir}"
            "${cmd[@]}" 2>&1 | tee launcher.log
        )
        if [[ "${PREPARE}" -eq 1 ]]; then
            echo "Preparation command returned successfully; no simulation was submitted."
        else
            echo "Launcher returned; this does not establish simulation completion."
            echo "Use the JobId in ${run_dir}/launcher.log with wait_for_fep_plus.py before export."
        fi
    done
done
