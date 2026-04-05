#!/usr/bin/env bash

set -euo pipefail

# check_status.sh
#
# Check OpenFE quickrun job status for all transformations and replicas.
#
# Status definitions:
#   completed : result JSON with valid estimate/uncertainty
#   active    : not complete, matching Slurm job in squeue
#   failed    : not complete, no matching Slurm job (or null estimates)
#   error     : multiple matching Slurm jobs for one replica
#
# Options:
#   -j N    number of parallel workers (default: 8)
#   -n N    number of replicas per transformation (default: auto-detect)
#   -r DIR  root directory to search (default: .)
#   -R      restart failed replicas via sbatch
#   -h      show help

# ---- Parse arguments ----

JOBS=8
ROOT="."
REPLICAS=""
RESTART=false

usage() {
    cat <<'EOF'
Usage: check_status.sh [-j N] [-n REPLICAS] [-r ROOT] [-R]

Options:
    -j N        Number of parallel workers (default: 8)
    -n REPLICAS Number of replicas per transformation (default: auto-detect)
    -r ROOT     Root directory (default: .)
    -R          Restart failed replicas via sbatch
    -h          Show this help
EOF
}

while getopts ":j:n:r:Rh" opt; do
    case "$opt" in
        j) JOBS="$OPTARG" ;;
        n) REPLICAS="$OPTARG" ;;
        r) ROOT="$OPTARG" ;;
        R) RESTART=true ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Error: invalid option -$OPTARG" >&2
            usage
            exit 2
            ;;
        :)
            echo "Error: option -$OPTARG requires an argument" >&2
            usage
            exit 2
            ;;
    esac
done

# ---- Validate dependencies ----

for cmd in parallel squeue scontrol; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "Error: ${cmd} not found in PATH" >&2
        exit 1
    }
done

if [[ -n "$REPLICAS" ]] && ! [[ "$REPLICAS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: -n REPLICAS must be a positive integer" >&2
    exit 2
fi

# ---- Setup ----

TMPDIR_MAIN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_MAIN"' EXIT

ROOT_ABS="$(cd "$ROOT" && pwd -P)"
TRANSFORMS_DIR="${ROOT_ABS}/transformations"
RESULTS_DIR="${ROOT_ABS}/results"

if [[ ! -d "$TRANSFORMS_DIR" ]]; then
    echo "Error: transformations directory not found: ${TRANSFORMS_DIR}" >&2
    exit 1
fi

ACTIVE_FILE="${TMPDIR_MAIN}/active_jobs.tsv"
RESTART_FILE="${TMPDIR_MAIN}/restart.tsv"

# ---- Functions: Slurm job cache ----

# Query squeue and scontrol to build a TSV of active jobs for this workdir.
# Output: transform_name \t task_id \t job_id \t state \t array_job_id
build_active_jobs() {
    local root_abs="$1"
    local raw="${TMPDIR_MAIN}/raw_squeue.tsv"
    local tmap="${TMPDIR_MAIN}/tmap.tsv"

    # Collect running jobs whose WorkDir matches our root.
    # %F=ArrayJobId, %K=ArrayTaskId, %A=JobId, %T=State, %Z=WorkDir
    squeue -h -u "${USER:-$(id -un)}" -o "%F|%K|%A|%T|%Z" |
        while IFS='|' read -r ajid taskid jobid state workdir; do
            [[ -z "$jobid" || -z "$state" ]] && continue
            workdir="$(realpath -m -- "$workdir" 2>/dev/null || true)"
            [[ "$workdir" == "$root_abs" ]] || continue
            printf '%s\t%s\t%s\t%s\n' "$ajid" "$taskid" "$jobid" "$state"
        done >"$raw"

    # Map each ArrayJobId -> transformation name via SubmitLine.
    : >"$tmap"
    awk -F'\t' '!seen[$1]++ { print $1 "\t" $3 }' "$raw" |
        while IFS=$'\t' read -r ajid sample_jobid; do
            [[ -z "$ajid" ]] && continue
            submit_line="$(scontrol show job "$sample_jobid" 2>/dev/null |
                sed -n 's/^[[:space:]]*SubmitLine=//p' | head -1)"
            tname=""
            for word in $submit_line; do
                if [[ "$word" == *.json ]]; then
                    tname="$(basename "$word" .json)"
                    break
                fi
            done
            printf '%s\t%s\n' "$ajid" "$tname"
        done >"$tmap"

    # Join raw queue data with transformation names.
    awk -F'\t' '
        NR==FNR { map[$1]=$2; next }
        { print map[$1] "\t" $2 "\t" $3 "\t" $4 "\t" $1 }
    ' "$tmap" "$raw"
}

# ---- Functions: work enumeration ----

# List all (transform_name, replica_id) pairs that need checking.
enumerate_work() {
    local transforms_dir="$1" results_dir="$2" active_tsv="$3" num_replicas="$4"

    find "$transforms_dir" -maxdepth 1 -name '*.json' -type f | sort |
        while IFS= read -r tfile; do
            tname="$(basename "$tfile" .json)"

            if [[ -n "$num_replicas" ]]; then
                count="$num_replicas"
            else
                # Auto-detect replica count from results dir + active jobs.
                max_id=-1

                if [[ -d "${results_dir}/${tname}" ]]; then
                    for rdir in "${results_dir}/${tname}"/replica_*; do
                        [[ -d "$rdir" ]] || continue
                        rid="${rdir##*replica_}"
                        [[ "$rid" =~ ^[0-9]+$ ]] && ((rid > max_id)) && max_id=$rid
                    done
                fi

                while IFS=$'\t' read -r tn tid _ _; do
                    [[ "$tn" == "$tname" && "$tid" =~ ^[0-9]+$ ]] && ((tid > max_id)) && max_id=$tid
                done <"$active_tsv"

                count=$((max_id + 1))
            fi

            for ((i = 0; i < count; i++)); do
                printf '%s\t%s\n' "$tname" "$i"
            done
        done
}

# ---- Functions: per-replica status check (run by GNU parallel) ----

# Extract simulation progress from the most recent real-time analysis YAML.
# Prints e.g. "25.0% (ETA: 1 day, 12:00:00)" or "0%" if no data.
get_progress() {
    local replica_dir="$1"
    local yaml_file
    yaml_file="$(ls -t "$replica_dir"/shared_*/simulation_real_time_analysis.yaml 2>/dev/null | head -1)"
    if [[ -n "$yaml_file" && -f "$yaml_file" ]]; then
        local pct eta
        pct="$(grep 'percent_complete:' "$yaml_file" | tail -1 | awk '{print $2}')"
        eta="$(grep 'estimated_time_remaining:' "$yaml_file" | tail -1 | sed 's/.*estimated_time_remaining: *//')"
        if [[ -n "$pct" ]]; then
            [[ -n "$eta" ]] && echo "${pct}% (ETA: ${eta})" || echo "${pct}%"
            return
        fi
    fi
    echo "0%"
}

# Count matching Slurm jobs for a given transformation + replica.
# Sets $job_count and $joblist in the caller's scope.
match_active_jobs() {
    local tname="$1" replica_id="$2" active_tsv="$3"
    local matches
    matches="$(awk -F'\t' -v tn="$tname" -v tid="$replica_id" \
        '$1 == tn && $2 == tid { print $3 "\t" $4 "\t" $5 }' "$active_tsv")"

    job_count=0
    joblist=""
    local jid jst ajid
    while IFS=$'\t' read -r jid jst ajid; do
        [[ -z "${jid:-}" ]] && continue
        job_count=$((job_count + 1))
        local entry="${jid}(${ajid}_${replica_id}):${jst}"
        joblist="${joblist:+${joblist},}${entry}"
    done <<<"$matches"
}

# Emit a status line. Args: replica_dir color status replica_id info
emit_status() {
    printf '%s\t%s\t%s\t%s\n' "$1" "$2$3$4" "replica_$5" "$6"
}

# Mark a replica for restart (picked up after parallel finishes).
mark_restart() {
    printf '__RESTART__\t%s\t%s\n' "$1" "$2"
}

# Check status of a single (transform_name, replica_id) pair.
process_one() {
    local tname="$1" replica_id="$2" results_dir="$3" active_tsv="$4"

    local c_green=$'\033[32m' c_blue=$'\033[34m'
    local c_red=$'\033[31m' c_yellow=$'\033[33m' c_reset=$'\033[0m'

    local replica_dir="${results_dir}/${tname}/replica_${replica_id}"
    local result_json="${replica_dir}/${tname}.json"

    # 1. Completed: valid result JSON with non-null estimates.
    if [[ -s "$result_json" ]]; then
        if grep -q '"estimate": null' "$result_json" ||
            grep -q '"uncertainty": null' "$result_json"; then
            local progress
            progress="$(get_progress "$replica_dir")"
            emit_status "$replica_dir" "$c_red" "failed" "$c_reset" "$replica_id" \
                "${progress} | result JSON has null estimate/uncertainty"
            mark_restart "$tname" "$replica_id"
            return
        fi
        local est unc
        read -r est unc < <(jq -r '[.estimate.magnitude, .uncertainty.magnitude] | @tsv' "$result_json")
        emit_status "$replica_dir" "$c_green" "completed" "$c_reset" "$replica_id" \
            "ddG = ${est:-?} +/- ${unc:-?} kcal/mol"
        return
    fi

    # 2. Check Slurm queue for matching jobs.
    local job_count joblist
    match_active_jobs "$tname" "$replica_id" "$active_tsv"

    if ((job_count > 1)); then
        emit_status "$replica_dir" "$c_yellow" "error" "$c_reset" "$replica_id" \
            "multiple matching jobs: ${joblist}"
        return
    fi

    local progress
    progress="$(get_progress "$replica_dir")"

    if ((job_count == 1)); then
        emit_status "$replica_dir" "$c_blue" "active" "$c_reset" "$replica_id" \
            "${progress} | ${joblist}"
        return
    fi

    # 3. Failed: no result, no active job.
    emit_status "$replica_dir" "$c_red" "failed" "$c_reset" "$replica_id" \
        "${progress} | incomplete and no matching active job"
    mark_restart "$tname" "$replica_id"
}

export -f get_progress match_active_jobs emit_status mark_restart process_one

# ---- Functions: restart ----

# Resubmit failed replicas grouped by transformation.
restart_failed() {
    local restart_file="$1" transforms_dir="$2" root_abs="$3"
    local script_dir sbatch_script

    script_dir="$(cd "$(dirname "$(readlink -f "$0")")" && pwd -P)"
    sbatch_script="${script_dir}/../quickrun/quickrun.sbatch"

    if [[ ! -f "$sbatch_script" ]]; then
        echo "Error: quickrun.sbatch not found at ${sbatch_script}" >&2
        exit 1
    fi

    echo ""
    awk -F'\t' '
        { replicas[$1] = (replicas[$1] ? replicas[$1] "," : "") $2 }
        END { for (t in replicas) print t "\t" replicas[t] }
    ' "$restart_file" | sort |
        while IFS=$'\t' read -r tname replica_ids; do
            echo "Resubmitting ${tname} replicas [${replica_ids}]"
            sbatch --chdir "$root_abs" --array="${replica_ids}" \
                "$sbatch_script" "${transforms_dir}/${tname}.json" -o "${root_abs}/results"
        done
}

# ---- Main ----

build_active_jobs "$ROOT_ABS" >"$ACTIVE_FILE"

shopt -s nullglob
enumerate_work "$TRANSFORMS_DIR" "$RESULTS_DIR" "$ACTIVE_FILE" "$REPLICAS" \
    >"${TMPDIR_MAIN}/work.tsv"

if [[ ! -s "${TMPDIR_MAIN}/work.tsv" ]]; then
    echo "No replicas found. Use -n to specify replica count." >&2
    exit 0
fi

# Run status checks in parallel; separate status lines from restart markers.
printf 'directory\tstatus\treplica\tinfo\n'

# shellcheck disable=SC1083  # {1} {2} are GNU parallel placeholders
_raw="$(parallel -j "$JOBS" -k --colsep '\t' \
    process_one {1} {2} "$RESULTS_DIR" "$ACTIVE_FILE" \
    <"${TMPDIR_MAIN}/work.tsv")" || true

if [[ -n "$_raw" ]]; then
    printf '%s\n' "$_raw" | { grep -v '^__RESTART__' || true; }
    printf '%s\n' "$_raw" | { grep '^__RESTART__' || true; } | cut -f2- >"$RESTART_FILE"
fi

# Restart if requested.
if [[ "$RESTART" == true && -s "$RESTART_FILE" ]]; then
    restart_failed "$RESTART_FILE" "$TRANSFORMS_DIR" "$ROOT_ABS"
fi
