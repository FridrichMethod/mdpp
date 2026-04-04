#!/usr/bin/env bash

set -euo pipefail

# check_status.sh
#
# Check OpenFE quickrun job status for all transformations and replicas.
#
# Status definitions:
#   completed : result JSON exists (non-empty) in replica directory
#   active    : not complete, matching Slurm job in squeue
#   failed    : not complete, no matching Slurm job in squeue
#   error     : multiple matching Slurm jobs for one replica
#
# Options:
#   -j N    number of parallel workers (default: 8)
#   -n N    number of replicas per transformation (default: auto-detect)
#   -r DIR  root directory to search (default: .)
#   -h      show help

JOBS=8
ROOT="."
REPLICAS=""

usage() {
    cat <<'EOF'
Usage: check_status.sh [-j N] [-n REPLICAS] [-r ROOT]

Options:
    -j N        Number of parallel workers (default: 8)
    -n REPLICAS Number of replicas per transformation (default: auto-detect)
    -r ROOT     Root directory (default: .)
    -h          Show this help
EOF
}

while getopts ":j:n:r:h" opt; do
    case "$opt" in
        j) JOBS="$OPTARG" ;;
        n) REPLICAS="$OPTARG" ;;
        r) ROOT="$OPTARG" ;;
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

command -v parallel >/dev/null 2>&1 || {
    echo "Error: GNU parallel not found in PATH" >&2
    exit 1
}
command -v squeue >/dev/null 2>&1 || {
    echo "Error: squeue not found in PATH" >&2
    exit 1
}
command -v scontrol >/dev/null 2>&1 || {
    echo "Error: scontrol not found in PATH" >&2
    exit 1
}

if [[ -n "$REPLICAS" ]] && ! [[ "$REPLICAS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: -n REPLICAS must be a positive integer" >&2
    exit 2
fi

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

# Build one cache of all active jobs for this user at this working directory.
# Uses scontrol SubmitLine to map each ArrayJobId to its transformation name.
# Output: transform_name \t task_id \t job_id \t state
build_active_jobs() {
    local root_abs="$1"
    local raw="${TMPDIR_MAIN}/raw_squeue.tsv"
    local tmap="${TMPDIR_MAIN}/tmap.tsv"

    # %F = ArrayJobId, %K = ArrayTaskId, %A = JobId, %T = State, %Z = WorkDir
    squeue -h -u "${USER:-$(id -un)}" -o "%F|%K|%A|%T|%Z" |
        while IFS='|' read -r ajid taskid jobid state workdir; do
            [[ -z "$jobid" || -z "$state" ]] && continue
            workdir="$(realpath -m -- "$workdir" 2>/dev/null || true)"
            [[ "$workdir" == "$root_abs" ]] || continue
            printf '%s\t%s\t%s\t%s\n' "$ajid" "$taskid" "$jobid" "$state"
        done >"$raw"

    # For each unique ArrayJobId, extract transformation name from SubmitLine.
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

    # Join: transform_name \t task_id \t job_id \t state \t array_job_id
    awk -F'\t' '
        NR==FNR { map[$1]=$2; next }
        { print map[$1] "\t" $2 "\t" $3 "\t" $4 "\t" $1 }
    ' "$tmap" "$raw"
}

build_active_jobs "$ROOT_ABS" >"$ACTIVE_FILE"

# Enumerate all (transform_name, replica_id) pairs to check.
# Each line: transform_name \t replica_id
WORK_FILE="${TMPDIR_MAIN}/work.tsv"

enumerate_work() {
    local transforms_dir="$1" results_dir="$2" active_tsv="$3" num_replicas="$4"

    find "$transforms_dir" -maxdepth 1 -name '*.json' -type f | sort |
        while IFS= read -r tfile; do
            tname="$(basename "$tfile" .json)"

            if [[ -n "$num_replicas" ]]; then
                count="$num_replicas"
            else
                # Auto-detect: max replica id from results dir + active jobs.
                max_id=-1

                if [[ -d "${results_dir}/${tname}" ]]; then
                    for rdir in "${results_dir}/${tname}"/replica_*; do
                        [[ -d "$rdir" ]] || continue
                        rid="${rdir##*replica_}"
                        if [[ "$rid" =~ ^[0-9]+$ ]]; then
                            if ((rid > max_id)); then
                                max_id=$rid
                            fi
                        fi
                    done
                fi

                while IFS=$'\t' read -r tn tid _ _; do
                    if [[ "$tn" == "$tname" && "$tid" =~ ^[0-9]+$ ]]; then
                        if ((tid > max_id)); then
                            max_id=$tid
                        fi
                    fi
                done <"$active_tsv"

                count=$((max_id + 1))
            fi

            for ((i = 0; i < count; i++)); do
                printf '%s\t%s\n' "$tname" "$i"
            done
        done
}

shopt -s nullglob
enumerate_work "$TRANSFORMS_DIR" "$RESULTS_DIR" "$ACTIVE_FILE" "$REPLICAS" >"$WORK_FILE"

if [[ ! -s "$WORK_FILE" ]]; then
    echo "No replicas found. Use -n to specify replica count." >&2
    exit 0
fi

# Check a single (transform_name, replica_id) pair.
process_one() {
    local tname="$1" replica_id="$2" results_dir="$3" active_tsv="$4"

    local c_green=$'\033[32m' c_blue=$'\033[34m'
    local c_red=$'\033[31m' c_yellow=$'\033[33m' c_reset=$'\033[0m'

    local replica_dir="${results_dir}/${tname}/replica_${replica_id}"
    local result_json="${replica_dir}/${tname}.json"

    # Completed: non-empty result JSON exists with valid estimates.
    if [[ -s "$result_json" ]]; then
        if grep -q '"estimate": null' "$result_json" ||
            grep -q '"uncertainty": null' "$result_json"; then
            printf '%s\t%s\t%s\t%s\n' \
                "$replica_dir" "${c_red}failed${c_reset}" "replica_${replica_id}" \
                "result JSON has null estimate/uncertainty"
            return
        fi
        printf '%s\t%s\t%s\t%s\n' \
            "$replica_dir" "${c_green}completed${c_reset}" "replica_${replica_id}" \
            "result JSON found"
        return
    fi

    # Match active Slurm jobs for this transformation + replica.
    local matches
    matches="$(awk -F'\t' -v tn="$tname" -v tid="$replica_id" \
        '$1 == tn && $2 == tid { print $3 "\t" $4 "\t" $5 }' "$active_tsv")"

    local count=0 joblist="" jid jst ajid
    while IFS=$'\t' read -r jid jst ajid; do
        [[ -z "${jid:-}" ]] && continue
        count=$((count + 1))
        local entry="${jid} (${ajid}_${replica_id}):${jst}"
        if [[ -z "$joblist" ]]; then
            joblist="$entry"
        else
            joblist="${joblist},${entry}"
        fi
    done <<<"$matches"

    if ((count > 1)); then
        printf '%s\t%s\t%s\t%s\n' \
            "$replica_dir" "${c_yellow}error${c_reset}" "replica_${replica_id}" \
            "multiple matching jobs: ${joblist}"
        return
    fi

    if ((count == 1)); then
        printf '%s\t%s\t%s\t%s\n' \
            "$replica_dir" "${c_blue}active${c_reset}" "replica_${replica_id}" \
            "job in squeue: ${joblist}"
        return
    fi

    printf '%s\t%s\t%s\t%s\n' \
        "$replica_dir" "${c_red}failed${c_reset}" "replica_${replica_id}" \
        "incomplete and no matching active job"
}

export -f process_one

printf 'directory\tstatus\treplica\tinfo\n'

# shellcheck disable=SC1083  # {1} {2} are GNU parallel placeholders
parallel -j "$JOBS" -k --colsep '\t' \
    process_one {1} {2} "$RESULTS_DIR" "$ACTIVE_FILE" <"$WORK_FILE"
