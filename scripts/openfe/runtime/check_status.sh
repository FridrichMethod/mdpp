#!/usr/bin/env bash

set -euo pipefail

# check_status.sh
#
# Check OpenFE quickrun job status for all transformations and replicas.
#
# Status definitions:
#   completed : result JSON with valid estimate/uncertainty
#   active    : not complete, matching Slurm job in squeue (or recently
#                preempted and likely requeuing)
#   failed    : not complete, no matching Slurm job, no recent preemption
#   error     : multiple matching Slurm jobs for one replica
#
# Options:
#   -j N    number of parallel workers (default: 8)
#   -r DIR  root directory to search (default: .)
#   -R      restart failed replicas via sbatch
#   -h      show help

# ---- Parse arguments ----

JOBS=8
ROOT="."
RESTART=false

# Grace period (minutes) for preemption detection.  On shared partitions like
# "owners", SLURM preempts and requeues jobs automatically.  During the requeue
# transition there is a brief window where the job is invisible to squeue.  If
# check_status runs during that window it would misclassify the replica as
# "failed" and (with -R) resubmit a duplicate.  To avoid this, we query sacct
# for jobs preempted within this grace period and treat them as still active.
# Set to roughly half the monitor check interval.  Override via the
# OPENFE_PREEMPT_GRACE_MINUTES environment variable; the 20 min default is
# generous for busy "owners" nodes where the requeue can lag behind the
# PREEMPTED event by several minutes.
PREEMPT_GRACE_MINUTES="${OPENFE_PREEMPT_GRACE_MINUTES:-20}"

usage() {
    cat <<'EOF'
Usage: check_status.sh [-j N] [-r ROOT] [-R] [-h]

Options:
    -j, --jobs N       Number of parallel workers (default: 8)
    -r, --root DIR     Root directory (default: .)
    -R, --restart      Restart failed replicas via sbatch
    -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j | --jobs)
            [[ $# -lt 2 ]] && {
                echo "Error: $1 requires an argument" >&2
                usage
                exit 2
            }
            JOBS="$2"
            shift 2
            ;;
        -r | --root)
            [[ $# -lt 2 ]] && {
                echo "Error: $1 requires an argument" >&2
                usage
                exit 2
            }
            ROOT="$2"
            shift 2
            ;;
        -R | --restart)
            RESTART=true
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

# ---- Validate dependencies ----

for cmd in parallel squeue scontrol jq realpath; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "Error: ${cmd} not found in PATH" >&2
        exit 1
    }
done
# sacct (preemption grace) and flock (restart serialization) are optional and
# guarded at their call sites, so they are not required here.

# ---- Setup ----

SCRIPTS_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd -P)"
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
    squeue -r -h -u "${USER:-$(id -un)}" -o "%F|%K|%A|%T|%Z" |
        while IFS='|' read -r ajid taskid jobid state workdir; do
            [[ -z "$jobid" || -z "$state" ]] && continue
            workdir="$(realpath -m -- "$workdir" 2>/dev/null || true)"
            [[ "$workdir" == "$root_abs" ]] || continue
            printf '%s\t%s\t%s\t%s\n' "$ajid" "$taskid" "$jobid" "$state"
        done >"$raw"

    # Phase 2 -- preemption detection via sacct.
    #
    # Problem: when SLURM preempts and requeues an array task, there is a
    # transient window (typically seconds, but could be longer under
    # scheduler load) during which the task vanishes from squeue.  If we
    # check during that window, job_count == 0 and the replica looks
    # "failed", triggering a duplicate submission with -R.
    #
    # Solution: query sacct for PREEMPTED events within the grace period.
    # For each (ArrayJobId, ArrayTaskId) that sacct reports as preempted
    # but that does NOT already appear in squeue (i.e. the requeue hasn't
    # surfaced yet), inject a synthetic entry with state "REQUEUING".
    # Downstream, process_one sees job_count == 1 and classifies the
    # replica as "active" -- no restart marker is emitted.
    #
    # Flags:
    #   --duplicates   show all records, not just the latest per job
    #   --state=PREEMPTED  only preemption events
    #   -S "now-Nminutes"  limit to the grace window
    #   -n -P              no header, pipe-delimited
    #
    # Graceful degradation: if sacct is not on PATH or the grace period
    # is 0, this block is silently skipped and the script behaves as
    # before (squeue-only detection).
    if ((PREEMPT_GRACE_MINUTES > 0)) && command -v sacct >/dev/null 2>&1; then
        local sacct_raw="${TMPDIR_MAIN}/sacct_preempted.tsv"
        sacct -u "${USER:-$(id -un)}" --duplicates \
            -S "now-${PREEMPT_GRACE_MINUTES}minutes" --state=PREEMPTED \
            -n -P --format="JobID%30,State%15,WorkDir%300" |
            while IFS='|' read -r jid_raw state workdir; do
                # Skip sub-step records (e.g. "12345_0.batch") and
                # non-array jobs (no underscore in the JobID).
                [[ "$jid_raw" == *.* || "$jid_raw" != *_* ]] && continue
                workdir="$(realpath -m -- "$workdir" 2>/dev/null || true)"
                [[ "$workdir" == "$root_abs" ]] || continue
                local ajid="${jid_raw%%_*}"
                local taskid="${jid_raw#*_}"
                printf '%s\t%s\n' "$ajid" "$taskid"
            done | sort -u >"$sacct_raw"

        # For each preempted task not already in the squeue snapshot,
        # append a synthetic REQUEUING entry so it counts as active.
        # shellcheck disable=SC2094  # awk reads $raw; loop appends new entries
        while IFS=$'\t' read -r ajid taskid; do
            if ! awk -F'\t' -v a="$ajid" -v t="$taskid" \
                '$1==a && $2==t { found=1; exit } END { exit !found }' \
                "$raw" 2>/dev/null; then
                printf '%s\t%s\t%s\t%s\n' \
                    "$ajid" "$taskid" "${ajid}_${taskid}" "REQUEUING"
            fi
        done <"$sacct_raw" >>"$raw"
    fi

    # Map each ArrayJobId -> transformation name via SubmitLine.  Try every
    # distinct jobid for the array (not just the first): when a whole array is
    # mid-requeue after preemption the first/synthetic id can be unresolvable
    # while another element still resolves.  If none resolve, label the array
    # "__UNKNOWN__" so its rows never masquerade as (or get matched to) a real
    # transformation -- and so the restart path can detect unattributable jobs.
    : >"$tmap"
    awk -F'\t' '{ ids[$1] = (ids[$1] ? ids[$1] " " : "") $3 }
                END { for (a in ids) print a "\t" ids[a] }' "$raw" |
        while IFS=$'\t' read -r ajid jobids; do
            [[ -z "$ajid" ]] && continue
            tname=""
            read -r -a jid_arr <<<"$jobids"
            for jid in "${jid_arr[@]}"; do
                # `|| true`: scontrol on an unresolvable jobid (e.g. a job that
                # just left the queue, or a synthetic requeue id) exits non-zero,
                # which under `set -e`+pipefail would otherwise abort the whole
                # script here.  Tolerate it and fall through to __UNKNOWN__.
                submit_line="$(scontrol show job "$jid" 2>/dev/null |
                    sed -n 's/^[[:space:]]*SubmitLine=//p' | head -1 || true)"
                if [[ "$submit_line" =~ ([^[:space:]]+\.json) ]]; then
                    tname="$(basename "${BASH_REMATCH[1]}" .json)"
                    break
                fi
            done
            printf '%s\t%s\n' "$ajid" "${tname:-__UNKNOWN__}"
        done >"$tmap"

    # Join raw queue data with transformation names.
    awk -F'\t' '
        NR==FNR { map[$1]=$2; next }
        { print map[$1] "\t" $2 "\t" $3 "\t" $4 "\t" $1 }
    ' "$tmap" "$raw"
}

# ---- Functions: work enumeration ----

# List all (transform_name, replica_id) pairs that need checking.
# The replica set is auto-detected per transformation from the UNION of:
#   1. Existing replica_* directories under results/<tname>/
#   2. Active SLURM array task IDs for that transformation in squeue
# Only ids that actually appear in one of those sources are enumerated (NOT
# 0..max_id), so a sparse or offset replica set left by a prior run cannot
# fabricate phantom "failed" replicas that -R would resubmit as new jobs.
enumerate_work() {
    local transforms_dir="$1" results_dir="$2" active_tsv="$3"

    find "$transforms_dir" -maxdepth 1 -name '*.json' -type f | sort |
        while IFS= read -r tfile; do
            tname="$(basename "$tfile" .json)"

            ids=""

            # Source 1: replica directories on disk.
            if [[ -d "${results_dir}/${tname}" ]]; then
                for rdir in "${results_dir}/${tname}"/replica_*; do
                    [[ -d "$rdir" ]] || continue
                    rid="${rdir##*replica_}"
                    # 10# forces base-10 so a zero-padded id (e.g. 08) is not
                    # misparsed as octal by bash arithmetic.
                    [[ "$rid" =~ ^[0-9]+$ ]] && ids+="$((10#$rid))"$'\n'
                done
            fi

            # Source 2: active SLURM array task IDs.
            while IFS=$'\t' read -r tn tid _ _; do
                [[ "$tn" == "$tname" && "$tid" =~ ^[0-9]+$ ]] && ids+="$((10#$tid))"$'\n'
            done <"$active_tsv"

            [[ -z "$ids" ]] && continue
            printf '%s' "$ids" | sort -n -u |
                while IFS= read -r rid; do
                    printf '%s\t%s\n' "$tname" "$rid"
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
        pct="$(grep -E '^[[:space:]]*percent_complete:' "$yaml_file" | tail -1 | awk '{print $2}')"
        eta="$(grep -E '^[[:space:]]*estimated_time_remaining:' "$yaml_file" | tail -1 | sed 's/.*estimated_time_remaining: *//')"
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

# Emit a status line. Args: replica_dir color status reset replica_id info
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

    # 1. Completed: result JSON that parses and carries non-null estimates.
    #    jq is the source of truth (robust to JSON whitespace), and a
    #    truncated/mid-write file (jq parse error -> no output -> read fails)
    #    falls through to the queue check below instead of being reported as
    #    "completed ddG = ?".  A null/empty estimate likewise falls through:
    #    if a job is still active it is mid-write (-> active), and only a
    #    missing/incomplete result with no active job is classified failed.
    if [[ -s "$result_json" ]]; then
        local est unc
        if read -r est unc < <(
            jq -er '[.estimate.magnitude, .uncertainty.magnitude] | @tsv' \
                "$result_json" 2>/dev/null
        ) && [[ -n "$est" && "$est" != "null" && -n "$unc" && "$unc" != "null" ]]; then
            emit_status "$replica_dir" "$c_green" "completed" "$c_reset" "$replica_id" \
                "ddG = ${est} +/- ${unc} kcal/mol"
            return
        fi
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

    # 3. No active job.  If this replica has exhausted its restart attempts AND
    #    has actually written a (failed) result, mark it "abandoned" (terminal)
    #    and do NOT queue another restart -- this stops both endless resubmission
    #    and an endless monitor loop.  Requiring a result file is important: a
    #    preempted/requeuing job is only transiently invisible and has no result
    #    (quickrun.sbatch removes it at each start and writes it only on
    #    conclusion), so it must NOT be abandoned -- it falls through to "failed"
    #    and is retried.  The cap check mirrors restart_failed exactly.
    local attempts_file max_attempts attempts
    attempts_file="$(dirname "$results_dir")/.openfe_restart_attempts.tsv"
    max_attempts="${OPENFE_MAX_RESTART_ATTEMPTS:-5}"
    attempts=0
    if [[ -f "$attempts_file" ]]; then
        attempts="$(awk -F'\t' -v t="$tname" -v r="$replica_id" \
            '$1 == t && $2 == r { c = $3 } END { print c + 0 }' "$attempts_file")"
    fi
    if ((max_attempts > 0 && attempts >= max_attempts)) && [[ -s "$result_json" ]]; then
        emit_status "$replica_dir" "$c_red" "abandoned" "$c_reset" "$replica_id" \
            "${progress} | gave up after ${attempts} failed attempts (cap ${max_attempts})"
        return
    fi
    emit_status "$replica_dir" "$c_red" "failed" "$c_reset" "$replica_id" \
        "${progress} | incomplete and no matching active job"
    mark_restart "$tname" "$replica_id"
}

export -f get_progress match_active_jobs emit_status mark_restart process_one
# Export the cap too, so process_one (run by GNU parallel in a child shell) and
# restart_failed agree on which replicas are "abandoned" vs still restartable.
export OPENFE_MAX_RESTART_ATTEMPTS="${OPENFE_MAX_RESTART_ATTEMPTS:-5}"

# ---- Functions: restart ----

# Resubmit failed replicas grouped by transformation.
#
# A per-replica attempt cap stops a permanently-failing replica from being
# resubmitted forever.  Counts persist in a state file under the project root
# (append-only; the latest line per replica wins).  Override the cap via
# OPENFE_MAX_RESTART_ATTEMPTS (default 5; set 0 to disable the cap).
restart_failed() {
    local restart_file="$1" transforms_dir="$2" root_abs="$3"
    local sbatch_script="${SCRIPTS_DIR}/../quickrun/quickrun.sbatch"
    local max_attempts="${OPENFE_MAX_RESTART_ATTEMPTS:-5}"
    local attempts_file="${root_abs}/.openfe_restart_attempts.tsv"

    if [[ ! -f "$sbatch_script" ]]; then
        echo "Error: quickrun.sbatch not found at ${sbatch_script}" >&2
        exit 1
    fi
    [[ -f "$attempts_file" ]] || : >"$attempts_file"

    echo ""
    awk -F'\t' '
        { replicas[$1] = (replicas[$1] ? replicas[$1] "," : "") $2 }
        END { for (t in replicas) print t "\t" replicas[t] }
    ' "$restart_file" | sort |
        while IFS=$'\t' read -r tname replica_ids; do
            # Drop replicas that have already hit the attempt cap so a
            # permanently-failing replica is not resubmitted forever.
            keep=""
            IFS=',' read -r -a rid_arr <<<"$replica_ids"
            for rid in "${rid_arr[@]}"; do
                attempts="$(awk -F'\t' -v t="$tname" -v r="$rid" \
                    '$1 == t && $2 == r { c = $3 } END { print c + 0 }' "$attempts_file")"
                if ((max_attempts > 0 && attempts >= max_attempts)); then
                    echo "Giving up on ${tname} replica ${rid} after ${attempts}" \
                        "restart attempts (cap ${max_attempts}); not resubmitting" >&2
                    continue
                fi
                keep="${keep:+${keep},}${rid}"
            done
            [[ -z "$keep" ]] && continue

            echo "Resubmitting ${tname} replicas [${keep}]"
            # Do not let one sbatch failure (e.g. hitting a QOS submit cap)
            # abort the loop under set -e; report it and keep going so the
            # remaining transformations are still resubmitted.  Only record an
            # attempt when the submission actually succeeded.
            if ! sbatch --chdir "$root_abs" --array="${keep}" \
                "$sbatch_script" "${transforms_dir}/${tname}.json" -o "${root_abs}/results"; then
                echo "Warning: sbatch failed for ${tname} replicas [${keep}]" \
                    "(e.g. QOS submit cap); will retry next cycle" >&2
                continue
            fi

            IFS=',' read -r -a kept_arr <<<"$keep"
            for rid in "${kept_arr[@]}"; do
                attempts="$(awk -F'\t' -v t="$tname" -v r="$rid" \
                    '$1 == t && $2 == r { c = $3 } END { print c + 0 }' "$attempts_file")"
                printf '%s\t%s\t%s\n' "$tname" "$rid" "$((attempts + 1))" >>"$attempts_file"
            done
        done
}

# ---- Main ----

build_active_jobs "$ROOT_ABS" >"$ACTIVE_FILE"

shopt -s nullglob
enumerate_work "$TRANSFORMS_DIR" "$RESULTS_DIR" "$ACTIVE_FILE" \
    >"${TMPDIR_MAIN}/work.tsv"

if [[ ! -s "${TMPDIR_MAIN}/work.tsv" ]]; then
    echo "No replicas found." >&2
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
    # Validate the resubmission script up front, at top level, so a missing
    # quickrun.sbatch fails loudly here instead of being swallowed (as a fake
    # "lock unavailable") by the restart subshell below.
    if [[ ! -f "${SCRIPTS_DIR}/../quickrun/quickrun.sbatch" ]]; then
        echo "Error: quickrun.sbatch not found at" \
            "${SCRIPTS_DIR}/../quickrun/quickrun.sbatch; cannot restart." >&2
        exit 1
    fi
    # Safety 1: if any active job could not be attributed to a transformation
    # (e.g. an entire array is mid-requeue after preemption), skip restarts
    # this cycle rather than risk duplicate submissions.  The next pass retries
    # once the queue settles.
    unknown_active="$(awk -F'\t' '$1 == "__UNKNOWN__"' "$ACTIVE_FILE" | wc -l)"
    if ((unknown_active > 0)); then
        echo "Skipping restarts this cycle: ${unknown_active} active job(s) could not be" \
            "attributed to a transformation (likely mid-requeue after preemption)." >&2
    elif command -v flock >/dev/null 2>&1; then
        # Safety 2: serialize restarts via a per-user lock so a concurrent
        # check_status -R (manual run or another monitor pass) cannot submit
        # duplicate restarts for the same replicas.  The lock file is per-user
        # (the realistic overlap is one user's monitor vs. a manual run) so a
        # group-mate's lock file in this shared dir can't block us.  The fd is
        # opened by a SUBSHELL redirection, not `exec`, so a failure to open the
        # lock (or a busy lock) degrades gracefully instead of killing the
        # script the way a special-builtin `exec` redirection error would.
        lockfile="${ROOT_ABS}/.openfe_restart.${USER:-$(id -un)}.lock"
        if ! (
            flock -n 9 || exit 3
            restart_failed "$RESTART_FILE" "$TRANSFORMS_DIR" "$ROOT_ABS"
        ) 9>"$lockfile"; then
            echo "Restart skipped for ${ROOT_ABS}: another restart is in progress" \
                "or the lock could not be acquired; will retry next cycle." >&2
        fi
    else
        restart_failed "$RESTART_FILE" "$TRANSFORMS_DIR" "$ROOT_ABS"
    fi
fi
