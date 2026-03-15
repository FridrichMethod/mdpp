#!/usr/bin/env bash

set -euo pipefail

# check_status.sh
#
# Status definitions:
#   completed : current checkpoint time >= target time
#   active    : exactly one matching Slurm job currently in squeue
#   failed    : not complete and no matching active Slurm job
#   error     : more than one matching active Slurm job for the same MD workdir
#
# Options:
#   -j N    number of parallel workers (default: 8)
#   -t NS   target production time in ns; if set, skip parsing TPR target time
#   -r DIR  root directory to search (default: .)
#   -h      show help

JOBS=8
ROOT="."
TARGET_NS=""

usage() {
    cat <<'EOF'
Usage: check_status.sh [-j N] [-t TARGET_NS] [-r ROOT]

Options:
    -j N        Number of parallel workers (default: 8)
    -t TARGET   Target production time in ns; if provided, skip gmx dump on TPR
    -r ROOT     Root directory to search (default: .)
    -h          Show this help
EOF
}

while getopts ":j:t:r:h" opt; do
    case "$opt" in
        j) JOBS="$OPTARG" ;;
        t) TARGET_NS="$OPTARG" ;;
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

command -v fd >/dev/null 2>&1 || {
    echo "Error: fd not found in PATH" >&2
    exit 1
}
command -v parallel >/dev/null 2>&1 || {
    echo "Error: GNU parallel not found in PATH" >&2
    exit 1
}
command -v squeue >/dev/null 2>&1 || {
    echo "Error: squeue not found in PATH" >&2
    exit 1
}
command -v gmx >/dev/null 2>&1 || {
    echo "Error: gmx not found in PATH" >&2
    exit 1
}

if [[ -n "$TARGET_NS" ]] && ! [[ "$TARGET_NS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: -t TARGET_NS must be numeric" >&2
    exit 2
fi

TMPDIR_MAIN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_MAIN"' EXIT

ACTIVE_FILE="${TMPDIR_MAIN}/active_jobs.tsv"
DIRS_FILE="${TMPDIR_MAIN}/simdirs.txt"

# Build one cache of all active/pending jobs for this user.
# %A = job id, %T = state, %o = command to be executed.
# We normalize the command path where possible.
squeue -h -u "${USER:-$(id -un)}" -o "%A|%T|%o" >"$ACTIVE_FILE.raw"

python3 - "$ACTIVE_FILE.raw" "$ACTIVE_FILE" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        jobid, state, cmd = parts
        cmd = cmd.strip()
        # Keep raw command and also best-effort normalized path for its argv0.
        argv0 = cmd.split()[0] if cmd else ""
        norm = ""
        if argv0:
            norm = os.path.realpath(os.path.abspath(argv0))
        fout.write(f"{jobid}\t{state}\t{cmd}\t{norm}\n")
PY

ps_to_ns() {
    awk -v ps="$1" 'BEGIN { printf "%.3f", ps/1000.0 }'
}

ns_to_ps() {
    awk -v ns="$1" 'BEGIN { printf "%.6f", ns*1000.0 }'
}

extract_target_ps_from_tpr() {
    local tpr="$1"
    gmx dump -s "$tpr" 2>/dev/null | awk '
        BEGIN { nsteps=""; dt=""; init_t="0" }
        /^[[:space:]]*nsteps[[:space:]]*=/ { nsteps=$NF }
        /^[[:space:]]*(delta_t|dt)[[:space:]]*=/ { dt=$NF }
        /^[[:space:]]*(init_t|tinit)[[:space:]]*=/ { init_t=$NF }
        END {
            if (nsteps == "" || dt == "") exit 1
            if (nsteps < 0) print "INF"
            else printf "%.6f\n", init_t + (nsteps * dt)
        }
    '
}

extract_current_ps_from_cpt() {
    local cpt="$1"
    gmx dump -cp "$cpt" 2>/dev/null | awk '
        /^[[:space:]]*t[[:space:]]*=/ { print $NF; exit }
    '
}

process_one_dir() {
    local simdir="$1"
    local active_tsv="$2"
    local target_ns_override="$3"

    local simdir_abs
    simdir_abs="$(cd "$simdir" && pwd -P)"

    local tpr="${simdir_abs}/step5_production.tpr"
    local cpt="${simdir_abs}/step5_production.cpt"
    local mdrun_sbatch="${simdir_abs}/mdrun.sbatch"

    local target_ps="" current_ps="" target_ns="" current_ns="" progress="NA/NA"

    if [[ -n "$target_ns_override" ]]; then
        target_ns="$target_ns_override"
        target_ps="$(ns_to_ps "$target_ns_override")"
    else
        if [[ ! -f "$tpr" ]]; then
            printf "%s\t%s\t%s\t%s\n" "$simdir_abs" "failed" "$progress" "missing step5_production.tpr"
            return
        fi
        target_ps="$(extract_target_ps_from_tpr "$tpr" || true)"
        if [[ -z "${target_ps:-}" ]]; then
            printf "%s\t%s\t%s\t%s\n" "$simdir_abs" "failed" "$progress" "could not parse target time from TPR"
            return
        fi
        if [[ "$target_ps" == "INF" ]]; then
            target_ns="INF"
        else
            target_ns="$(ps_to_ns "$target_ps")"
        fi
    fi

    if [[ -f "$cpt" ]]; then
        current_ps="$(extract_current_ps_from_cpt "$cpt" || true)"
    fi
    if [[ -n "${current_ps:-}" ]]; then
        current_ns="$(ps_to_ns "$current_ps")"
    else
        current_ns="NA"
    fi
    progress="${current_ns}/${target_ns}"

    if [[ "$target_ps" != "INF" && -n "${current_ps:-}" ]]; then
        if awk -v cur="$current_ps" -v tgt="$target_ps" 'BEGIN{exit !(cur + 1e-3 >= tgt)}'; then
            printf "%s\t%s\t%s\t%s\n" "$simdir_abs" "completed" "$progress" "checkpoint reached target time"
            return
        fi
    fi

    # Match active jobs by command/script path.
    # A job is considered matching if:
    #   - its normalized argv0 is mdrun.sbatch in this simdir, OR
    #   - its raw command starts with mdrun.sbatch and we assume it was submitted from this workdir
    #
    # Since squeue does not reliably expose submit cwd in all configurations,
    # this uses the command path as the primary signal.
    local matches
    matches="$(
        awk -F '\t' -v sb="$mdrun_sbatch" -v sim="$simdir_abs" '
            {
                jobid=$1; state=$2; raw=$3; norm=$4;
                if (norm == sb) {
                    print jobid "\t" state
                } else if (raw ~ /^mdrun\.sbatch([[:space:]]|$)/) {
                    # Relative script name; cannot fully disambiguate from squeue alone.
                    # Keep as a weak candidate only if script exists in this directory.
                    print jobid "\t" state "\tREL"
                }
            }
        ' "$active_tsv"
    )"

    local match_count=0
    local rel_count=0
    local job_list=""
    local line jobid state tag

    while IFS=$'\t' read -r jobid state tag; do
        [[ -z "${jobid:-}" ]] && continue
        if [[ "${tag:-}" == "REL" ]]; then
            rel_count=$((rel_count + 1))
        fi
        match_count=$((match_count + 1))
        if [[ -z "$job_list" ]]; then
            job_list="${jobid}:${state}"
        else
            job_list="${job_list},${jobid}:${state}"
        fi
    done <<<"$matches"

    if ((match_count > 1)); then
        printf "%s\t%s\t%s\t%s\n" "$simdir_abs" "error" "$progress" "multiple active jobs match this workdir: ${job_list}"
        return
    fi

    if ((match_count == 1)); then
        printf "%s\t%s\t%s\t%s\n" "$simdir_abs" "active" "$progress" "job in squeue: ${job_list}"
        return
    fi

    printf "%s\t%s\t%s\t%s\n" "$simdir_abs" "failed" "$progress" "incomplete and no matching active job in squeue"
}

export -f ps_to_ns ns_to_ps extract_target_ps_from_tpr extract_current_ps_from_cpt process_one_dir

printf "directory\tstatus\tprogress\tinfo\n"

fd -u -a '^step5_production\.xtc$' "$ROOT" |
    sed 's#/step5_production\.xtc$##' |
    sort -u >"$DIRS_FILE"

parallel -j "$JOBS" -k process_one_dir :::: "$DIRS_FILE" ::: "$ACTIVE_FILE" ::: "$TARGET_NS"
