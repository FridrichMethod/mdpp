#!/bin/bash
#SBATCH --job-name=gromacs_monitor
#SBATCH --partition=normal,owners
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${USER}@stanford.edu
#SBATCH --output=logs/monitor_%j.out
#SBATCH --error=logs/monitor_%j.err

set -euo pipefail

# monitor.sbatch
#
# Monitor GROMACS simulations, restart failures, send email reports.
# Self-resubmits via SLURM until all jobs complete.
#
# Can be run directly (bash monitor.sbatch ...) or submitted (sbatch monitor.sbatch ...).
# When submitting via sbatch for the first time, run from this directory or use --chdir:
#   cd scripts/gromacs/runtime && sbatch monitor.sbatch -d /path/to/project
#   sbatch --chdir scripts/gromacs/runtime scripts/gromacs/runtime/monitor.sbatch -d /path/to/project
# Self-resubmission handles --chdir automatically.
#
# Options:
#   -d DIR [DIR ...]  Project directories to monitor (required)
#   -t NS             Target production time in ns (passed to check_status.sh)
#   -e EMAIL          Notification email (default: ${USER}@stanford.edu)
#   -i HOURS          Hours between checks (default: 1)
#   -n                Dry run: report only, no restarts or resubmissions
#   -h                Show help

# Under SLURM, BASH_SOURCE points to the spool copy — use the working
# directory set by --chdir instead.  When run directly with bash,
# BASH_SOURCE is reliable.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SCRIPTS_DIR="$(pwd -P)"
else
    SCRIPTS_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)"
fi
CHECK_STATUS="${SCRIPTS_DIR}/check_status.sh"

# ---- Parse arguments ----

DIRS=()
EMAIL="${USER}@stanford.edu"
INTERVAL=1
DRY_RUN=false
TARGET_NS=""

usage() {
    cat <<'EOF'
Usage: monitor.sbatch -d DIR [DIR ...] [OPTIONS]

Options:
    -d DIR [DIR ...]  Project directories to monitor (required)
    -t NS             Target production time in ns (passed to check_status.sh)
    -e EMAIL          Notification email (default: ${USER}@stanford.edu)
    -i HOURS          Interval between checks in hours (default: 1)
    -n                Dry run: report only, no restarts or resubmissions
    -h                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d)
            shift
            [[ $# -eq 0 || "$1" == -* ]] && {
                echo "Error: -d requires at least one directory" >&2
                exit 2
            }
            while [[ $# -gt 0 && "$1" != -* ]]; do
                DIRS+=("$1")
                shift
            done
            ;;
        -t)
            [[ $# -lt 2 ]] && {
                echo "Error: -t requires an argument" >&2
                exit 2
            }
            TARGET_NS="$2"
            if ! [[ "$TARGET_NS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                echo "Error: -t TARGET_NS must be numeric, got: ${TARGET_NS}" >&2
                exit 2
            fi
            shift 2
            ;;
        -e)
            [[ $# -lt 2 ]] && {
                echo "Error: -e requires an argument" >&2
                exit 2
            }
            EMAIL="$2"
            shift 2
            ;;
        -i)
            [[ $# -lt 2 ]] && {
                echo "Error: -i requires an argument" >&2
                exit 2
            }
            INTERVAL="$2"
            if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: -i INTERVAL must be a positive integer, got: ${INTERVAL}" >&2
                exit 2
            fi
            shift 2
            ;;
        -n)
            DRY_RUN=true
            shift
            ;;
        -h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ ${#DIRS[@]} -eq 0 ]]; then
    echo "Error: -d DIR is required" >&2
    usage
    exit 2
fi

# Resolve directories to absolute paths so that resubmission with --chdir
# does not break relative paths.
for i in "${!DIRS[@]}"; do
    if [[ -d "${DIRS[$i]}" ]]; then
        DIRS[$i]="$(cd "${DIRS[$i]}" && pwd -P)"
    fi
done

# ---- Helpers ----

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }

send_email() {
    local subject="$1" body="$2" recipient="$3"
    if command -v mail >/dev/null 2>&1; then
        printf '%s\n' "$body" | mail -s "$subject" "$recipient"
    elif command -v sendmail >/dev/null 2>&1; then
        printf 'To: %s\nSubject: %s\nContent-Type: text/plain; charset=UTF-8\n\n%s\n' \
            "$recipient" "$subject" "$body" | sendmail "$recipient"
    else
        echo "WARNING: No mail command available. Email report:" >&2
        echo "Subject: $subject" >&2
        printf '%s\n' "$body" >&2
    fi
}

# Split check_status.sh output into status TSV and restart section.
# Sets STATUS_PART and RESTART_PART in the caller's scope.
split_output() {
    local clean="$1"
    STATUS_PART=""
    RESTART_PART=""
    local past_status=false
    while IFS= read -r line; do
        if [[ "$past_status" == true ]]; then
            RESTART_PART+="${line}"$'\n'
        elif [[ -z "$line" && -n "$STATUS_PART" ]]; then
            past_status=true
        else
            STATUS_PART+="${line}"$'\n'
        fi
    done <<<"$clean"
}

# Count statuses from TSV. Sets COMPLETED, ACTIVE, FAILED, ERROR, TOTAL,
# and ERROR_LINES in the caller's scope.
count_statuses() {
    COMPLETED=0 ACTIVE=0 FAILED=0 ERROR=0 TOTAL=0
    ERROR_LINES=""
    local directory status progress info simname
    while IFS=$'\t' read -r directory status progress info; do
        [[ "$status" == "status" || -z "$status" ]] && continue
        TOTAL=$((TOTAL + 1))
        case "$status" in
            completed) COMPLETED=$((COMPLETED + 1)) ;;
            active) ACTIVE=$((ACTIVE + 1)) ;;
            failed) FAILED=$((FAILED + 1)) ;;
            error)
                ERROR=$((ERROR + 1))
                simname="$(basename "$directory")"
                ERROR_LINES+="  ${simname}  ${progress}: ${info}"$'\n'
                ;;
            *) echo "Warning: unexpected status '${status}' for ${directory}" >&2 ;;
        esac
    done <<<"$1"
}

# Parse restart output from check_status.sh -R. Sets RESTART_COUNT and
# RESTART_LINES in the caller's scope.
parse_restarts() {
    RESTART_COUNT=0 RESTART_LINES=""
    local line
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        RESTART_LINES+="  ${line}"$'\n'
        [[ "$line" == "Submitted batch job"* ]] && RESTART_COUNT=$((RESTART_COUNT + 1))
    done <<<"$1"
}

# Build dry-run restart lines from failed entries.
dry_run_restarts() {
    RESTART_COUNT=0 RESTART_LINES=""
    local directory status progress info simname
    while IFS=$'\t' read -r directory status progress info; do
        [[ "$status" == "failed" ]] || continue
        simname="$(basename "$directory")"
        RESTART_LINES+="  ${simname}  ${progress} -> [DRY RUN] would restart"$'\n'
        RESTART_COUNT=$((RESTART_COUNT + 1))
    done <<<"$1"
}

echo "=== GROMACS Monitor ==="
echo "Timestamp: $(date)"
echo ""

# ---- Process each directory ----

GRAND_COMPLETED=0 GRAND_ACTIVE=0 GRAND_FAILED=0 GRAND_ERROR=0
GRAND_TOTAL=0 GRAND_RESTARTS=0 GRAND_SKIPPED=0
REPORT=""

for DIR in "${DIRS[@]}"; do
    if [[ ! -d "$DIR" ]]; then
        echo "  Warning: directory does not exist: $DIR" >&2
        GRAND_SKIPPED=$((GRAND_SKIPPED + 1))
        continue
    fi

    DIR_ABS="$(cd "$DIR" && pwd -P)"
    DIR_NAME="$(basename "$DIR_ABS")"

    # Run check_status.sh (with -R unless dry run).
    CS_FLAGS=(-r "$DIR_ABS")
    [[ -n "$TARGET_NS" ]] && CS_FLAGS+=(-t "$TARGET_NS")
    [[ "$DRY_RUN" == false ]] && CS_FLAGS+=(-R)

    if ! RAW="$(bash "$CHECK_STATUS" "${CS_FLAGS[@]}")"; then
        echo "  Warning: check_status.sh failed for ${DIR_NAME}" >&2
        GRAND_SKIPPED=$((GRAND_SKIPPED + 1))
        continue
    fi

    CLEAN="$(echo "$RAW" | strip_ansi)"
    split_output "$CLEAN"
    count_statuses "$STATUS_PART"

    if [[ "$DRY_RUN" == true && FAILED -gt 0 ]]; then
        dry_run_restarts "$STATUS_PART"
    else
        parse_restarts "$RESTART_PART"
    fi

    # Accumulate totals.
    GRAND_COMPLETED=$((GRAND_COMPLETED + COMPLETED))
    GRAND_ACTIVE=$((GRAND_ACTIVE + ACTIVE))
    GRAND_FAILED=$((GRAND_FAILED + FAILED))
    GRAND_ERROR=$((GRAND_ERROR + ERROR))
    GRAND_TOTAL=$((GRAND_TOTAL + TOTAL))
    GRAND_RESTARTS=$((GRAND_RESTARTS + RESTART_COUNT))

    # Build per-directory report.
    R="[${DIR_NAME}] ${COMPLETED}/${TOTAL} completed, ${ACTIVE} active, ${FAILED} failed, ${ERROR} error"$'\n'
    [[ -n "$RESTART_LINES" ]] && R+="${RESTART_LINES}"
    [[ -n "$ERROR_LINES" ]] && R+="${ERROR_LINES}"
    REPORT+="${R}"$'\n'
    echo "$R"
done

# ---- All done? ----

ALL_DONE=false
if ((GRAND_SKIPPED == 0 && GRAND_TOTAL > 0 && GRAND_COMPLETED == GRAND_TOTAL)); then
    ALL_DONE=true
fi

# ---- Email ----

SUMMARY="${GRAND_COMPLETED}/${GRAND_TOTAL} completed, ${GRAND_ACTIVE} active, ${GRAND_FAILED} failed, ${GRAND_ERROR} error"

if [[ "$ALL_DONE" == true ]]; then
    SUBJECT="[GROMACS Monitor] All ${GRAND_TOTAL} jobs completed"
    BODY="All jobs completed."$'\n\n'"${REPORT}"
else
    SUBJECT="[GROMACS Monitor] ${SUMMARY}"
    BODY="$(date)"$'\n\n'"${REPORT}"
fi

echo "Sending email to ${EMAIL}..."
if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would send email:"
    echo "Subject: ${SUBJECT}"
    echo "$BODY"
else
    send_email "$SUBJECT" "$BODY" "$EMAIL"
fi

# ---- Self-resubmit or exit ----

if [[ "$ALL_DONE" == true ]]; then
    echo "All jobs completed. No resubmission needed."
    exit 0
fi

ARGS=(-d "${DIRS[@]}" -e "$EMAIL" -i "$INTERVAL")
[[ -n "$TARGET_NS" ]] && ARGS+=(-t "$TARGET_NS")
[[ "$DRY_RUN" == true ]] && ARGS+=(-n)

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would resubmit: sbatch --chdir ${SCRIPTS_DIR} --begin=now+${INTERVAL}hour --dependency=singleton ${SCRIPTS_DIR}/monitor.sbatch ${ARGS[*]}"
else
    echo "Resubmitting for next check in ${INTERVAL} hour(s)..."
    sbatch --chdir "$SCRIPTS_DIR" --begin="now+${INTERVAL}hour" --dependency=singleton "${SCRIPTS_DIR}/monitor.sbatch" "${ARGS[@]}"
fi

echo "Done."
