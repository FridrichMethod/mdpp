#!/usr/bin/env bash

set -euo pipefail

# monitor.sh
#
# Monitor OpenFE quickrun jobs, restart failures, send email reports.
# Self-resubmits via SLURM until all jobs complete.
#
# Options:
#   -d DIR [DIR ...]  Project directories to monitor (required)
#   -e EMAIL          Notification email (default: zhaoyangli@stanford.edu)
#   -i HOURS          Hours between checks (default: 1)
#   -s STATE_FILE     Iteration state file (default: ~/.openfe_monitor_state)
#   -n                Dry run: report only, no restarts or resubmissions
#   -h                Show help

SCRIPTS_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)"
CHECK_STATUS="${SCRIPTS_DIR}/check_status.sh"

# ---- Parse arguments ----

DIRS=()
EMAIL="zhaoyangli@stanford.edu"
INTERVAL=1
STATE_FILE="${HOME}/.openfe_monitor_state"
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: monitor.sh -d DIR [DIR ...] [OPTIONS]

Options:
    -d DIR [DIR ...]  Project directories to monitor (required)
    -e EMAIL          Notification email (default: zhaoyangli@stanford.edu)
    -i HOURS          Interval between checks in hours (default: 1)
    -s STATE_FILE     Iteration state file (default: ~/.openfe_monitor_state)
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
            shift 2
            ;;
        -s)
            [[ $# -lt 2 ]] && {
                echo "Error: -s requires an argument" >&2
                exit 2
            }
            STATE_FILE="$2"
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

# ---- Helpers ----

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }

send_email() {
    local subject="$1" body="$2" recipient="$3"
    if command -v mail >/dev/null 2>&1; then
        echo "$body" | mail -s "$subject" "$recipient"
    elif command -v sendmail >/dev/null 2>&1; then
        printf 'To: %s\nSubject: %s\nContent-Type: text/plain; charset=UTF-8\n\n%s\n' \
            "$recipient" "$subject" "$body" | sendmail "$recipient"
    else
        echo "WARNING: No mail command available. Email report:" >&2
        echo "Subject: $subject" >&2
        echo "$body" >&2
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
# and ERROR_LINES / FAILED_LINES in the caller's scope.
count_statuses() {
    COMPLETED=0 ACTIVE=0 FAILED=0 ERROR=0 TOTAL=0
    FAILED_LINES="" ERROR_LINES=""
    local directory status replica info tname
    while IFS=$'\t' read -r directory status replica info; do
        [[ "$status" == "status" || -z "$status" ]] && continue
        TOTAL=$((TOTAL + 1))
        case "$status" in
            completed) COMPLETED=$((COMPLETED + 1)) ;;
            active) ACTIVE=$((ACTIVE + 1)) ;;
            failed)
                FAILED=$((FAILED + 1))
                tname="$(basename "$(dirname "$directory")")"
                FAILED_LINES+="  ${tname}  ${replica}: ${info}"$'\n'
                ;;
            error)
                ERROR=$((ERROR + 1))
                tname="$(basename "$(dirname "$directory")")"
                ERROR_LINES+="  ${tname}  ${replica}: ${info}"$'\n'
                ;;
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
    local directory status replica info tname
    while IFS=$'\t' read -r directory status replica info; do
        [[ "$status" == "failed" ]] || continue
        tname="$(basename "$(dirname "$directory")")"
        RESTART_LINES+="  ${tname}  ${replica} -> [DRY RUN] would restart"$'\n'
        RESTART_COUNT=$((RESTART_COUNT + 1))
    done <<<"$1"
}

# ---- Iteration tracking ----

ITERATION=1
if [[ -f "$STATE_FILE" ]]; then
    ITERATION=$(($(cat "$STATE_FILE") + 1))
fi
echo "$ITERATION" >"$STATE_FILE"

echo "=== OpenFE Monitor: Iteration ${ITERATION} ==="
echo "Timestamp: $(date)"
echo ""

# ---- Process each directory ----

GRAND_COMPLETED=0 GRAND_ACTIVE=0 GRAND_FAILED=0 GRAND_ERROR=0
GRAND_TOTAL=0 GRAND_RESTARTS=0
REPORT=""

for DIR in "${DIRS[@]}"; do
    if [[ ! -d "$DIR" ]]; then
        echo "  Warning: directory does not exist: $DIR" >&2
        continue
    fi

    DIR_ABS="$(cd "$DIR" && pwd -P)"
    DIR_NAME="$(basename "$DIR_ABS")"

    # Run check_status.sh (with -R unless dry run).
    CS_FLAGS=(-r "$DIR_ABS")
    [[ "$DRY_RUN" == false ]] && CS_FLAGS+=(-R)

    if ! RAW="$(bash "$CHECK_STATUS" "${CS_FLAGS[@]}")"; then
        echo "  Warning: check_status.sh failed for ${DIR_NAME}" >&2
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
if ((GRAND_ACTIVE == 0 && GRAND_FAILED == 0 && GRAND_ERROR == 0 && GRAND_COMPLETED > 0)); then
    ALL_DONE=true
fi

# ---- Email ----

SUMMARY="${GRAND_COMPLETED}/${GRAND_TOTAL} completed, ${GRAND_ACTIVE} active, ${GRAND_FAILED} failed, ${GRAND_ERROR} error"

if [[ "$ALL_DONE" == true ]]; then
    SUBJECT="[OpenFE Monitor] All ${GRAND_TOTAL} jobs completed"
    BODY="All jobs completed (iteration ${ITERATION})."$'\n\n'"${REPORT}"
else
    SUBJECT="[OpenFE Monitor] #${ITERATION}: ${SUMMARY}"
    BODY="Iteration ${ITERATION} -- $(date)"$'\n\n'"${REPORT}"
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
    rm -f "$STATE_FILE"
    exit 0
fi

MONITOR_SBATCH="${SCRIPTS_DIR}/monitor.sbatch"
ARGS=(-d "${DIRS[@]}" -e "$EMAIL" -i "$INTERVAL" -s "$STATE_FILE")
[[ "$DRY_RUN" == true ]] && ARGS+=(-n)

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would resubmit: sbatch --begin=now+${INTERVAL}hour --dependency=singleton ${MONITOR_SBATCH} ${ARGS[*]}"
else
    echo "Resubmitting for next check in ${INTERVAL} hour(s)..."
    sbatch --begin="now+${INTERVAL}hour" --dependency=singleton "$MONITOR_SBATCH" "${ARGS[@]}"
fi

echo "Done."
