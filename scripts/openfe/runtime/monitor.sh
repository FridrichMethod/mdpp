#!/usr/bin/env bash

set -euo pipefail

# monitor.sh
#
# Monitor OpenFE quickrun jobs across project directories, restart failed jobs,
# and send email reports. Designed to self-resubmit via SLURM for periodic
# monitoring (default: hourly). Runs indefinitely until all jobs complete.
#
# Status handling:
#   completed : no action
#   active    : no action (job still running)
#   failed    : restart via check_status.sh -R
#   error     : report only (multiple matching jobs, needs manual attention)
#
# Options:
#   -d DIR         Project directory to monitor (repeatable, required)
#   -e EMAIL       Notification email (default: zhaoyangli@stanford.edu)
#   -i HOURS       Hours between checks (default: 1)
#   -s STATE_FILE  Iteration state file (default: ~/.openfe_monitor_state)
#   -n             Dry run: parse and report but do not restart or resubmit
#   -h             Show help

SCRIPTS_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)"
CHECK_STATUS="${SCRIPTS_DIR}/check_status.sh"

DIRS=()
EMAIL="zhaoyangli@stanford.edu"
INTERVAL=1
STATE_FILE="${HOME}/.openfe_monitor_state"
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: monitor.sh -d DIR [-d DIR ...] [OPTIONS]

Options:
    -d DIR         Project directory to monitor (repeatable, required)
    -e EMAIL       Notification email (default: zhaoyangli@stanford.edu)
    -i HOURS       Interval between checks in hours (default: 1)
    -s STATE_FILE  Iteration state file (default: ~/.openfe_monitor_state)
    -n             Dry run: report only, no restarts or resubmissions
    -h             Show this help
EOF
}

while getopts ":d:e:i:s:nh" opt; do
    case "$opt" in
        d) DIRS+=("$OPTARG") ;;
        e) EMAIL="$OPTARG" ;;
        i) INTERVAL="$OPTARG" ;;
        s) STATE_FILE="$OPTARG" ;;
        n) DRY_RUN=true ;;
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

if [[ ${#DIRS[@]} -eq 0 ]]; then
    echo "Error: at least one -d DIR is required" >&2
    usage
    exit 2
fi

# ---- Helper functions ----

strip_ansi() {
    sed 's/\x1b\[[0-9;]*m//g'
}

send_email() {
    local subject="$1"
    local body="$2"
    local recipient="$3"

    if command -v mail >/dev/null 2>&1; then
        echo "$body" | mail -s "$subject" "$recipient"
    elif command -v sendmail >/dev/null 2>&1; then
        {
            printf 'To: %s\n' "$recipient"
            printf 'Subject: %s\n' "$subject"
            printf 'Content-Type: text/plain; charset=UTF-8\n'
            printf '\n'
            printf '%s\n' "$body"
        } | sendmail "$recipient"
    else
        echo "WARNING: No mail command available. Email report:" >&2
        echo "Subject: $subject" >&2
        echo "$body" >&2
    fi
}

# ---- Iteration tracking ----

ITERATION=1
if [[ -f "$STATE_FILE" ]]; then
    ITERATION=$(($(cat "$STATE_FILE") + 1))
fi

echo "$ITERATION" >"$STATE_FILE"

NL=$'\n'

echo "=== OpenFE Monitor: Iteration ${ITERATION} ==="
echo "Timestamp: $(date)"
echo ""

# ---- Main loop over directories ----

TOTAL_RESTARTS=0
TOTAL_COMPLETED=0
TOTAL_ACTIVE=0
TOTAL_FAILED=0
TOTAL_ERROR=0
REPORT=""

for DIR in "${DIRS[@]}"; do
    if [[ ! -d "$DIR" ]]; then
        echo "  Warning: directory does not exist: $DIR" >&2
        REPORT+="${NL}--- $(basename "$DIR") ---${NL}  Directory not found: ${DIR}${NL}"
        continue
    fi

    DIR_ABS="$(cd "$DIR" && pwd -P)"
    DIR_NAME="$(basename "$DIR_ABS")"

    echo "--- Checking: ${DIR_NAME} ---"

    # Build check_status.sh flags.
    CS_FLAGS=(-r "$DIR_ABS")
    if [[ "$DRY_RUN" == false ]]; then
        CS_FLAGS+=(-R)
    fi

    # Run check_status.sh; capture full output (status TSV + restart messages).
    if ! FULL_OUTPUT="$(bash "$CHECK_STATUS" "${CS_FLAGS[@]}" 2>&1)"; then
        echo "  Warning: check_status.sh returned non-zero for ${DIR_NAME}" >&2
        REPORT+="${NL}--- ${DIR_NAME} ---${NL}  check_status.sh failed${NL}"
        continue
    fi

    # Strip ANSI codes for parsing.
    CLEAN_OUTPUT="$(echo "$FULL_OUTPUT" | strip_ansi)"

    # Split into status TSV (before first blank line after header) and restart section.
    STATUS_PART=""
    RESTART_PART=""
    past_status=false

    while IFS= read -r line; do
        if [[ "$past_status" == true ]]; then
            RESTART_PART+="${line}${NL}"
        elif [[ -z "$line" && -n "$STATUS_PART" ]]; then
            past_status=true
        else
            STATUS_PART+="${line}${NL}"
        fi
    done <<<"$CLEAN_OUTPUT"

    # Count statuses from TSV (skip header).
    COMPLETED=0
    ACTIVE=0
    FAILED=0
    ERROR=0
    TOTAL=0
    ERROR_INFO=""

    while IFS=$'\t' read -r directory status replica info; do
        [[ "$status" == "status" ]] && continue
        [[ -z "$status" ]] && continue

        TOTAL=$((TOTAL + 1))

        case "$status" in
            completed) COMPLETED=$((COMPLETED + 1)) ;;
            active) ACTIVE=$((ACTIVE + 1)) ;;
            failed) FAILED=$((FAILED + 1)) ;;
            error)
                ERROR=$((ERROR + 1))
                tname="$(basename "$(dirname "$directory")")"
                ERROR_INFO+="    ${tname}  ${replica}: ${info}${NL}"
                ;;
        esac
    done <<<"$STATUS_PART"

    TOTAL_COMPLETED=$((TOTAL_COMPLETED + COMPLETED))
    TOTAL_ACTIVE=$((TOTAL_ACTIVE + ACTIVE))
    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
    TOTAL_ERROR=$((TOTAL_ERROR + ERROR))

    # Parse restart info from check_status.sh -R output.
    RESTART_COUNT=0
    RESTART_INFO=""

    if [[ "$DRY_RUN" == true && FAILED -gt 0 ]]; then
        # Dry run: show what would be restarted from the failed entries.
        while IFS=$'\t' read -r directory status replica info; do
            [[ "$status" == "failed" ]] || continue
            tname="$(basename "$(dirname "$directory")")"
            RESTART_INFO+="    ${tname}  ${replica} -> [DRY RUN] would restart${NL}"
            RESTART_COUNT=$((RESTART_COUNT + 1))
        done <<<"$STATUS_PART"
    elif [[ -n "$RESTART_PART" ]]; then
        # Parse actual restart output from check_status.sh -R.
        local_restart=""
        while IFS= read -r line; do
            if [[ "$line" == Resubmitting* ]]; then
                local_restart+="    ${line}${NL}"
            elif [[ "$line" == "Submitted batch job"* ]]; then
                RESTART_COUNT=$((RESTART_COUNT + 1))
                local_restart+="    ${line}${NL}"
            fi
        done <<<"$RESTART_PART"
        RESTART_INFO="$local_restart"
    fi

    TOTAL_RESTARTS=$((TOTAL_RESTARTS + RESTART_COUNT))

    # Build directory report section.
    DIR_REPORT="--- ${DIR_NAME} ---${NL}"
    DIR_REPORT+="  Completed: ${COMPLETED}/${TOTAL}  Active: ${ACTIVE}  Failed: ${FAILED}"
    if ((RESTART_COUNT > 0)); then
        DIR_REPORT+=" (restarted)"
    fi
    DIR_REPORT+="  Error: ${ERROR}${NL}"

    if [[ -n "$RESTART_INFO" ]]; then
        DIR_REPORT+="${NL}  Restarts:${NL}${RESTART_INFO}"
    fi

    if ((ERROR > 0)); then
        DIR_REPORT+="${NL}  Errors (not restarted):${NL}${ERROR_INFO}"
    fi

    REPORT+="${NL}${DIR_REPORT}"
    echo "$DIR_REPORT"
done

# ---- Check if all done ----

ALL_DONE=false
if ((TOTAL_ACTIVE == 0 && TOTAL_FAILED == 0 && TOTAL_ERROR == 0 && TOTAL_COMPLETED > 0)); then
    ALL_DONE=true
fi

# ---- Build and send email ----

if [[ "$ALL_DONE" == true ]]; then
    SUBJECT="[OpenFE Monitor] All jobs completed!"
    BODY="All jobs across ${#DIRS[@]} directories have completed.${NL}${NL}"
    BODY+="Final status (iteration ${ITERATION}):${NL}"
    BODY+="${REPORT}"
else
    SUBJECT="[OpenFE Monitor] Iteration ${ITERATION}"
    if ((TOTAL_RESTARTS > 0)); then
        SUBJECT+=" - ${TOTAL_RESTARTS} restart(s)"
    fi
    BODY="Monitoring report for iteration ${ITERATION}${NL}"
    BODY+="Timestamp: $(date)${NL}"
    BODY+="${REPORT}"
fi

echo ""
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

# Reconstruct original arguments for resubmission.
MONITOR_SBATCH="${SCRIPTS_DIR}/monitor.sbatch"
ARGS=()
for d in "${DIRS[@]}"; do
    ARGS+=(-d "$d")
done
ARGS+=(-e "$EMAIL" -i "$INTERVAL" -s "$STATE_FILE")
if [[ "$DRY_RUN" == true ]]; then
    ARGS+=(-n)
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would resubmit: sbatch --begin=now+${INTERVAL}hour --dependency=singleton ${MONITOR_SBATCH} ${ARGS[*]}"
else
    echo "Resubmitting for next check in ${INTERVAL} hour(s)..."
    sbatch --begin="now+${INTERVAL}hour" --dependency=singleton "$MONITOR_SBATCH" "${ARGS[@]}"
fi

echo "Done."
