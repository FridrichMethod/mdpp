#!/usr/bin/env bash

set -euo pipefail

# monitor.sh
#
# Monitor OpenFE quickrun jobs across project directories, restart failed jobs,
# and send email reports. Designed to self-resubmit via SLURM for periodic
# monitoring (default: hourly for 48 iterations / 2 days).
#
# Status handling:
#   completed : no action
#   active    : no action (job still running)
#   failed    : restart via sbatch
#   error     : report only (multiple matching jobs, needs manual attention)
#
# Options:
#   -d DIR         Project directory to monitor (repeatable, required)
#   -e EMAIL       Notification email (default: zhaoyangli@stanford.edu)
#   -m MAX_ITER    Maximum iterations before stopping (default: 48)
#   -i HOURS       Hours between checks (default: 1)
#   -s STATE_FILE  Iteration state file (default: ~/.openfe_monitor_state)
#   -n             Dry run: parse and report but do not restart or resubmit
#   -h             Show help

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CHECK_STATUS="${SCRIPTS_DIR}/check_status.sh"

DIRS=()
EMAIL="zhaoyangli@stanford.edu"
MAX_ITER=48
INTERVAL=1
STATE_FILE="${HOME}/.openfe_monitor_state"
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: monitor.sh -d DIR [-d DIR ...] [OPTIONS]

Options:
    -d DIR         Project directory to monitor (repeatable, required)
    -e EMAIL       Notification email (default: zhaoyangli@stanford.edu)
    -m MAX_ITER    Maximum iterations (default: 48)
    -i HOURS       Interval between checks in hours (default: 1)
    -s STATE_FILE  Iteration state file (default: ~/.openfe_monitor_state)
    -n             Dry run: report only, no restarts or resubmissions
    -h             Show this help
EOF
}

while getopts ":d:e:m:i:s:nh" opt; do
    case "$opt" in
        d) DIRS+=("$OPTARG") ;;
        e) EMAIL="$OPTARG" ;;
        m) MAX_ITER="$OPTARG" ;;
        i) INTERVAL="$OPTARG" ;;
        s) STATE_FILE="$OPTARG" ;;
        n) DRY_RUN=true ;;
        h) usage; exit 0 ;;
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
    ITERATION=$(( $(cat "$STATE_FILE") + 1 ))
fi

if (( ITERATION > MAX_ITER )); then
    echo "Maximum iterations ($MAX_ITER) reached. Exiting."
    rm -f "$STATE_FILE"
    exit 0
fi

echo "$ITERATION" > "$STATE_FILE"

NL=$'\n'

echo "=== OpenFE Monitor: Iteration ${ITERATION}/${MAX_ITER} ==="
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

    # Run check_status.sh; stdout has TSV data, stderr goes to monitor log.
    if ! STATUS_OUTPUT="$(bash "$CHECK_STATUS" -r "$DIR_ABS")"; then
        echo "  Warning: check_status.sh returned non-zero for ${DIR_NAME}" >&2
        REPORT+="${NL}--- ${DIR_NAME} ---${NL}  check_status.sh failed${NL}"
        continue
    fi

    # Strip ANSI codes for parsing.
    CLEAN_OUTPUT="$(echo "$STATUS_OUTPUT" | strip_ansi)"

    # Count statuses (skip header line).
    COMPLETED=0
    ACTIVE=0
    FAILED=0
    ERROR=0
    TOTAL=0

    FAILED_ENTRIES=()
    ERROR_ENTRIES=()

    while IFS=$'\t' read -r directory status replica info; do
        # Skip header and blank lines.
        [[ "$status" == "status" ]] && continue
        [[ -z "$status" ]] && continue

        TOTAL=$((TOTAL + 1))

        case "$status" in
            completed)
                COMPLETED=$((COMPLETED + 1))
                ;;
            active)
                ACTIVE=$((ACTIVE + 1))
                ;;
            failed)
                FAILED=$((FAILED + 1))
                # Extract tname from directory: .../results/<tname>/replica_<N>
                tname="$(basename "$(dirname "$directory")")"
                # Extract replica_id from replica column: replica_<N>
                replica_id="${replica#replica_}"
                FAILED_ENTRIES+=("${tname}|${replica_id}|${info}")
                ;;
            error)
                ERROR=$((ERROR + 1))
                tname="$(basename "$(dirname "$directory")")"
                replica_id="${replica#replica_}"
                ERROR_ENTRIES+=("${tname}|${replica_id}|${info}")
                ;;
        esac
    done <<< "$CLEAN_OUTPUT"

    TOTAL_COMPLETED=$((TOTAL_COMPLETED + COMPLETED))
    TOTAL_ACTIVE=$((TOTAL_ACTIVE + ACTIVE))
    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
    TOTAL_ERROR=$((TOTAL_ERROR + ERROR))

    # Restart failed jobs.
    DIR_RESTARTS=""
    RESTART_COUNT=0

    if (( FAILED > 0 )); then
        for entry in "${FAILED_ENTRIES[@]}"; do
            IFS='|' read -r tname replica_id info <<< "$entry"

            tfile="${DIR_ABS}/transformations/${tname}.json"
            if [[ ! -f "$tfile" ]]; then
                DIR_RESTARTS+="    ${tname}  replica_${replica_id}: transformation file not found${NL}"
                continue
            fi

            if [[ "$DRY_RUN" == true ]]; then
                DIR_RESTARTS+="    ${tname}  replica_${replica_id} -> [DRY RUN] would restart${NL}"
                RESTART_COUNT=$((RESTART_COUNT + 1))
            else
                # Submit restart from the project directory where quickrun.sbatch is symlinked.
                RESTART_OUTPUT="$(cd "$DIR_ABS" && sbatch --array="$replica_id" quickrun.sbatch "transformations/${tname}.json" -o results/ 2>&1)" || true
                JOB_ID="$(echo "$RESTART_OUTPUT" | grep -oP 'Submitted batch job \K[0-9]+' || echo "unknown")"
                DIR_RESTARTS+="    ${tname}  replica_${replica_id} -> Job ${JOB_ID}${NL}"
                RESTART_COUNT=$((RESTART_COUNT + 1))
            fi
        done
    fi

    TOTAL_RESTARTS=$((TOTAL_RESTARTS + RESTART_COUNT))

    # Build directory report section.
    DIR_REPORT="--- ${DIR_NAME} ---${NL}"
    DIR_REPORT+="  Completed: ${COMPLETED}/${TOTAL}  Active: ${ACTIVE}  Failed: ${FAILED}"
    if (( RESTART_COUNT > 0 )); then
        DIR_REPORT+=" (restarted)"
    fi
    DIR_REPORT+="  Error: ${ERROR}${NL}"

    if [[ -n "$DIR_RESTARTS" ]]; then
        DIR_REPORT+="${NL}  Restarts:${NL}${DIR_RESTARTS}"
    fi

    if (( ERROR > 0 )); then
        DIR_REPORT+="${NL}  Errors (not restarted):${NL}"
        for entry in "${ERROR_ENTRIES[@]}"; do
            IFS='|' read -r tname replica_id info <<< "$entry"
            DIR_REPORT+="    ${tname}  replica_${replica_id}: ${info}${NL}"
        done
    fi

    REPORT+="${NL}${DIR_REPORT}"
    echo "$DIR_REPORT"
done

# ---- Check if all done ----

ALL_DONE=false
if (( TOTAL_ACTIVE == 0 && TOTAL_FAILED == 0 && TOTAL_ERROR == 0 && TOTAL_COMPLETED > 0 )); then
    ALL_DONE=true
fi

# ---- Build and send email ----

if [[ "$ALL_DONE" == true ]]; then
    SUBJECT="[OpenFE Monitor] All jobs completed!"
    BODY="All jobs across ${#DIRS[@]} directories have completed.${NL}${NL}"
    BODY+="Final status (iteration ${ITERATION}/${MAX_ITER}):${NL}"
    BODY+="${REPORT}"
else
    SUBJECT="[OpenFE Monitor] Iteration ${ITERATION}/${MAX_ITER}"
    if (( TOTAL_RESTARTS > 0 )); then
        SUBJECT+=" - ${TOTAL_RESTARTS} restart(s)"
    fi
    BODY="Monitoring report for iteration ${ITERATION}/${MAX_ITER}${NL}"
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

if (( ITERATION >= MAX_ITER )); then
    echo "Maximum iterations (${MAX_ITER}) reached. No resubmission."
    rm -f "$STATE_FILE"
    exit 0
fi

# Reconstruct original arguments for resubmission.
MONITOR_SBATCH="${SCRIPTS_DIR}/monitor.sbatch"
ARGS=()
for d in "${DIRS[@]}"; do
    ARGS+=(-d "$d")
done
ARGS+=(-e "$EMAIL" -m "$MAX_ITER" -i "$INTERVAL" -s "$STATE_FILE")
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
