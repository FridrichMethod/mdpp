#!/bin/bash
#SBATCH --job-name=openfe_monitor
#SBATCH --partition=normal,owners
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --output=logs/monitor_%j.out
#SBATCH --error=logs/monitor_%j.err

set -euo pipefail

# SLURM copies batch scripts to a spool directory, so BASH_SOURCE won't point
# to the original location. Extract the original script path from the submit
# line recorded by SLURM.
_submit_line="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^[[:space:]]*SubmitLine=//p')"
SCRIPTS_DIR=""
for _word in $_submit_line; do
    if [[ "$_word" == *monitor.sbatch ]]; then
        SCRIPTS_DIR="$(cd "${SLURM_SUBMIT_DIR:-.}" && cd "$(dirname "$_word")" && pwd -P)"
        break
    fi
done

if [[ -z "$SCRIPTS_DIR" || ! -f "${SCRIPTS_DIR}/monitor.sh" ]]; then
    echo "Error: could not resolve scripts directory from submit line: $_submit_line" >&2
    exit 1
fi

exec bash "${SCRIPTS_DIR}/monitor.sh" "$@"
