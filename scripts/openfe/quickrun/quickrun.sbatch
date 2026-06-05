#!/usr/bin/env bash
#SBATCH --job-name=openfe
#SBATCH --output=logs/openfe_%A_%a.out
#SBATCH --error=logs/openfe_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${USER}@stanford.edu
#SBATCH --partition=gpu,owners
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --gpu_cmode=shared
#SBATCH --mem=12G
#SBATCH --constraint="GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:A40|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S|GPU_SKU:V100_PCIE|GPU_SKU:V100S_PCIE|GPU_SKU:V100_SXM2"
#SBATCH --time=2-00:00:00

# Here we enable default mode so that multiple contexts are allowed per device
# See https://www.sherlock.stanford.edu/docs/user-guide/gpu/#gpu-compute-modes

set -euo pipefail

# Override OPENFE_CONTAINER (apptainer .sif) via the environment for non-Sherlock
# or non-ayting-group deployments; the default below is the Sherlock ayting path.
OPENFE_CONTAINER="${OPENFE_CONTAINER:-/scratch/groups/ayting/containers/openfe/openfe_latest-apptainer.sif}"

usage() {
    cat >&2 <<EOF
Usage: sbatch [--array=0-N] $0 <transformation.json> -o <results_dir> [-h]

Arguments:
    <transformation.json>   Path to the OpenFE transformation JSON file
    -o, --output-dir DIR    Directory for output results (required)
    -h                      Show this help
EOF
    exit 1
}

RESULTS_DIR=""
FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir|--output_dir)
            [[ $# -lt 2 ]] && usage
            RESULTS_DIR=$2
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            [[ -n "${FILE}" ]] && usage
            FILE=$1
            shift
            ;;
    esac
done

[[ -z "${RESULTS_DIR}" || -z "${FILE}" ]] && usage

if [[ ! -f "${FILE}" ]]; then
    echo "Missing transformation: ${FILE}" >&2
    exit 1
fi
if [[ "${FILE}" != *.json ]]; then
    echo "Not a json file: ${FILE}" >&2
    exit 1
fi

# NOTE: We deliberately do NOT start a per-job CUDA MPS control daemon here.
# MPS's control daemon is effectively a per-node/per-user singleton; when two
# array tasks land on the same node, the competing per-job daemons collide and
# OpenMM reports "No compatible CUDA device is available" -- empirically every
# job that shares a node with another concurrent task fails, while a job alone
# on its node succeeds.  MPS is also unnecessary here: --gpu_cmode=shared puts
# the GPU in Default compute mode, which already lets a single openfe process
# hold the multiple CUDA contexts openmmtools ContextCache needs.  (MPS would
# only be required to share an Exclusive_Process GPU across multiple processes,
# which is not our case -- each job is a single process with its own GPU.)

filename=$(basename "${FILE}" .json)
run_dir="${RESULTS_DIR}/${filename}/replica_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${run_dir}"

# Remove stale result JSON so openfe can write to -o.
# Checkpoint data in -d is preserved for --resume.
rm -f "${run_dir}/${filename}.json"

apptainer run --nv "${OPENFE_CONTAINER}" openfe quickrun \
    --resume "${FILE}" -o "${run_dir}/${filename}.json" -d "${run_dir}"
