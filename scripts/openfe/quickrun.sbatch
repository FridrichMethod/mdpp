#!/bin/bash
#SBATCH --job-name=openfe
#SBATCH --output=logs/openfe_%j.out
#SBATCH --error=logs/openfe_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=gpu,owners
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=12G
#SBATCH --constraint="GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:A40|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S|GPU_SKU:V100_PCIE|GPU_SKU:V100S_PCIE|GPU_SKU:V100_SXM2"
#SBATCH --time=2-00:00:00

set -euo pipefail

OPENFE_CONTAINER=/scratch/groups/ayting/containers/openfe/openfe_latest-apptainer.sif

usage() {
    echo "Usage: sbatch [--array=0-N] $0 <transformation.json> -o|--output-dir <results_dir> [-n|--n-replicas <N>]" >&2
    exit 1
}

RESULTS_DIR=""
N_REPLICAS=1
FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir|--output_dir)
            [[ $# -lt 2 ]] && usage
            RESULTS_DIR=$2
            shift 2
            ;;
        -n|--n-replicas)
            [[ $# -lt 2 ]] && usage
            N_REPLICAS=$2
            shift 2
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

filename=$(basename "${FILE}" .json)
run_dir="${RESULTS_DIR}/${filename}/replica_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${run_dir}"

apptainer run --nv "${OPENFE_CONTAINER}" openfe quickrun \
    "${FILE}" -o "${run_dir}/${filename}.json" -d "${run_dir}"
