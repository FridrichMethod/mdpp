#!/bin/bash
#SBATCH --job-name=openfe
#SBATCH --output=logs/openfe_%j.out
#SBATCH --error=logs/openfe_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=btrippe,stat,bioe,possu,gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

mkdir -p logs

OPENFE_CONTAINER=/scratch/groups/ayting/containers/openfe/openfe_1.8.1-apptainer.sif

usage() {
    echo "Usage: sbatch $0 <transformation.json...> -o|--output-dir <results_dir>" >&2
    exit 1
}

RESULTS_DIR=""
FILES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir|--output_dir)
            [[ $# -lt 2 ]] && usage
            RESULTS_DIR=$2
            shift 2
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

[[ -z "${RESULTS_DIR}" || ${#FILES[@]} -eq 0 ]] && usage

mkdir -p "${RESULTS_DIR}"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID:-$$}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID:-$$}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    nvidia-cuda-mps-control -d
    trap 'echo quit | nvidia-cuda-mps-control' EXIT
fi

for file in "${FILES[@]}"; do
    if [[ ! -f "${file}" ]]; then
        echo "Missing transformation: ${file}" >&2
        exit 1
    fi
    if [[ "${file}" != *.json ]]; then
        echo "Not a json file: ${file}" >&2
        exit 1
    fi
    filename=$(basename "${file}" .json)
    mkdir -p "${RESULTS_DIR}/${filename}"
    apptainer run --nv "${OPENFE_CONTAINER}" openfe quickrun \
        "${file}" -o "${RESULTS_DIR}/${filename}/${filename}.json" -d "${RESULTS_DIR}/${filename}" &
done

wait
