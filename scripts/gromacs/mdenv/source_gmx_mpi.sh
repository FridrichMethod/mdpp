#!/bin/bash
# This script must be sourced, not executed.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: source this script: . ${0}" >&2
    exit 1
fi

ml gcc/12.4.0
ml cmake/3.31.4
ml make/4.4
ml cuda/12.6.1
ml openmpi/5.0.5
ml fftw/3.3.9
ml gsl/2.7
ml openblas/0.3.28
ml ucx/1.17.0
ml python/3.12.1

GMX_VERSION="2025.0"
GMX_PREFIX="${GROUP_HOME}/gromacs-${GMX_VERSION}"

source "${GMX_PREFIX}/bin/GMXRC"
