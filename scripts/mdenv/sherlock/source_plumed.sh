#!/bin/bash

set -euo pipefail

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

PLUMED_VERSION="2.10.0"
PLUMED_PREFIX="${GROUP_HOME}/plumed-${PLUMED_VERSION}"

export PLUMED_KERNEL="${PLUMED_PREFIX}/lib/libplumedKernel.so"
export PATH="$PLUMED_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PLUMED_PREFIX/lib:$LD_LIBRARY_PATH"
