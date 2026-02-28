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
SRC_BASE="${GROUP_SCRATCH}/plumed"
mkdir -p "${PLUMED_PREFIX}" "${SRC_BASE}"

PLUMED_TGZ_URL="https://github.com/plumed/plumed2/releases/download/v${PLUMED_VERSION}/plumed-src-${PLUMED_VERSION}.tgz"

NPROC="${1:-$(nproc)}"

# Download sources

cd "${SRC_BASE}"
if [[ ! -f "plumed-src-${PLUMED_VERSION}.tgz" ]]; then
    echo "Downloading PLUMED ${PLUMED_VERSION}"
    curl -fL --retry 5 --retry-delay 2 -o "plumed-src-${PLUMED_VERSION}.tgz" "${PLUMED_TGZ_URL}"
fi

# Build & install PLUMED

cd "${SRC_BASE}"
rm -rf "plumed-${PLUMED_VERSION}"
tar -xzvf "plumed-src-${PLUMED_VERSION}.tgz"

cd "plumed-${PLUMED_VERSION}"
./configure --prefix="${PLUMED_PREFIX}" \
    --enable-mpi \
    --enable-modules=all
make -j"${NPROC}"
make install

export PLUMED_KERNEL="${PLUMED_PREFIX}/lib/libplumedKernel.so"
export PATH="$PLUMED_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PLUMED_PREFIX/lib:$LD_LIBRARY_PATH"
plumed -h
