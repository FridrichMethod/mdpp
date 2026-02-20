#!/bin/bash

set -euo pipefail

for compiler in cmake make gcc g++ nvcc; do
    if ! command -v $compiler >/dev/null 2>&1; then
        echo "$compiler could not be found"
        exit 1
    fi
done

GMX_VERSION="2026.0"
GMX_PREFIX="/apps/gromacs-${GMX_VERSION}"
NPROC="${1:-$(nproc)}"

tar -xzvf gromacs-${GMX_VERSION}.tar.gz
cd gromacs-${GMX_VERSION}
mkdir build
cd build
cmake .. \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DGMX_GPU=CUDA \
    -DGMX_FFT_LIBRARY=fftw3 \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DGMX_MPI=ON \
    -DGMX_THREAD_MPI=OFF \
    -DREGRESSIONTEST_DOWNLOAD=ON \
    -DCMAKE_INSTALL_PREFIX="${GMX_PREFIX}"
make -j"${NPROC}"
make check -j"${NPROC}"
make install
