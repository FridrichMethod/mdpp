#!/bin/bash

set -euo pipefail

for compiler in g++ cmake make gcc nvcc; do
    if ! command -v $compiler >/dev/null 2>&1; then
        echo "$compiler could not be found"
        exit 1
    fi
done

GMX_VERSION="2025.2"
GMX_PREFIX="/apps/${GMX_VERSION}"
NPROC=$(nproc)

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
    -DGMX_MPI=OFF \
    -DGMX_THREAD_MPI=ON \
    -DREGRESSIONTEST_DOWNLOAD=ON \
    -DCMAKE_INSTALL_PREFIX="${GMX_PREFIX}"
make -j"${NPROC}"
make check -j"${NPROC}"
make install
