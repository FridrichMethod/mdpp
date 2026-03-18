#!/bin/bash

set -euo pipefail

for compiler in cmake make gcc g++ nvcc; do
    if ! command -v $compiler >/dev/null 2>&1; then
        echo "$compiler could not be found"
        exit 1
    fi
done

GMX_VERSION="2026.0"
GMX_PREFIX="/usr/local/gromacs-${GMX_VERSION}"
GMX_TGZ_URL="https://ftp.gromacs.org/gromacs/gromacs-${GMX_VERSION}.tar.gz"
NPROC="${1:-$(nproc)}"

if [[ ! -f "gromacs-${GMX_VERSION}.tar.gz" ]]; then
    echo "Downloading GROMACS ${GMX_VERSION}"
    curl -fL --retry 5 --retry-delay 2 -o "gromacs-${GMX_VERSION}.tar.gz" "${GMX_TGZ_URL}"
fi

tar -xzvf "gromacs-${GMX_VERSION}.tar.gz"
cd "gromacs-${GMX_VERSION}"
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
