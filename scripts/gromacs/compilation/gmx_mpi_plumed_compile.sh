#!/usr/bin/env bash

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
GMX_VERSION="2025.0"
GMX_PREFIX="${GROUP_HOME}/gromacs-${GMX_VERSION}-plumed-${PLUMED_VERSION}"
GMX_TGZ_URL="https://ftp.gromacs.org/gromacs/gromacs-${GMX_VERSION}.tar.gz"
NPROC="${1:-$(nproc)}"

if [[ ! -f "gromacs-${GMX_VERSION}.tar.gz" ]]; then
    echo "Downloading GROMACS ${GMX_VERSION}"
    curl -fL --retry 5 --retry-delay 2 -o "gromacs-${GMX_VERSION}.tar.gz" "${GMX_TGZ_URL}"
fi

tar -xzvf "gromacs-${GMX_VERSION}.tar.gz"
cd "gromacs-${GMX_VERSION}"

# Patch GROMACS with PLUMED (here 4 is gromacs-2025.0)
plumed patch -p --runtime <<EOF
4
EOF

mkdir -p build
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
