#!/bin/bash

set -euo pipefail

ml gcc/14.2.0
ml cmake/3.31.4
ml make/4.4
ml cuda/12.8.0

GMX_VERSION="2026.0"
GMX_PREFIX="/home/groups/ayting/gromacs-${GMX_VERSION}"

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
make -j16
make check
make install
