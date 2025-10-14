#!/bin/bash

ml gcc/12.4.0
ml cmake/3.31.4
ml make/4.4
ml cuda/12.6.1


tar -xzvf gromacs-2025.2.tar.gz
cd gromacs-2025.2 || exit
mkdir build
cd build || exit
cmake .. \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DGMX_GPU=CUDA \
    -DGMX_FFT_LIBRARY=fftw3 \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DGMX_MPI=OFF \
    -DGMX_THREAD_MPI=ON \
    -DREGRESSIONTEST_DOWNLOAD=ON \
    -DCMAKE_INSTALL_PREFIX="/home/groups/ayting/gromacs-2025.2"
make -j16
make check
make install
