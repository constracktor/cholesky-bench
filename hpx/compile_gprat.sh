#!/bin/bash
# $1: python/cpp
# $2: cpu/gpu
# $3: mkl
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.

################################################################################
# Configurations
################################################################################
if command -v spack &> /dev/null; then
    echo "Spack command found, checking for environments..."

    # Get current hostname
    HOSTNAME=$(hostname -s)

    if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
	# Check if the gprat_cpu_gcc environment exists
    	if spack env list | grep -q "gprat_cpu_gcc"; then
	   echo "Found gprat_cpu_gcc environment, activating it."
	    module load gcc/14.2.0
	    export CXX=g++
	    export CC=gcc
	    spack env activate gprat_cpu_gcc
	    GPRAT_WITH_CUDA=OFF # whether GPRAT_WITH_CUDA is ON of OFF is irrelevant for this example
	fi
    elif [[ "$HOSTNAME" == "sven0"  ||  "$HOSTNAME" == "sven1" ]]; then
	#module load gcc/13.2.1
	spack load openblas arch=linux-fedora38-riscv64
	HPX_CMAKE=$HOME/git_workspace/build-scripts/build/hpx/lib64/cmake/HPX
	USE_MKL=OFF
    elif [[ $(uname -i) == "aarch64" ]]; then
	spack load gcc@14.2.0
	# Check if the gprat_cpu_arm environment exists
	if spack env list | grep -q "gprat_cpu_arm"; then
	    echo "Found gprat_cpu_arm environment, activating it."
	    spack env activate gprat_cpu_arm
	fi
	USE_MKL=OFF
    elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then
	# Check if the gprat_gpu_clang environment exists
	if spack env list | grep -q "gprat_gpu_clang"; then
	    echo "Found gprat_gpu_clang environment, activating it."
	    module load clang/17.0.1
	    export CXX=clang++
	    export CC=clang
	    module load cuda/12.0.1
	    spack env activate gprat_gpu_clang
	    GPRAT_WITH_CUDA=ON
	fi
    elif [[ "$HOSTNAME" == "nasrin0" || "$HOSTNAME" == "nasrin1" ]]; then
	#module load gcc
	spack load hpx@1.11.0%gcc@14.2.0 arch=linux-almalinux9-zen3
	spack load openblas%gcc@14.2.0 arch=linux-almalinux9-zen3 threads=none
    else
    	echo "Hostname is $HOSTNAME — no action taken."
    fi
else
    echo "Spack command not found. Building example without Spack."
    # Assuming that Spack is not required on given system
fi
rm -rf build && mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
	-DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
	-DENABLE_FORMAT_TARGETS=OFF \
        -DENABLE_MKL=OFF ..
make -j
################################################################################
# Compile code
################################################################################
