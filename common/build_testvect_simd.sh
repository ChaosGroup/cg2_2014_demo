#!/bin/bash

CXX=${CXX:-clang++}
TARGET=testvect_simd
SOURCE=(
	testvect_simd.cpp
)
LFLAGS=(
	-lpthread
)

if [[ ${MACHTYPE} =~ "-apple-darwin" ]]; then
	NUM_LOGICAL_CORES=`sysctl hw.ncpu | sed s/^[^[:digit:]]*//`
	SOURCE+=(
		pthread_barrier.cpp
	)
	CXXFLAGS+=(
		-DCACHELINE_SIZE=`sysctl hw.cachelinesize | sed 's/^hw.cachelinesize: //g'`
		-march=armv8.4-a
		-mtune=native
	)

elif [[ ${MACHTYPE} =~ "-linux-" ]]; then
	NUM_LOGICAL_CORES=`lscpu | grep ^"CPU(s)" | sed s/^[^[:digit:]]*//`
	LFLAGS+=(
		-lrt
	)
else
	echo Unknown platform
	exit 255
fi

source cxx_util.sh

CXXFLAGS+=(
	-pipe
	-fno-exceptions
	-fno-rtti
	-fstrict-aliasing
# Use reciprocals instead of division
	-DVECT_SIMD_DIV_AS_RCP
# Use reciprocal sqrt instead if sqrt and division
#	-DVECT_SIMD_SQRT_DIV_AS_RSQRT
# Perform arithmetic conformace tests as well
	-DSIMD_TEST_CONFORMANCE
# Use as many worker threads, including the main thread
	-DSIMD_NUM_THREADS=$NUM_LOGICAL_CORES
# Enforce worker thread affinity; value represents affinity stride (for control over physical/logical CPU distribution)
#	-DSIMD_THREAD_AFFINITY=1
# Minimal alignment of any SIMD type (might be overridden in the architecture-dependant sections below)
	-DSIMD_ALIGNMENT=16
# For the performance test, use matx4 from etal namespace instead of counterpart from simd/scal namespace
#	-DSIMD_ETALON
# Alternative etalon v0 (in combination with SIMD_ETALON above); clang++-3.6 recommended
#	-DSIMD_ETALON_ALT0
# Alternative etalon v1 (in combination with SIMD_ETALON above); clang++-3.6 recommended
#	-DSIMD_ETALON_ALT1
# Alternative etalon v2 (in combination with SIMD_ETALON above); clang++-3.6 recommended
#	-DSIMD_ETALON_ALT2
# Alternative etalon v3 (in combination with SIMD_ETALON above); SSE4.1 only
#	-DSIMD_ETALON_ALT3
# Alternative etalon v4 (in combination with SIMD_ETALON above); ARMv7 only
#	-DSIMD_ETALON_ALT4
# Alternative etalon v5 (in combination with SIMD_ETALON above); Intel MIC only
#	-DSIMD_ETALON_ALT5
# Increase iteration workload by factor of 4; kills ALT1 performance on Sandy Bridge, unless DUBIOUS_AVX_OPTIMISATION is enabled
#	-DSIMD_WORKLOAD_ITERATION_DENSITY_X4
# Use read-port depressurising in ALT1 on AVX; only situation this has been known to help is with dense iteration workloads on Sandy Bridge
#	-DDUBIOUS_AVX_OPTIMISATION 
# Use manual unrolling of innermost loops
#	-DSIMD_MANUAL_UNROLL
# Use vector templates from scal namespace in both conformance and performance tests
#	-DSIMD_SCALAR
# Use vector templates from platform-agnostic rend namespace in performance tests
#	-DSIMD_AGNOSTIC
# Produce asm listings instead of binaries
#	-S
)
CXX_FILENAME=${CXX##*/}
if [[ ${CXX_FILENAME:0:3} == "g++" ]]; then
	CXXFLAGS+=(
		-ffast-math
	)

elif [[ ${CXX_FILENAME:0:7} == "clang++" ]]; then
	CXXFLAGS+=(
		-ffp-contract=fast
	)

elif [[ ${CXX_FILENAME:0:4} == "icpc" ]]; then
	CXXFLAGS+=(
		-fp-model fast=2
		-opt-prefetch=0
		-opt-streaming-cache-evict=0
	)
fi

if [[ ${HOSTTYPE:0:3} == "arm" && ${MACHTYPE} =~ "-linux-" ]]; then

	# clang can fail auto-detecting the host armv7/armv8 cpu on some setups; collect all part numbers
	cxx_uarch_arm

	CXXFLAGS+=(
		-marm
		-mfpu=neon
	)

elif [[ ${HOSTTYPE} == "aarch64" ]]; then

	# clang can fail auto-detecting the host armv8 cpu on some setups; collect all part numbers
	cxx_uarch_arm

elif [[ ${HOSTTYPE} == "x86_64" ]]; then

	CXXFLAGS+=(
# Set -march and -mtune accordingly:
		-march=native
		-mtune=native
# For MIC, set 512-bit alignment:
#		-mmic
#		-DSIMD_ALIGNMENT=64
# Use xmm intrinsics with SSE (rather than autovectorization) to level the
# ground with altivec
#		-DSIMD_AUTOVECT=SIMD_8WAY
# For AVX, disable any SSE which might be enabled by default:
#		-mno-sse
#		-mno-sse2
#		-mno-sse3
#		-mno-ssse3
#		-mno-sse4
#		-mno-sse4.1
#		-mno-sse4.2
#		-mavx
# For AVX with FMA/FMA4 (cumulative to above):
#		-mfma4
#		-mfma
		-DCACHELINE_SIZE=64
	)

elif [[ ${HOSTTYPE} == "powerpc" ]]; then

	UNAME_SUFFIX=`uname -r | grep -o -E -e -[^-]+$`
	CXXFLAGS+=(
		-DCACHELINE_SIZE=32
	)

	if [[ $UNAME_SUFFIX == "-wii" ]]; then

		CXXFLAGS+=(
			-mpowerpc
			-mcpu=750
			-mpaired
			-DSIMD_AUTOVECT=SIMD_2WAY
		)

	elif [[ $UNAME_SUFFIX == "-powerpc" || $UNAME_SUFFIX == "-smp" ]]; then

		CXXFLAGS+=(
			-mpowerpc
			-mcpu=7450
			-maltivec
			-mvrsave
# Do not use autovectorization with altivec - as of 4.6.3 gcc will not use
# splat/permute altivec ops during autovectorization, rendering that useless.
#			-DSIMD_AUTOVECT=SIMD_4WAY
		)
	fi

elif [[ ${HOSTTYPE} == "powerpc64" || ${HOSTTYPE} == "ppc64" ]]; then

	CXXFLAGS+=(
		-mpowerpc64
		-mcpu=powerpc64
		-mtune=power6
		-maltivec
		-mvrsave
# Do not use autovectorization with altivec - as of 4.6.3 gcc will not use
# splat/permute altivec ops during autovectorization, rendering that useless.
#		-DSIMD_AUTOVECT=SIMD_4WAY
		-DCACHELINE_SIZE=64
	)
fi

if [[ $1 == "debug" ]]; then
	CXXFLAGS+=(
		-Wall
		-O0
		-g
		-DDEBUG)
else
	CXXFLAGS+=(
		-funroll-loops
		-O3
		-DNDEBUG)
fi

BUILD_CMD=$CXX" -o "$TARGET" "${CXXFLAGS[@]}" "${SOURCE[@]}" "${LFLAGS[@]}
echo $BUILD_CMD
$BUILD_CMD
