#!/bin/bash

CC=clang++-3.5
TARGET=testvect_simd
SOURCE=(
	testvect_simd.cpp
)
LFLAGS=(
	-lstdc++
	-lpthread
)

if [[ ${MACHTYPE} =~ "-apple-darwin" ]]; then
	NUM_LOGICAL_CORES=`sysctl hw.ncpu | sed s/^[^[:digit:]]*//`
	SOURCE+=(
		pthread_barrier.cpp
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
CFLAGS=(
	-pipe
	-fno-exceptions
	-fno-rtti
	-fstrict-aliasing
# Use reciprocals instead of division
	-DVECT_SIMD_DIV_AS_RCP
# Use reciprocal sqrt instead if sqrt and division
#	-DVECT_SIMD_SQRT_DIV_AS_RSQRT
# Perform arithmetic conformace tests as well
#	-DSIMD_TEST_CONFORMANCE
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
# Alternative etalon v3 (in combination with SIMD_ETALON above)
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
CC_FILENAME=${CC##*/}
if [[ ${CC_FILENAME:0:3} == "g++" ]]; then
	CFLAGS+=(
		-ffast-math
	)

elif [[ ${CC_FILENAME:0:7} == "clang++" ]]; then
	CFLAGS+=(
		-ffp-contract=fast
	)

elif [[ ${CC_FILENAME:0:4} == "icpc" ]]; then
	CFLAGS+=(
		-fp-model fast=2
		-opt-prefetch=0
		-opt-streaming-cache-evict=0
	)
fi

if [[ $HOSTTYPE == "arm" ]]; then

	# clang can fail auto-detecting the host armv7/armv8 cpu on some setups; collect all part numbers
	UARCH=`cat /proc/cpuinfo | grep "^CPU part" | sed s/^[^[:digit:]]*//`

	# in order of preference, in case of big.LITTLE (armv7 and armv8 lumped together)
	if   [ `echo $UARCH | grep -c 0xd09` -ne 0 ]; then
		CFLAGS+=(
			-march=armv8-a
			-mtune=cortex-a73
		)
	elif [ `echo $UARCH | grep -c 0xd08` -ne 0 ]; then
		CFLAGS+=(
			-march=armv8-a
			-mtune=cortex-a72
		)
	elif [ `echo $UARCH | grep -c 0xd07` -ne 0 ]; then
		CFLAGS+=(
			-march=armv8-a
			-mtune=cortex-a57
		)
	elif [ `echo $UARCH | grep -c 0xd03` -ne 0 ]; then
		CFLAGS+=(
			-march=armv8-a
			-mtune=cortex-a53
		)
	elif [ `echo $UARCH | grep -c 0xc0f` -ne 0 ]; then
		CFLAGS+=(
			-march=armv7-a
			-mtune=cortex-a15
		)
	elif [ `echo $UARCH | grep -c 0xc0e` -ne 0 ]; then
		CFLAGS+=(
			-march=armv7-a
			-mtune=cortex-a17
		)
	elif [ `echo $UARCH | grep -c 0xc09` -ne 0 ]; then
		CFLAGS+=(
			-march=armv7-a
			-mtune=cortex-a9
		)
	elif [ `echo $UARCH | grep -c 0xc08` -ne 0 ]; then
		CFLAGS+=(
			-march=armv7-a
			-mtune=cortex-a8
		)
	elif [ `echo $UARCH | grep -c 0xc07` -ne 0 ]; then
		CFLAGS+=(
			-march=armv7-a
			-mtune=cortex-a7
		)
	fi

	UNAME_MACHINE=`uname -m`

	if [[ $UNAME_MACHINE == "armv7l" ]]; then

		CFLAGS+=(
			-marm
			-mfpu=neon
			-DCACHELINE_SIZE=32
		)
	elif [[ $UNAME_MACHINE == "aarch64" ]]; then # for armv8 devices with aarch64 kernels + armv7 userspaces

		CFLAGS+=(
			-marm
			-mfpu=neon
			-DCACHELINE_SIZE=64
		)
	fi

elif [[ $HOSTTYPE == "aarch64" ]]; then

	# clang can fail auto-detecting the host armv8 cpu on some setups; collect all part numbers
	UARCH=`cat /proc/cpuinfo | grep "^CPU part" | sed s/^[^[:digit:]]*//`

	# choose in order of preference, in case of big.LITTLE
	if   [ `echo $UARCH | grep -c 0xd09` -ne 0 ]; then
		CFLAGS+=(
			-mtune=cortex-a73
		)
	elif [ `echo $UARCH | grep -c 0xd08` -ne 0 ]; then
		CFLAGS+=(
			-mtune=cortex-a72
		)
	elif [ `echo $UARCH | grep -c 0xd07` -ne 0 ]; then
		CFLAGS+=(
			-mtune=cortex-a57
		)
	elif [ `echo $UARCH | grep -c 0xd03` -ne 0 ]; then
		CFLAGS+=(
			-mtune=cortex-a53
		)
	fi

	CFLAGS+=(
		-march=armv8-a
		-DCACHELINE_SIZE=64
	)

elif [[ $HOSTTYPE == "x86_64" ]]; then

	CFLAGS+=(
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

elif [[ $HOSTTYPE == "powerpc" ]]; then

	UNAME_SUFFIX=`uname -r | grep -o -E -e -[^-]+$`
	CFLAGS+=(
		-DCACHELINE_SIZE=32
	)

	if [[ $UNAME_SUFFIX == "-wii" ]]; then

		CFLAGS+=(
			-mpowerpc
			-mcpu=750
			-mpaired
			-DSIMD_AUTOVECT=SIMD_2WAY
		)

	elif [[ $UNAME_SUFFIX == "-powerpc" || $UNAME_SUFFIX == "-smp" ]]; then

		CFLAGS+=(
			-mpowerpc
			-mcpu=7450
			-maltivec
			-mvrsave
# Do not use autovectorization with altivec - as of 4.6.3 gcc will not use
# splat/permute altivec ops during autovectorization, rendering that useless.
#			-DSIMD_AUTOVECT=SIMD_4WAY
		)
	fi

elif [[ $HOSTTYPE == "powerpc64" || $HOSTTYPE == "ppc64" ]]; then

	CFLAGS+=(
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
	CFLAGS+=(
		-Wall
		-O0
		-g
		-DDEBUG)
else
	CFLAGS+=(
		-funroll-loops
		-O3
		-DNDEBUG)
fi

BUILD_CMD=$CC" -o "$TARGET" "${CFLAGS[@]}" "${SOURCE[@]}" "${LFLAGS[@]}
echo $BUILD_CMD
$BUILD_CMD
