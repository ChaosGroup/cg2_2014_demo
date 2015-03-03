#!/bin/bash

CC=clang++
TARGET=testvect_simd
SOURCE=(
	testvect_simd.cpp
)
CFLAGS=(
	-pipe
	-fno-exceptions
	-fno-rtti
	-fstrict-aliasing
	-DVECT_SIMD_SSE_DIV_AS_RCP
#	-DVECT_SIMD_SSE_SQRT_DIV_AS_RSQRT
# Use as many worker threads, including the main thread
	-DSIMD_NUM_THREADS=1
# Worker thread affinity stride (for control over physical/logical CPU distribution)
	-DSIMD_THREAD_AFFINITY_STRIDE=1
# Natural alignment of any SIMD type (might be overridden in the architecture-dependant sections below)
	-DSIMD_ALIGNMENT=16
# For the performance test, use matx4 from etal namespace instead of counterpart from simd/scal namespace
#	-DSIMD_ETALON
# Use manual unrolling of innermost loops under -DSIMD_ETALON
#	-DSIMD_ETALON_MANUAL_UNROLL
# Use vector templates from scal namespace in both conformance and performance tests
#	-DSIMD_SCALAR
# Use vector templates from platform-agnostic rend namespace in performance tests
#	-DSIMD_AGNOSTIC
# Produce asm listings instead of binaries
#	-S
)
if [[ $CC == "g++" ]]; then
	CFLAGS+=(
		-ffast-math
	)

elif [[ $CC == "clang++" ]]; then
	CFLAGS+=(
		-ffp-contract=fast
	)

elif [[ $CC == "icpc" ]]; then
	CFLAGS+=(
		-fp-model fast=2
#		-unroll-aggressive
#		-no-opt-prefetch
	)
fi

LFLAGS=(
	-lstdc++
	-lrt
	-lpthread
)

if [[ $HOSTTYPE == "arm" ]]; then

	UNAME_MACHINE=`uname -m`

	if [[ $UNAME_MACHINE == "armv7l" ]]; then

		CFLAGS+=(
			-mfloat-abi=softfp
			-marm
			-march=armv7-a
			-mtune=cortex-a8
			-mcpu=cortex-a8
			-mfpu=neon
# Intrinsics yield better results for now
#			-DSIMD_AUTOVECT=SIMD_4WAY
			-DCACHELINE_SIZE=32
		)
	fi

elif [[ $HOSTTYPE == "x86_64" ]]; then

	CFLAGS+=(
# Set -march and -mtune accordingly:
		-march=native
		-mtune=native
# For SSE, establish a baseline:
#		-mfpmath=sse
#		-mmic
# Use xmm intrinsics with SSE (rather than autovectorization) to level the
# ground with altivec
#		-DSIMD_AUTOVECT=SIMD_4WAY
# For AVX, disable any SSE which might be enabled by default:
#		-mno-sse
#		-mno-sse2
#		-mno-sse3
#		-mno-ssse3
#		-mno-sse4
#		-mno-sse4.1
#		-mno-sse4.2
#		-mavx
#		-DSIMD_AUTOVECT=SIMD_4WAY
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
# Don't use autovectorization with altivec - as of 4.6.3 gcc will not use
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
# Don't use autovectorization with altivec - as of 4.6.3 gcc will not use
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
