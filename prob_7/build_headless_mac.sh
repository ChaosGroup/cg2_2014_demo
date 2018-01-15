#!/bin/bash

# known issue with clang++-3.6 and the lifespan of temporaries, causing multiple failures at unittest
CC=/usr/bin/clang++
BINARY=problem_4
COMMON=../common
SOURCE=(
	$COMMON/get_file_size.cpp
	problem_6.cpp
	cl_util.cpp
	cl_wrap.cpp
	main_cl.cpp
)
CFLAGS=(
# g++-4.8 through 4.9 suffer from a name-mangling conflict in cephes headers; switching to the following ABI version resolves that
#	-fabi-version=6
# g++-6 warns about ignored attributes in template arguments (e.g. __m256 as a typename template arg to simd::native8)
#	-Wno-ignored-attributes
	-ansi
	-Wno-logical-op-parentheses
	-Wno-bitwise-op-parentheses
	-Wno-parentheses
	-I$COMMON
	-I/opt/local/include/libpng16
	-I..
	-pipe
	-fno-exceptions
	-fno-rtti
# Fixed framerate
	-DFRAME_RATE=30
# Case-specific optimisation
	-DMINIMAL_TREE=1
# Draw octree cells instead of octree content
#	-DDRAW_TREE_CELLS=1
# Clang static code analysis:
#	--analyze
# Compiler quirk 0001: control definition location of routines posing entry points to recursion for more efficient inlining
	-DCLANG_QUIRK_0001=1
# Compiler quirk 0002: type size_t is unrelated to same-size type uint*_t
	-DCLANG_QUIRK_0002=1
# OpenCL quirk 0001: broken alignment of wide-alignment-type (eg. float4) buffers in __constant space
	-DOCL_QUIRK_0001=1
# OpenCL kernel build full verbosity; macro mandatory
	-DOCL_KERNEL_BUILD_VERBOSE=0
)
# For non-native or tweaked architecture targets, comment out 'native' and uncomment the correct target architecture and flags
TARGET=(
	native
	native
# AMD Bobcat:
#	btver1
#	btver1
# AMD Jaguar:
#	btver2
#	btver2
# note: Jaguars have 4-wide SIMD, so our avx256 code is not beneficial to them
#	-mno-avx
# Intel Core2
#	core2
#	core2
# Intel Nehalem
#	corei7
#	corei7
# Intel Sandy Bridge
#	corei7-avx
#	corei7-avx
# Intel Ivy Bridge
#	core-avx-i
#	core-avx-i
# ARM Cortex-A57/A72
#	armv8-a
#	cortex-a57
)
LFLAGS=(
# Alias some glibc6 symbols to older ones for better portability
#	-Wa,-defsym,memcpy=memcpy@GLIBC_2.2.5
#	-Wa,-defsym,__sqrtf_finite=__sqrtf_finite@GLIBC_2.2.5
	-lstdc++
	-ldl
	-framework OpenCL
	/opt/local/lib/libpng16.dylib
)

if [[ $1 == "debug" ]]; then
	CFLAGS+=(
		-Wall
		-O0
		-g
		-fstandalone-debug
		-DDEBUG
	)
else
	CFLAGS+=(
# Enable some optimisations that may or may not be enabled by the global optimisation level of choice in this compiler version
		-ffast-math
		-fstrict-aliasing
		-fstrict-overflow
		-funroll-loops
		-fomit-frame-pointer
		-O3
		-flto
		-DNDEBUG
	)
fi

BUILD_CMD=$CC" -o "$BINARY" "${CFLAGS[@]}" -march="${TARGET[0]}" -mtune="${TARGET[@]:1}" "${SOURCE[@]}" "${LFLAGS[@]}
echo $BUILD_CMD
CCC_ANALYZER_CPLUSPLUS=1 $BUILD_CMD
