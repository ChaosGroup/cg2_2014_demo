#!/bin/bash

CC=${CXX:-clang++}
BINARY=problem_4
COMMON=../common
SOURCE=(
	$COMMON/platform_glx.cpp
	$COMMON/util_gl.cpp
	$COMMON/util_eth.cpp
	$COMMON/prim_rgb_view.cpp
	$COMMON/get_file_size.cpp
	problem_7.cpp
	main.cpp
)
CFLAGS=(
	-std=c++11
	-Wno-logical-op-parentheses
	-Wno-bitwise-op-parentheses
	-Wno-parentheses
	-I$COMMON
	-pipe
	-fno-exceptions
	-fno-rtti
# Instruct GL headers to properly define their prototypes
	-DGLX_GLXEXT_PROTOTYPES
	-DGLCOREARB_PROTOTYPES
	-DGL_GLEXT_PROTOTYPES
# Grab every frame to PNG
#	-DFRAMEGRAB=1
# Enforce fixed time step based on a fixed frame rate
	-DFRAME_RATE=60
# Case-specific optimisation
	-DMINIMAL_TREE=1
# Show on screen what was rendered
	-DVISUALIZE=1
# Fixed framebuffer geometry
#	-DFB_RES_FIXED_W=512
#	-DFB_RES_FIXED_H=512
# Vector aliasing control
#	-DVECTBASE_MINIMISE_ALIASING=1
# High-precision ray reciprocal direction
	-DRAY_HIGH_PRECISION_RCP_DIR=1
# Number of workforce threads (normally equating the number of logical cores)
	-DWORKFORCE_NUM_THREADS=`lscpu | grep ^"CPU(s)" | sed s/^[^[:digit:]]*//`
# Make workforce threads sticky (NUMA, etc)
	-DWORKFORCE_THREADS_STICKY=`lscpu | grep ^"Socket(s)" | echo "\`sed s/^[^[:digit:]]*//\` > 1" | bc`
# Colorize the output of individual threads
#	-DCOLORIZE_THREADS=1
# Threading model 'division of labor' alternatives: 0, 1, 2
	-DDIVISION_OF_LABOR_VER=2
# Number of AO rays per pixel
	-DAO_NUM_RAYS=64
# Enable tweaks targeting Mesa quirks
#	-DOUTDATED_MESA=1
# Draw octree cells instead of octree content
#	-DDRAW_TREE_CELLS=1
# Clang static code analysis:
#	--analyze
# Compiler quirk 0001: control definition location of routines posing entry points to recursion for more efficient inlining
	-DCLANG_QUIRK_0001=1
# Two-node distributed rendering by nodes on the same LAN; a node can be either a core or a supplement -- uncomment one; number gives the ratio of core / supplement workload, and is the same for both
#	-DDR_CORE=2
#	-DDR_SUPPLEMENT=2
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
)
LFLAGS=(
# Alias some glibc6 symbols to older ones for better portability
#	-Wa,-defsym,memcpy=memcpy@GLIBC_2.2.5
#	-Wa,-defsym,__sqrtf_finite=__sqrtf_finite@GLIBC_2.2.5
	-lstdc++
	-ldl
	-lrt
	`ldconfig -p | grep -m 1 ^[[:space:]]libGL.so | sed "s/^.\+ //"`
	-lX11
	-lpthread
	-lpng16
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
		-Ofast
		-fno-unsafe-math-optimizations
		-fstrict-aliasing
		-fstrict-overflow
		-funroll-loops
		-fomit-frame-pointer
		-flto
		-DNDEBUG
	)
fi

BUILD_CMD=$CC" -o "$BINARY" "${CFLAGS[@]}" -march="${TARGET[0]}" -mtune="${TARGET[@]:1}" "${SOURCE[@]}" "${LFLAGS[@]}
echo $BUILD_CMD
CCC_ANALYZER_CPLUSPLUS=1 $BUILD_CMD
