#!/bin/bash

# clang version suffix, e.g. 3.4, 3.5, etc
VERSION=${CLANG_VERSION:-3.5}
BINARY=problem_4
COMMON=../common
SOURCE=(
	$COMMON/platform_glx.cpp
	$COMMON/util_gl.cpp
	$COMMON/prim_mono_view.cpp
	$COMMON/get_file_size.cpp
	problem_6.cpp
	cl_util.cpp
	cl_wrap.cpp
	main_cl.cpp
)
CFLAGS=(
	-std=c++11
	-Wno-bitwise-op-parentheses
	-Wno-logical-op-parentheses
	-Wno-parentheses
	-I$COMMON
	-I..
	-fno-exceptions
	-fno-rtti
	-fdata-sections
	-ffunction-sections
# Enable some optimisations that may or may not be enabled by the global optimisation level of choice in this compiler version
	-Ofast
	-fno-unsafe-math-optimizations
	-fstrict-aliasing
	-fstrict-overflow
	-funroll-loops
	-fomit-frame-pointer
	-DNDEBUG
# Instruct GL headers to properly define their prototypes
	-DGLX_GLXEXT_PROTOTYPES
	-DGLCOREARB_PROTOTYPES
	-DGL_GLEXT_PROTOTYPES
# Enforce fixed time step based on a fixed frame rate
#	-DFRAME_RATE=60
# Case-specific optimisation
	-DMINIMAL_TREE=1
# Show on screen what was rendered
	-DVISUALIZE=1
# Enable tweaks targeting Mesa quirks
#	-DOUTDATED_MESA=1
# Draw octree cells instead of octree content
#	-DDRAW_TREE_CELLS=1
# Clang static code analysis:
#	--analyze
# Compiler quirk 0001: control definition location of routines posing entry points to recursion for more efficient inlining
#	-DCLANG_QUIRK_0001=1
# OpenCL quirk 0001: broken alignment of wide-alignment-type (eg. float4) buffers in __constant space
#	-DOCL_QUIRK_0001=1
# OpenCL quirk 0002: slow select(); even though vector select() and vector (?:) should perform identically, on some hw and stacks select is slower
#	-DOCL_QUIRK_0002=1
# OpenCL quirk 0004: forced O(logN) add-reduction over native add-reduction
#	-DOCL_QUIRK_0004=1
# OpenCL quirk 0005: control child indexing via shuffles at hit-loops
#	-DOCL_QUIRK_0005=1
# OpenCL kernel build full verbosity; macro mandatory
	-DOCL_KERNEL_BUILD_VERBOSE=0
# Use buffer copying rather than buffer mapping when not using interop
	-DOCL_BUFFER_COPY=1
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
# Arm Cortex-A76
#	armv8.2-a
#	cortex-a76
)
LFLAGS=(
# Alias some glibc6 symbols to older ones for better portability
#	-Wa,-defsym,memcpy=memcpy@GLIBC_2.2.5
#	-Wa,-defsym,__sqrtf_finite=__sqrtf_finite@GLIBC_2.2.5
	-lstdc++
	-ldl
	-lrt
	`ldconfig -p | grep -m 1 ^[[:space:]]libGL.so | sed "s/^.\+ //"`
	`ldconfig -p | grep -m 1 ^[[:space:]]libOpenCL.so | sed "s/^.\+ //"`
	-lX11
#	-lpng16
)

BITCODE=( "${SOURCE[@]##*/}" )
BITCODE=( "${BITCODE[@]%.cpp}" )
BITCODE=( "${BITCODE[@]/%/.bc}" )

SOURCE_COUNT=${#SOURCE[@]}


for (( i=0; i < $SOURCE_COUNT; i++ )); do
	(set -x; clang++-${VERSION} -c -flto -emit-llvm ${CFLAGS[@]} -march=${TARGET[0]} ${SOURCE[$i]} -o ${BITCODE[$i]})
done

(set -x; llvm-link-${VERSION} -o _${BINARY}.bc ${BITCODE[@]})

if [[ ${TARGET[1]} == "native" ]]; then
	LLC_TARGET=`llc-${VERSION} --version | grep "Host CPU:" | sed s/^[[:space:]]*"Host CPU:"[[:space:]]*//`
else
	LLC_TARGET=${TARGET[1]}
fi

set -x
opt-${VERSION} -filetype=asm -O3 -data-sections -function-sections -enable-unsafe-fp-math -fp-contract=fast -enable-no-infs-fp-math -enable-no-nans-fp-math -enable-misched -mcpu=${LLC_TARGET} _${BINARY}.bc -o=__${BINARY}.bc

llc-${VERSION} -filetype=obj -O=3 -data-sections -function-sections -enable-unsafe-fp-math -fp-contract=fast -enable-no-infs-fp-math -enable-no-nans-fp-math -enable-misched -mcpu=${LLC_TARGET} __${BINARY}.bc

clang++-${VERSION} -o $BINARY ${LFLAGS[@]} __${BINARY}.o -Wl,--gc-sections
